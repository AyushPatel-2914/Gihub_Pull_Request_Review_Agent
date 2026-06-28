"""
src/server/webhook_server.py
─────────────────────────────────────────────────────────────
FastAPI Webhook Server — Production Deployment (Phase 5B)

WHAT THIS IS:
  Instead of running inside GitHub Actions, this is a standalone
  web service that GitHub calls directly via webhooks whenever a
  PR is opened or updated. This is how you'd deploy the agent at
  scale across many repos without GitHub Actions minutes.

THE ASYNC PATTERN (critical for webhooks):
  GitHub expects a webhook to respond within ~10 seconds, but a
  full review takes 30-90 seconds. So we:
    1. Receive the webhook
    2. Verify it's really from GitHub (signature check)
    3. Immediately return 202 Accepted
    4. Run the actual review in a BACKGROUND task

  This prevents GitHub from timing out and retrying.

SECURITY:
  GitHub signs every webhook with a shared secret (HMAC-SHA256).
  We verify that signature so attackers can't trigger fake reviews.

ENDPOINTS:
  GET  /              → health check
  GET  /health        → detailed health/readiness
  POST /webhook       → GitHub webhook receiver

DEPLOYMENT:
  Local:      uvicorn server.webhook_server:app --reload --port 8000
  Production: uvicorn server.webhook_server:app --host 0.0.0.0 --port 8000
  Then expose via ngrok (testing) or deploy to Render/Railway/Fly.io.
─────────────────────────────────────────────────────────────
"""

import os
import sys
import hmac
import hashlib
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Header
from fastapi.responses import JSONResponse

from agent_core import run_review

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("webhook_server")

# ── Configuration ──────────────────────────────────────────────
# The webhook secret you set in GitHub repo → Settings → Webhooks
WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")

# Events we actually review
REVIEWABLE_ACTIONS = {"opened", "synchronize", "reopened"}

app = FastAPI(
    title="PR Review Agent",
    description="Self-Healing RAG PR Review Agent — webhook server",
    version="5.0",
)


# ──────────────────────────────────────────────────────────────
# SIGNATURE VERIFICATION
# ──────────────────────────────────────────────────────────────

def verify_signature(payload_body: bytes, signature_header: str) -> bool:
    """
    Verifies the webhook came from GitHub using HMAC-SHA256.

    GitHub sends the signature in the X-Hub-Signature-256 header
    as 'sha256=<hexdigest>'. We recompute it with our shared secret
    and compare. A mismatch means the request is forged or corrupted.

    Args:
      payload_body     : raw request body bytes
      signature_header : value of X-Hub-Signature-256 header

    Returns:
      True if the signature is valid (or if no secret is configured,
      in which case verification is skipped with a warning).
    """
    if not WEBHOOK_SECRET:
        logger.warning(
            "GITHUB_WEBHOOK_SECRET not set — skipping signature verification. "
            "Set it in production!"
        )
        return True

    if not signature_header:
        logger.error("Missing X-Hub-Signature-256 header")
        return False

    try:
        algo, received_sig = signature_header.split("=", 1)
    except ValueError:
        return False

    if algo != "sha256":
        return False

    expected_sig = hmac.new(
        WEBHOOK_SECRET.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    # Constant-time comparison prevents timing attacks
    return hmac.compare_digest(expected_sig, received_sig)


# ──────────────────────────────────────────────────────────────
# BACKGROUND REVIEW TASK
# ──────────────────────────────────────────────────────────────

def run_review_task(repo_full_name: str, pr_number: int):
    """
    Wrapper that runs the review and logs the outcome.
    Executed in the background so the webhook returns immediately.
    Any exception is caught and logged — a failed review must never
    crash the server.
    """
    try:
        logger.info("Background review starting | %s PR #%d", repo_full_name, pr_number)
        result = run_review(repo_full_name, pr_number)
        logger.info("Background review complete | %s", result)
    except Exception as e:
        logger.exception("Background review failed for %s PR #%d: %s",
                         repo_full_name, pr_number, e)


# ──────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Simple health check — confirms the server is up."""
    return {"status": "ok", "service": "PR Review Agent", "version": "5.0"}


@app.get("/health")
async def health():
    """
    Detailed readiness check. Reports which required secrets are
    configured (without exposing their values).
    """
    return {
        "status": "healthy",
        "config": {
            "github_token": bool(os.environ.get("GITHUB_TOKEN")),
            "groq_api_key": bool(os.environ.get("GROQ_API_KEY")),
            "pinecone_api_key": bool(os.environ.get("PINECONE_API_KEY")),
            "webhook_secret": bool(WEBHOOK_SECRET),
        },
    }


@app.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str = Header(None),
    x_hub_signature_256: str = Header(None),
):
    """
    GitHub webhook receiver.

    Flow:
      1. Read the raw body
      2. Verify the HMAC signature
      3. Check the event type and action
      4. Schedule the review in the background
      5. Return 202 immediately
    """
    # Read raw body for signature verification
    body = await request.body()

    # ── Verify signature ──────────────────────────────────────
    if not verify_signature(body, x_hub_signature_256):
        logger.error("Webhook signature verification FAILED")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # ── Parse payload ─────────────────────────────────────────
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # ── Only handle pull_request events ───────────────────────
    if x_github_event == "ping":
        logger.info("Received ping event — webhook is configured correctly")
        return {"status": "pong"}

    if x_github_event != "pull_request":
        logger.info("Ignoring event type: %s", x_github_event)
        return {"status": "ignored", "reason": f"event '{x_github_event}' not handled"}

    action = payload.get("action", "")
    if action not in REVIEWABLE_ACTIONS:
        logger.info("Ignoring PR action: %s", action)
        return {"status": "ignored", "reason": f"action '{action}' not reviewed"}

    # ── Extract PR info ───────────────────────────────────────
    try:
        repo_full_name = payload["repository"]["full_name"]
        pr_number = payload["pull_request"]["number"]
    except KeyError as e:
        logger.error("Malformed payload, missing key: %s", e)
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")

    # ── Schedule background review ────────────────────────────
    logger.info(
        "Accepted webhook | %s PR #%d (action=%s) — scheduling review",
        repo_full_name, pr_number, action,
    )
    background_tasks.add_task(run_review_task, repo_full_name, pr_number)

    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "repo": repo_full_name,
            "pr_number": pr_number,
            "message": "Review scheduled in background",
        },
    )
