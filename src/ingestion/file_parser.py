import ast, json, logging, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
MAX_CHUNK_CHARS = 6000
OVERLAP_CHARS = 200

SKIP_DIRS = {".git", ".github", "__pycache__", "node_modules", ".venv", "venv", "env", "dist", "build", ".next", ".nuxt", ".pytest_cache", ".mypy_cache", "migrations", ".ruff_cache"}
SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip", ".tar", ".gz", ".exe", ".bin", ".pyc", ".pyo", ".so", ".dll", ".class", ".lock"}

@dataclass
class CodeChunk:
    chunk_id: str
    content: str
    filepath: str
    file_type: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_pinecone_metadata(self):
        return {"filepath": self.filepath, "file_type": self.file_type, "start_line": self.start_line, "end_line": self.end_line, "symbol_name": self.symbol_name or "", "content": self.content[:2000], **self.metadata}

def detect_file_type(filepath):
    suffix = Path(filepath).suffix.lower()
    mapping = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby", ".cpp": "cpp", ".c": "c", ".h": "c", ".cs": "csharp", ".php": "php", ".swift": "swift", ".kt": "kotlin", ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".html": "html", ".htm": "html", ".css": "css", ".scss": "css", ".sql": "sql", ".ipynb": "notebook", ".xml": "xml"}
    if Path(filepath).name.lower() == "dockerfile":
        return "dockerfile"
    return mapping.get(suffix, "unknown")

def _chunk_sliding_window(content, filepath, file_type, namespace):
    chunks = []
    lines = content.splitlines()
    total_chars = 0
    chunk_start_line = 1
    current_lines = []
    chunk_idx = 0
    for line_num, line in enumerate(lines, start=1):
        current_lines.append(line)
        total_chars += len(line) + 1
        if total_chars >= MAX_CHUNK_CHARS:
            chunk_content = "\n".join(current_lines)
            chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{chunk_idx}", content=chunk_content, filepath=filepath, file_type=file_type, start_line=chunk_start_line, end_line=line_num, metadata={"namespace": namespace}))
            chunk_idx += 1
            overlap_lines = []
            overlap_chars = 0
            for ol in reversed(current_lines):
                if overlap_chars + len(ol) > OVERLAP_CHARS: break
                overlap_lines.insert(0, ol)
                overlap_chars += len(ol) + 1
            current_lines = overlap_lines
            chunk_start_line = line_num - len(overlap_lines) + 1
            total_chars = overlap_chars
    if current_lines:
        chunk_content = "\n".join(current_lines)
        if chunk_content.strip():
            chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{chunk_idx}", content=chunk_content, filepath=filepath, file_type=file_type, start_line=chunk_start_line, end_line=len(lines), metadata={"namespace": namespace}))
    return chunks

def _chunk_python(content, filepath, namespace):
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_sliding_window(content, filepath, "python", namespace)
    lines = content.splitlines()
    chunks = []
    top_level_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(n, "col_offset", 0) == 0]
    top_level_nodes.sort(key=lambda n: n.lineno)
    if not top_level_nodes:
        return _chunk_sliding_window(content, filepath, "python", namespace)
    for idx, node in enumerate(top_level_nodes):
        start = node.lineno - 1
        end = node.end_lineno
        chunk_content = "\n".join(lines[start:end])
        chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{idx}", content=chunk_content, filepath=filepath, file_type="python", start_line=start+1, end_line=end, symbol_name=node.name, metadata={"namespace": namespace}))
    return chunks

def _chunk_markdown(content, filepath, namespace):
    lines = content.splitlines()
    chunks = []
    current_section_lines = []
    current_start = 1
    section_idx = 0
    for i, line in enumerate(lines, start=1):
        if (line.startswith("## ") or line.startswith("# ")) and current_section_lines:
            section_content = "\n".join(current_section_lines)
            if section_content.strip():
                chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{section_idx}", content=section_content, filepath=filepath, file_type="markdown", start_line=current_start, end_line=i-1, symbol_name=current_section_lines[0].lstrip("#").strip() if current_section_lines else None, metadata={"namespace": namespace}))
                section_idx += 1
            current_section_lines = [line]
            current_start = i
        else:
            current_section_lines.append(line)
    if current_section_lines:
        section_content = "\n".join(current_section_lines)
        if section_content.strip():
            chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{section_idx}", content=section_content, filepath=filepath, file_type="markdown", start_line=current_start, end_line=len(lines), symbol_name=current_section_lines[0].lstrip("#").strip() if current_section_lines else None, metadata={"namespace": namespace}))
    return chunks if chunks else _chunk_sliding_window(content, filepath, "markdown", namespace)

def _chunk_json(content, filepath, namespace):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _chunk_sliding_window(content, filepath, "json", namespace)
    if not isinstance(parsed, dict):
        return _chunk_sliding_window(content, filepath, "json", namespace)
    chunks = []
    for idx, (key, value) in enumerate(parsed.items()):
        chunks.append(CodeChunk(chunk_id=f"{namespace}::{filepath}::{idx}", content=json.dumps({key: value}, indent=2), filepath=filepath, file_type="json", start_line=1, end_line=1, symbol_name=key, metadata={"namespace": namespace}))
    return chunks if chunks else _chunk_sliding_window(content, filepath, "json", namespace)

def parse_file(filepath, content, namespace):
    if not content or not content.strip():
        return []
    file_type = detect_file_type(filepath)
    if Path(filepath).suffix.lower() in SKIP_EXTENSIONS:
        return []
    if file_type == "python":
        return _chunk_python(content, filepath, namespace)
    elif file_type in ("markdown", "rst", "text"):
        return _chunk_markdown(content, filepath, namespace)
    elif file_type == "json":
        return _chunk_json(content, filepath, namespace)
    else:
        return _chunk_sliding_window(content, filepath, file_type, namespace)
