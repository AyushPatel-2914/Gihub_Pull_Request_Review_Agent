import logging, re
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

@dataclass
class BM25Result:
    content: str
    filepath: str
    score: float
    symbol_name: str = ""
    start_line: int = 0
    end_line: int = 0

class BM25Index:
    def __init__(self):
        self.corpus = []
        self.tokenized = []
        self.bm25 = None
        self._built = False

    def build(self, chunks):
        if not chunks:
            return
        self.corpus = chunks
        self.tokenized = [self._tokenize(chunk.get("content", "")) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        self._built = True

    def search(self, query, top_k=5):
        if not self._built:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.corpus[idx]
            results.append(BM25Result(content=chunk.get("content", ""), filepath=chunk.get("filepath", ""), score=float(scores[idx]), symbol_name=chunk.get("symbol_name", ""), start_line=chunk.get("start_line", 0), end_line=chunk.get("end_line", 0)))
        return results

    def _tokenize(self, text):
        if not text:
            return []
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 1]

def extract_symbols_from_diff(diff):
    symbols = set()
    added_lines = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    added_content = "\n".join(added_lines)
    symbols.update(re.findall(r"(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", added_content))
    symbols.update(re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", added_content))
    symbols.update(re.findall(r"(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", added_content))
    symbols.update(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", added_content))
    stop_words = {"if","else","elif","for","while","try","except","finally","with","as","return","yield","lambda","and","or","not","in","is","True","False","None","self","cls","def","class","import","from","pass","break","continue","raise","del","global","nonlocal","assert","async","await","print","len","range","str","int","float","list","dict","set","tuple","type"}
    return [s for s in symbols if s.lower() not in stop_words]
