"""read_doc: passage retrieval over the SEC PDFs, returning page references."""
import math
import re
from collections import Counter
from functools import lru_cache

from pypdf import PdfReader

from quantaccelerator import build
from quantaccelerator.datasets import active

DOCS = dict(active().docs)  # doc id -> .pdf or .txt for the active case study
TXT_PAGE_CHARS = 2500       # plain-text docs (fetched web pages) are cut into pseudo-pages of this size
from quantaccelerator.tools.registry import tool

TOKEN = re.compile(r"[A-Za-z0-9_]+")
HEADING = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+([A-Z][A-Za-z0-9_ ./&,()-]{2,60}?)\s*$")
TOC_LEADER = "....."
XREF = re.compile(r"Appendix\s+section\s+(\d+(?:\.\d+)+)", re.I)
MAX_FOLLOWED = 2


@lru_cache(maxsize=None)
def passages(doc: str) -> list[dict]:
    """Split each page into ~600-char passages on line boundaries.

    Each passage records the numbered section it falls in (e.g. '6.2 Trans Code List'), so that the later passages
    of a long code list are still found by the section's name; a new section starts a new passage.
    """
    out, section = [], ""
    for i, page_text in enumerate(_pages(DOCS[doc]), start=1):
        buf = ""
        for line in page_text.splitlines():
            m = HEADING.match(line) if TOC_LEADER not in line else None
            if m and buf.strip():
                out.append({"doc": doc, "page": i, "section": section, "text": buf.strip()})
                buf = ""
            if m:
                section = f"{m.group(1)} {m.group(2).strip()}"
            buf += line + "\n"
            if len(buf) > 600:
                out.append({"doc": doc, "page": i, "section": section, "text": buf.strip()})
                buf = line + "\n"  # one-line overlap
        if buf.strip():
            out.append({"doc": doc, "page": i, "section": section, "text": buf.strip()})
    return out


def _pages(path) -> list[str]:
    """Page texts of a PDF, or ~TXT_PAGE_CHARS pseudo-pages (cut on line boundaries) of a .txt document."""
    if str(path).endswith(".txt"):
        pages, buf = [], ""
        for line in open(path, encoding="utf-8").read().splitlines():
            if len(buf) + len(line) > TXT_PAGE_CHARS and buf:
                pages.append(buf)
                buf = ""
            buf += line + "\n"
        return pages + ([buf] if buf.strip() else [])
    return [p.extract_text() or "" for p in PdfReader(path).pages]


def _section_match(query: str, section: str) -> float:
    """Share of the section title's words named in the query ('Appendix 6.2 Trans Code List' -> 1.0)."""
    words = set(_tokens(re.sub(r"^[\d.]+\s*", "", section)))
    return len(words & set(_tokens(query))) / len(words) if words else 0.0


def _tokens(s: str) -> list[str]:
    """Lowercased tokens; identifiers like TRANS_CODE also yield their parts ('trans', 'code')."""
    out = []
    for t in TOKEN.findall(s):
        t = t.lower()
        out.append(t)
        if "_" in t:
            out += [x for x in t.split("_") if x]
    return out


def search(query: str, doc: str | None = None, k: int = 3) -> list[dict]:
    corpus = [p for d in ([doc] if doc else DOCS) for p in passages(d)]
    toks = [Counter(_tokens(p["section"] + " " + p["text"])) for p in corpus]
    df = Counter(t for c in toks for t in c)
    n, avg = len(corpus), sum(sum(c.values()) for c in toks) / len(corpus)
    q = set(_tokens(query))
    scores, named = [], []
    for p, c in zip(corpus, toks):
        L = sum(c.values())
        s = sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * c[t] * 2.2 / (c[t] + 1.2 * (0.25 + 0.75 * L / avg))
                for t in q if t in c)
        num = p["section"].split(" ")[0]
        if num and not build.prototype() and (_section_match(query, p["section"]) == 1.0 or re.search(rf"(?<![\d.]){re.escape(num)}(?![\d.])", query)):
            s += 20.0  # the query names this section ("6.2", "Trans Code List")
            named.append(p["section"])
        if p["text"].count(TOC_LEADER) >= 3:
            s *= 0.2   # table of contents: mentions every section, defines nothing
        scores.append(s)
    def whole(p):
        lines = []
        for q in corpus:
            if (q["doc"], q["page"], q["section"]) == (p["doc"], p["page"], p["section"]):
                lines += [ln for ln in q["text"].splitlines() if not lines or ln != lines[-1]]
        return {**p, "text": "\n".join(lines)}

    out, seen = [], set()
    for i in sorted(range(n), key=lambda i: -scores[i]):
        p = corpus[i]
        if scores[i] <= 0 or len(out) >= k:
            break
        if p["section"] in named:
            # a named section (typically a code list) is returned whole, one result per page
            if (p["doc"], p["page"], p["section"]) in seen:
                continue
            seen.add((p["doc"], p["page"], p["section"]))
            p = whole(p)
        out.append({**p, "score": round(scores[i], 2)})
    if build.prototype():
        return out
    # follow cross-references ("values ... are listed in the Appendix section 6.2 Trans Code List") like a reader would
    followed = 0
    for hit in list(out):
        for num in dict.fromkeys(XREF.findall(hit["text"])):
            for p in corpus:
                key = (p["doc"], p["page"], p["section"])
                if p["doc"] == hit["doc"] and p["section"].split(" ")[0] == num and key not in seen \
                        and followed < MAX_FOLLOWED:
                    seen.add(key)
                    followed += 1
                    out.append({**whole(p), "score": None, "followed_from": f"{hit['doc']} p{hit['page']}"})
    return out


@tool("read_doc", "Search the documentation of the dataset under audit and return the best-matching passages with "
      f"page numbers. Docs: {active().doc_help}" + ("" if build.prototype() else " References to appendix sections "
      "(code lists) are followed automatically and returned too.") + " Cite as '<doc> p<page>'.",
      {"type": "object", "properties": {"query": {"type": "string"},
                                        "doc": {"type": "string", "enum": list(DOCS)},
                                        "k": {"type": "integer", "minimum": 2, "maximum": 5}},
       "required": ["query"]},
      inputs=lambda query, doc=None, k=3: [DOCS[d] for d in ([doc] if doc else DOCS)])
def read_doc(query: str, doc: str | None = None, k: int = 3):
    return search(query, doc, max(2, min(int(k), 5)))
