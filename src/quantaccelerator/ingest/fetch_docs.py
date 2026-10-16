"""Fetch public web documentation for grounding: html as served plus readable text, recorded in a manifest.

Usage: python -m quantaccelerator.ingest.fetch_docs <name>=<url> [...]  -> data/external_docs/<name>.{html,txt}
"""
import datetime as dt
import hashlib
import html
import json
import re
import sys
import urllib.request

from quantaccelerator.paths import EXTERNAL_DOCS

USER_AGENT = "Mozilla/5.0 (academic research; QuantAccelerator documentation fetch)"


def to_text(page: str) -> str:
    main = re.search(r"(?is)<main\b.*?</main>", page)  # skip site navigation when the page marks its content
    page = main.group(0) if main else page
    page = re.sub(r"(?is)<(script|style|noscript|svg)\b.*?</\1>", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|section)>", "\n", page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    lines = (re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if ln)


def fetch(name: str, url: str) -> dict:
    EXTERNAL_DOCS.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    raw = urllib.request.urlopen(req, timeout=60).read()
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out = {}
    for ext, data in (("html", raw), ("txt", to_text(raw.decode("utf-8", "replace")).encode())):
        path = EXTERNAL_DOCS / f"{name}.{ext}"
        path.write_bytes(data)
        rec = {"url": url, "path": f"data/external_docs/{path.name}", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "fetched_utc": now}
        with open(EXTERNAL_DOCS / "manifest.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        out[ext] = rec
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, _, url = arg.partition("=")
        r = fetch(name, url)
        print(name, r["txt"]["bytes"], "bytes of text")
