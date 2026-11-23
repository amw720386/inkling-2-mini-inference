#!/usr/bin/env python3
"""Fail on local links and images in README.md and docs/*.md that point at files missing from the tree.

    python3 docs/media/src/check_links.py [repo_root]

Checks Markdown links and images ([x](path), ![x](path), [x]: path) and HTML src / srcset / href attributes,
outside fenced code blocks and inline code. Paths resolve relative to the file that contains them (a leading /
means the repo root). External URLs and pure #anchors are skipped. Standard library only.
"""
import os
import re
import sys

MD_LINK = re.compile(r"!?\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
MD_REF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+\"[^\"]*\")?\s*$")
HTML_ATTR = re.compile(r"""\b(?:src|href|srcset)\s*=\s*["']([^"']+)["']""", re.I)
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.I)


def targets(text):
    """Yield (line number, target) for every link target outside code."""
    fence = None
    for n, line in enumerate(text.splitlines(), 1):
        m = re.match(r"^\s*(`{3,}|~{3,})", line)
        if m:
            mark = m.group(1)[0]
            if fence is None:
                fence = mark
            elif fence == mark:
                fence = None
            continue
        if fence:
            continue
        line = re.sub(r"`[^`]*`", "", line)            # inline code
        for rx in (MD_LINK, MD_REF, HTML_ATTR):
            for m in rx.finditer(line):
                raw = m.group(1)
                # srcset may list "a.png 1x, b.png 2x"
                for part in (raw.split(",") if "srcset" in m.group(0).lower() else [raw]):
                    t = part.strip().split()[0] if part.strip() else ""
                    if t:
                        yield n, t


def check(root):
    root = os.path.abspath(root)
    files = [os.path.join(root, "README.md")]
    docs = os.path.join(root, "docs")
    if os.path.isdir(docs):
        files += sorted(os.path.join(docs, f) for f in os.listdir(docs) if f.endswith(".md"))
    missing, checked = [], 0
    for f in files:
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        for n, t in targets(text):
            if EXTERNAL.match(t) or t.startswith("#"):
                continue
            path = t.split("#", 1)[0].split("?", 1)[0]
            if not path:
                continue
            full = os.path.join(root, path.lstrip("/")) if path.startswith("/") else os.path.join(os.path.dirname(f), path)
            checked += 1
            if not os.path.exists(os.path.normpath(full)):
                missing.append(f"{os.path.relpath(f, root)}:{n}: {t}")
    return files, checked, missing


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    files, checked, missing = check(root)
    print(f"{len(files)} files, {checked} local links checked, {len(missing)} missing")
    for m in missing:
        print("  missing:", m)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
