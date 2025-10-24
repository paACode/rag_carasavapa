#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_gold.py – für CSV mit Spalten: ID, Resume_str, Resume_html, Category
- Auto-Delimiter (Komma/Semikolon) via csv.Sniffer
- Spaltennamen case-insensitive
- Nimmt HTML aus 'Resume_html' bevorzugt, sonst 'Resume_str'
- Schreibt canonical text und leere Gold-Templates
"""

import argparse, csv, json, re, io
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit("Bitte zuerst installieren: pip install beautifulsoup4")

def load_ids(path: str | None):
    if not path: return None
    ids = set()
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                ids.add(ln)
    return ids

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]): tag.decompose()
    for br in soup.find_all(["br"]): br.replace_with("\n")
    for blk in soup.find_all(["p","li","div","section","article","header","footer"]):
        if not blk.text.endswith("\n"): blk.append("\n")
    text = soup.get_text()
    text = text.replace("\r\n","\n").replace("\r","\n").replace("\t"," ")
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    text = re.sub(r"[ \u00A0]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def plain_normalize(txt: str) -> str:
    t = (txt or "").replace("\r\n","\n").replace("\r","\n").replace("\t"," ")
    t = "\n".join(ln.rstrip() for ln in t.split("\n"))
    t = re.sub(r"[ \u00A0]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def write_gold_template(path: Path, cv_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): return
    tpl = {"cv_id": cv_id, "gold_extractions": []}
    path.write_text(json.dumps(tpl, ensure_ascii=False, indent=2), encoding="utf-8")

def sniff_delimiter(csv_path: Path, override: str | None = None) -> str:
    if override: return override
    sample = csv_path.read_text(encoding="utf-8", errors="ignore")[:10000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","|","\t"])
        return dialect.delimiter
    except Exception:
        return ","  # Fallback

def map_fields(fieldnames):
    lower = { (fn or "").strip().lower(): fn for fn in fieldnames }
    get = lambda key: lower.get(key.lower())
    id_col = get("id")
    html_col = get("resume_html")
    str_col = get("resume_str")
    cat_col = get("category")
    missing = [k for k,v in [("ID",id_col),("Resume_html",html_col),("Resume_str",str_col)] if v is None]
    if missing:
        raise SystemExit(f"Spalten fehlen in CSV: {', '.join(missing)}. Gefunden: {list(fieldnames)}")
    return id_col, str_col, html_col, cat_col

def from_csv(csv_path: Path, out_txt_dir: Path, out_gold_dir: Path,
             ids_allow: set[str] | None, delimiter: str | None, limit: int | None):
    dlm = sniff_delimiter(csv_path, delimiter)
    with csv_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        while first and not first.strip():
            first = f.readline()
        if not first:
            raise SystemExit("Leere CSV?")
        content = first + f.read()
    import io as _io
    buf = _io.StringIO(content)

    reader = csv.reader(buf, delimiter=dlm)
    try:
        headers = next(reader)
    except StopIteration:
        raise SystemExit("Keine Header-Zeile gefunden.")
    headers = [h.strip() for h in headers]
    id_col, str_col, html_col, cat_col = map_fields(headers)

    buf.seek(0)
    r = csv.DictReader(buf, fieldnames=headers, delimiter=dlm)
    next(r)  # header überspringen

    count_out = 0
    line_no = 1
    for row in r:
        line_no += 1
        try:
            cv_id = (row.get(id_col) or "").strip()
            if not cv_id:
                continue
            if ids_allow and cv_id not in ids_allow:
                continue

            html = row.get(html_col) or ""
            txt  = row.get(str_col) or ""
            if html and len(html) < 20 and txt and len(txt) > 200:
                if "<" in txt and ">" in txt:
                    html, txt = txt, html

            if html:
                canonical = html_to_text(html)
            elif txt:
                canonical = plain_normalize(txt)
            else:
                continue

            write_text(out_txt_dir / f"{cv_id}.txt", canonical)
            write_gold_template(out_gold_dir / f"{cv_id}_gold.json", cv_id)

            count_out += 1
            if limit and count_out >= limit:
                break
        except Exception as e:
            print(f"[WARN] Zeile {line_no}: {e}; Überspringe.")
            continue

    print(f"✅ Fertig: {count_out} CVs geschrieben. Delimiter erkannt: '{dlm}'")

def main():
    ap = argparse.ArgumentParser(description="CVs normalisieren (HTML bevorzugt) & Gold-Templates anlegen")
    ap.add_argument("--csv", required=True, help="Pfad zu CSV mit Spalten: ID, Resume_str, Resume_html, Category")
    ap.add_argument("--ids", help="Pfad zu Datei mit erlaubten IDs (eine pro Zeile)")
    ap.add_argument("--delimiter", help="Optional: CSV-Delimiter erzwingen (',' oder ';')")
    ap.add_argument("--limit", type=int, help="Optional: max. Anzahl CVs verarbeiten")
    ap.add_argument("--out-text", default="data/canonical_text")
    ap.add_argument("--out-gold", default="data/gold")
    args = ap.parse_args()

    ids_allow = load_ids(args.ids)
    from_csv(
        Path(args.csv),
        Path(args.out_text),
        Path(args.out_gold),
        ids_allow=ids_allow,
        delimiter=args.delimiter,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
