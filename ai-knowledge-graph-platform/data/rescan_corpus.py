"""
Rescan Corpus — actualizează metadata JSON pentru toate documentele existente.
Rulează după actualizarea semnalelor GROUND_TRUTH în manage_corpus.py.

Utilizare:
  python rescan_corpus.py
  python rescan_corpus.py --industry automotive
"""

import json
import argparse
from pathlib import Path

BASE = Path(__file__).parent

GROUND_TRUTH = {
    "C01": [
        "trei oferte", "minimum trei", "3 oferte", "minim trei",
        "doua oferte", "două oferte", "2 oferte",
    ],
    "C02": [
        "audit on-site", "auditul on-site", "on-site audit",
        "audit initial", "audit inițial", "mandatory on-site",
    ],
    "C03": [
        "semestriale", "semestrial", "semestrială", "semi-annual",
        "reevaluarii semestr", "reevaluarea furnizorilor activi se realizeaz",
    ],
    "C04": [
        "il-ins-03** - procedura", "il-ins-03 rev.2",
        "il-ins-03** rev.2", "rev.2",
    ],
    "C05": [
        "critici sunt supusi", "critici sunt supuși",
        "reevaluarii semestr", "furnizori critici",
    ],
}

INDUSTRIES = ["automotive", "iso", "aerospace", "medical"]
DOC_TYPES  = ["long", "medium", "short", "archive"]


def detect(text):
    found = []
    tl = text.lower()
    for cid, signals in GROUND_TRUTH.items():
        for sig in signals:
            if sig.lower() in tl:
                found.append(cid)
                break
    return found


def rescan(industry):
    updated = 0
    unchanged = 0
    missing = 0

    for dtype in DOC_TYPES:
        doc_dir  = BASE / industry / dtype
        meta_dir = BASE / "metadata" / industry
        if not doc_dir.exists():
            continue
        for txt in sorted(doc_dir.glob("*.txt")):
            meta_path = meta_dir / f"{txt.stem}.json"
            if not meta_path.exists():
                print(f"  WARN: lipseste metadata pentru {txt.name}")
                missing += 1
                continue
            text = txt.read_text(encoding="utf-8")
            found = detect(text)
            meta  = json.loads(meta_path.read_text())
            old   = meta.get("detected_contradictions", [])
            meta["detected_contradictions"] = found
            meta["word_count"] = len(text.split())
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            if sorted(found) != sorted(old):
                print(f"  UPDATED {txt.name}: {old} -> {found}")
                updated += 1
            else:
                unchanged += 1

    return updated, unchanged, missing


def main():
    parser = argparse.ArgumentParser(description="Re-scanează corpus si actualizeaza metadata")
    parser.add_argument("--industry", choices=INDUSTRIES, help="Doar o industrie")
    args = parser.parse_args()

    industries = [args.industry] if args.industry else INDUSTRIES
    total_updated = 0

    print("Re-scanare corpus...")
    for industry in industries:
        ind_path = BASE / industry
        if not ind_path.exists():
            continue
        print(f"\n{industry.upper()}:")
        upd, unch, miss = rescan(industry)
        print(f"  Actualizate: {upd} | Neschimbate: {unch} | Fara metadata: {miss}")
        total_updated += upd

    print(f"\nTotal actualizate: {total_updated}")
    print("\nRuleaza acum:")
    print("  python manage_corpus.py validate --industry automotive")


if __name__ == "__main__":
    main()
