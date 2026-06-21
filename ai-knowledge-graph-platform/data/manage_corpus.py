"""
Corpus Manager — Demo Documents ISO/IATF
Salvează, validează și pregătește documentele pentru ingestie.

Structură:
  data/
    automotive/long|medium|short|archive/
    iso/long|medium|short|archive/
    aerospace/long|medium|short|archive/
    medical/long|medium|short|archive/
    sample_docs/     <- documente reale/referință (flat)
    eval_golden/     <- queries + răspunsuri așteptate pentru evaluare
    metadata/automotive|iso|aerospace|medical/

Limba se stochează ca metadata pe document (language: ro|en|mixed)
nu ca director separat.

Utilizare:
  python manage_corpus.py save --file out.txt --name PQ-07-rev3 --industry automotive
  python manage_corpus.py save --file out.txt --name PQ-07-rev3 --industry automotive --lang en
  python manage_corpus.py list
  python manage_corpus.py list --industry automotive
  python manage_corpus.py list --industry automotive --lang ro
  python manage_corpus.py validate --industry automotive
  python manage_corpus.py report
  python manage_corpus.py golden --add
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent

INDUSTRIES = ["automotive", "iso", "aerospace", "medical"]
DOC_TYPES  = ["long", "medium", "short", "archive"]
LANGUAGES  = ["ro", "en", "mixed"]

WORD_TARGETS = {
    "long":    (4000, 99999),
    "medium":  (1500, 5000),
    "short":   (200,  900),
    "archive": (200,  99999),
}

NAMING_MAP = {
    "PQ-":    "long",   "CSR-":  "long",   "MC-":   "long",
    "PFMEA":  "long",   "PPAP":  "long",   "SPEC-": "long",
    "IL-":    "medium", "PC-":   "medium", "PN-":   "medium",
    "PAI-":   "medium", "RFA-":  "medium", "PSC-":  "medium",
    "PMEN-":  "medium", "PCAL-": "medium", "PTRAS-":"medium",
    "PTRN-":  "medium", "PRIZ-": "medium",
    "FP-":    "short",  "REG-":  "short",
}

GROUND_TRUTH = {
    "C01": {
        "title": "Număr oferte",
        "risk": "Major",
        "doc_a": "PQ-07 rev.3 §4.2",
        "doc_b": "IL-PROC-12 rev.1 §3.1",
        "signals": ["trei oferte", "minimum trei", "3 oferte", "minim trei", "doua oferte", "două oferte", "2 oferte"],
        "desc": "PQ-07 impune 3 oferte, IL-PROC-12 permite 2 oferte sub 2000 EUR",
    },
    "C02": {
        "title": "Audit inițial lipsă",
        "risk": "Major",
        "doc_a": "CSR-CLIENT-2023 §2.4",
        "doc_b": "PQ-07 rev.3 §7.3",
        "signals": ["audit on-site", "auditul on-site", "on-site audit", "audit initial", "audit inițial", "mandatory on-site"],
        "desc": "CSR impune audit on-site, PQ-07 nu îl menționează",
    },
    "C03": {
        "title": "Frecvență reevaluare",
        "risk": "Minor",
        "doc_a": "PQ-07 rev.3 §8.1",
        "doc_b": "MC-01-S8 §8.4.1",
        "signals": ["semestriale", "semestrial", "semestrială", "semi-annual", "reevaluarii semestr", "reevaluarea furnizorilor activi se realizeaz"],
        "desc": "PQ-07 spune anual, MC-01 spune semestrial",
    },
    "C04": {
        "title": "Referință versiune depășită",
        "risk": "Major",
        "doc_a": "PC-COMP-07 rev.3 §2.1",
        "doc_b": "IL-INS-03 rev.4 (curentă)",
        "signals": ["il-ins-03** - procedura", "il-ins-03 rev.2", "il-ins-03** rev.2", "rev.2"],
        "desc": "PC-COMP-07 referențiază IL-INS-03 rev.2, curentă e rev.4",
    },
    "C05": {
        "title": "Termen furnizori critici",
        "risk": "Major",
        "doc_a": "RFA-REG-01 rev.5 §4.1",
        "doc_b": "CSR-CLIENT-2023 §3.7",
        "signals": ["critici sunt supusi", "critici sunt supuși", "reevaluarii semestr", "furnizori critici", "critical suppliers"],
        "desc": "RFA spune anual, CSR impune semestrial pentru furnizori critici",
    },
}


def get_dirs(industry: str) -> dict:
    ind = BASE / industry
    dirs = {dt: ind / dt for dt in DOC_TYPES}
    dirs["metadata"] = BASE / "metadata" / industry
    return dirs


def detect_type(name: str) -> str:
    n = name.upper()
    for prefix, dt in NAMING_MAP.items():
        if n.startswith(prefix.upper()):
            return dt
    return "medium"


def detect_language(text: str) -> str:
    ro_markers = ["și", "în", "că", "este", "sunt", "procedură", "furnizor", "calitate"]
    en_markers = ["the", "and", "shall", "procedure", "supplier", "quality", "requirement"]
    text_lower = text.lower()
    ro_score = sum(1 for w in ro_markers if f" {w} " in text_lower)
    en_score = sum(1 for w in en_markers if f" {w} " in text_lower)
    if ro_score > en_score * 1.5:
        return "ro"
    elif en_score > ro_score * 1.5:
        return "en"
    return "mixed"


def count_words(text: str) -> int:
    return len(text.split())


def detect_contradictions(text: str) -> list:
    found = []
    tl = text.lower()
    for cid, cd in GROUND_TRUTH.items():
        for sig in cd["signals"]:
            if sig.lower() in tl and cid not in found:
                found.append(cid)
                break
    return found


def extract_metadata(text: str, name: str, industry: str, doc_type: str, lang_override: str = None) -> dict:
    lang = lang_override or detect_language(text)
    meta = {
        "doc_id":   name,
        "industry": industry,
        "doc_type": doc_type,
        "language": lang,
        "saved_at": datetime.now().isoformat(),
        "word_count":  count_words(text),
        "char_count":  len(text),
        "detected_contradictions": detect_contradictions(text),
    }
    block = re.search(r"METADATA[:\s]*(.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    if block:
        for line in block.group(1).strip().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                key = k.strip().lower().replace(" ", "_")
                if key not in meta:
                    meta[key] = v.strip()
    return meta


def cmd_save(args):
    industry = args.industry
    name     = args.name.replace(" ", "-")
    doc_type = args.type or detect_type(name)
    dirs     = get_dirs(industry)

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        print("Paste conținutul (termini cu o linie care conține doar END):")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        text = "\n".join(lines)

    wc = count_words(text)
    min_w, max_w = WORD_TARGETS.get(doc_type, (0, 99999))
    lang = args.lang or detect_language(text)

    print(f"\nDocument : {name}")
    print(f"Industrie: {industry}")
    print(f"Tip      : {doc_type}")
    print(f"Limbă    : {lang}")
    print(f"Cuvinte  : {wc} (target {min_w}–{max_w})")

    if wc < min_w:
        print(f"AVERTISMENT: prea scurt ({wc} < {min_w})")
    elif wc > max_w:
        print(f"AVERTISMENT: prea lung ({wc} > {max_w})")
    else:
        print("Dimensiune: OK")

    out_path = dirs[doc_type] / f"{name}.txt"
    out_path.write_text(text, encoding="utf-8")

    meta = extract_metadata(text, name, industry, doc_type, args.lang)
    meta_path = dirs["metadata"] / f"{name}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if meta["detected_contradictions"]:
        print(f"Contradicții detectate: {meta['detected_contradictions']}")
    else:
        print("Contradicții detectate: niciuna — verifică manual")

    print(f"Salvat: {out_path}")


def cmd_list(args):
    industries = [args.industry] if args.industry else INDUSTRIES
    lang_filter = args.lang
    grand_docs = 0
    grand_words = 0

    print("\n" + "=" * 68)
    print("CORPUS DOCUMENTE DEMO")
    print("=" * 68)

    for industry in industries:
        dirs = get_dirs(industry)
        has_content = any(list(dirs[dt].glob("*.txt")) for dt in DOC_TYPES)
        if not has_content:
            print(f"\n{industry.upper()}: (gol)")
            continue

        print(f"\n{'─'*68}")
        print(f"  {industry.upper()}")
        print(f"{'─'*68}")

        ind_docs = 0
        ind_words = 0

        for doc_type in DOC_TYPES:
            files = sorted(dirs[doc_type].glob("*.txt"))
            if not files:
                continue

            filtered = []
            for f in files:
                if lang_filter:
                    mp = dirs["metadata"] / f"{f.stem}.json"
                    if mp.exists():
                        m = json.loads(mp.read_text())
                        if m.get("language") != lang_filter:
                            continue
                filtered.append(f)

            if not filtered:
                continue

            type_words = 0
            print(f"\n  {doc_type} ({len(filtered)} doc):")
            for f in filtered:
                text = f.read_text(encoding="utf-8")
                wc   = count_words(text)
                type_words += wc
                mp   = dirs["metadata"] / f"{f.stem}.json"
                ctag = ""
                lang = "?"
                if mp.exists():
                    m = json.loads(mp.read_text())
                    ctag = (" [" + ",".join(m.get("detected_contradictions", [])) + "]"
                            if m.get("detected_contradictions") else "")
                    lang = m.get("language", "?")
                print(f"    {f.stem:<30} {wc:>6} cuv  [{lang}]{ctag}")
            print(f"    {'subtotal':<30} {type_words:>6} cuv")
            ind_docs  += len(filtered)
            ind_words += type_words

        print(f"\n  TOTAL {industry.upper()}: {ind_docs} doc, {ind_words:,} cuv")
        grand_docs  += ind_docs
        grand_words += ind_words

    # sample_docs
    sample_files = list((BASE / "sample_docs").glob("*.txt"))
    if sample_files:
        print(f"\n{'─'*68}")
        print(f"  SAMPLE_DOCS ({len(sample_files)} fișiere referință)")
        print(f"{'─'*68}")
        for f in sorted(sample_files):
            wc = count_words(f.read_text(encoding="utf-8", errors="ignore"))
            print(f"    {f.name:<36} {wc:>6} cuv")

    print("\n" + "=" * 68)
    print(f"GRAND TOTAL: {grand_docs} documente, {grand_words:,} cuvinte")
    if lang_filter:
        print(f"(filtrat pe limba: {lang_filter})")
    print("=" * 68)


def cmd_validate(args):
    industries = [args.industry] if args.industry else INDUSTRIES
    print("\nVALIDARE CORPUS")
    print("=" * 68)
    all_ok = True

    for industry in industries:
        dirs = get_dirs(industry)
        coverage = {cid: [] for cid in GROUND_TRUTH}
        errors   = []
        warnings = []

        for doc_type in ["long", "medium", "short"]:
            for f in dirs[doc_type].glob("*.txt"):
                mp = dirs["metadata"] / f"{f.stem}.json"
                if not mp.exists():
                    warnings.append(f"{f.name}: lipsește metadata")
                    continue
                m  = json.loads(mp.read_text())
                wc = m.get("word_count", 0)
                mn, mx = WORD_TARGETS[doc_type]
                if wc < mn:
                    errors.append(f"{f.name}: prea scurt ({wc} cuv, min {mn})")
                for cid in m.get("detected_contradictions", []):
                    if cid in coverage:
                        entry = f"{f.stem} [{m.get('language','?')}]"
                        coverage[cid].append(entry)

        print(f"\n{industry.upper()} — acoperire contradicții:")
        for cid, docs in coverage.items():
            status = "OK  " if docs else "LIPSĂ"
            if not docs:
                all_ok = False
            ct   = GROUND_TRUTH[cid]
            line = f"  {cid} [{ct['risk']:5}] {ct['title']:<28} {status}"
            if docs:
                line += " → " + ", ".join(docs)
            print(line)

        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  WARN:  {w}")

    print("\n" + "─" * 68)
    print("STATUS: CORPUS VALID — gata pentru ingestie" if all_ok
          else "STATUS: CORPUS INCOMPLET — rezolvă lipsurile înainte de ingestie")


def cmd_report(args):
    report = {"generated_at": datetime.now().isoformat(), "industries": {}}

    for industry in INDUSTRIES:
        dirs  = get_dirs(industry)
        docs  = []
        cov   = {cid: [] for cid in GROUND_TRUTH}
        words = 0
        langs = {"ro": 0, "en": 0, "mixed": 0}

        for dt in DOC_TYPES:
            for f in dirs[dt].glob("*.txt"):
                mp = dirs["metadata"] / f"{f.stem}.json"
                if mp.exists():
                    m = json.loads(mp.read_text())
                    docs.append(m)
                    words += m.get("word_count", 0)
                    lang   = m.get("language", "?")
                    if lang in langs:
                        langs[lang] += 1
                    for cid in m.get("detected_contradictions", []):
                        if cid in cov:
                            cov[cid].append(f.stem)

        report["industries"][industry] = {
            "total_documents": len(docs),
            "total_words": words,
            "language_breakdown": langs,
            "contradiction_coverage": cov,
            "ground_truth": GROUND_TRUTH,
            "documents": docs,
        }

    out = BASE / "metadata" / "corpus_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Raport salvat: {out}")
    for ind, d in report["industries"].items():
        if d["total_documents"] > 0:
            covered = sum(1 for v in d["contradiction_coverage"].values() if v)
            lb = d["language_breakdown"]
            print(f"  {ind}: {d['total_documents']} doc, {d['total_words']:,} cuv, "
                  f"contradicții {covered}/5, ro={lb['ro']} en={lb['en']} mixed={lb['mixed']}")


def cmd_golden(args):
    golden_dir = BASE / "eval_golden"
    out_file   = golden_dir / "queries.json"

    existing = []
    if out_file.exists():
        existing = json.loads(out_file.read_text())

    if args.add:
        print("Adaugă query golden (Enter gol pentru a termina câmpul):")
        entry = {
            "id": f"Q{len(existing)+1:03d}",
            "industry": input("Industrie (automotive/iso/aerospace/medical): ").strip(),
            "language": input("Limbă (ro/en/mixed): ").strip(),
            "query": input("Query: ").strip(),
            "expected_contradictions": input("Contradicții așteptate (ex: C01,C04): ").strip().split(","),
            "expected_answer_contains": input("Cuvinte cheie în răspuns așteptat: ").strip().split(","),
            "added_at": datetime.now().isoformat(),
        }
        existing.append(entry)
        out_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Salvat query {entry['id']} în {out_file}")
    else:
        if existing:
            print(f"\nGolden queries ({len(existing)}):")
            for q in existing:
                print(f"  {q['id']} [{q.get('language','?')}] {q['query'][:60]}")
        else:
            print("eval_golden/queries.json este gol — folosește --add pentru a adăuga queries.")


def main():
    p = argparse.ArgumentParser(description="Corpus Manager — ISO/IATF Demo Documents")
    sub = p.add_subparsers(dest="cmd")

    ps = sub.add_parser("save", help="Salvează document generat de Haiku")
    ps.add_argument("--file",     help="Fișier text input")
    ps.add_argument("--name",     required=True, help="Numele documentului (ex: PQ-07-rev3)")
    ps.add_argument("--industry", required=True, choices=INDUSTRIES)
    ps.add_argument("--type",     choices=DOC_TYPES, help="Autodetectat dacă lipsește")
    ps.add_argument("--lang",     choices=LANGUAGES, help="ro|en|mixed (autodetectat dacă lipsește)")

    pl = sub.add_parser("list", help="Listează documentele")
    pl.add_argument("--industry", choices=INDUSTRIES)
    pl.add_argument("--lang",     choices=LANGUAGES, help="Filtrează pe limbă")

    pv = sub.add_parser("validate", help="Validează corpus")
    pv.add_argument("--industry", choices=INDUSTRIES)

    sub.add_parser("report", help="Generează raport JSON complet")

    pg = sub.add_parser("golden", help="Gestionează eval_golden queries")
    pg.add_argument("--add", action="store_true", help="Adaugă un query nou")

    args = p.parse_args()
    {
        "save":     cmd_save,
        "list":     cmd_list,
        "validate": cmd_validate,
        "report":   cmd_report,
        "golden":   cmd_golden,
    }.get(args.cmd, lambda _: p.print_help())(args)


if __name__ == "__main__":
    main()
