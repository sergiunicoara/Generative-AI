"""
Document Generator - ISO/IATF Demo Corpus
Genereaza documente fictive cu LLM si le salveaza automat.

Utilizare:
  python generate_doc.py --industry automotive --doctype csr --lang mixed --name CSR-CLIENT-2023 --contradictions C01 C02 C05 --model gpt-4o-mini
  python generate_doc.py --industry automotive --doctype proc --lang ro --name PQ-07-rev3 --contradictions C01 C03
  python generate_doc.py --list-doctypes
  python generate_doc.py --industry automotive --doctype csr --lang mixed --name TEST --contradictions C01 --dry-run
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("Instaleaza: pip install anthropic")
    sys.exit(1)

BASE = Path(__file__).parent

CFGS = {
    "automotive": {"co": "PlastiAuto SRL",  "cl": "AutoCorp GmbH",  "std": "IATF 16949"},
    "iso":        {"co": "MechProd SRL",    "cl": "EuroClient AG",   "std": "ISO 9001:2015"},
    "aerospace":  {"co": "AeroComp SRL",    "cl": "AirSystems GmbH", "std": "EN 9100"},
    "medical":    {"co": "MedDev SRL",      "cl": "MedSupply GmbH",  "std": "ISO 13485"},
}

DOC_META = {
    "fi":   {"defaultName": "FP-INJ-01",       "dtype": "short",  "len": "350-500",   "min_words": 300},
    "il":   {"defaultName": "IL-PROC-12-rev1", "dtype": "medium", "len": "2000-3000", "min_words": 1500},
    "proc": {"defaultName": "PQ-07-rev3",      "dtype": "long",   "len": "6000-8000", "min_words": 4000},
    "csr":  {"defaultName": "CSR-CLIENT-2023", "dtype": "long",   "len": "8000-12000","min_words": 5000},
    "mc":   {"defaultName": "MC-01-S8-rev6",   "dtype": "long",   "len": "6000-9000", "min_words": 4000},
}

LENGTH_PROFILES = {
    "csr":  {"target_words": 12000, "min_words": 10000},
    "proc": {"target_words": 8000,  "min_words": 7000},
    "mc":   {"target_words": 7000,  "min_words": 6000},
    "il":   {"target_words": 3000,  "min_words": 2500},
    "fi":   {"target_words": 600,   "min_words": 400},
}

LONG_APPENDICES = """Appendix A – Example Forms
Appendix B – KPI Scorecards
Appendix C – Audit Checklists
Appendix D – Escalation Workflow"""

DOC_TITLES = {
    "fi":   {"ro": "FISA DE PROCES",               "en": "PROCESS SHEET"},
    "il":   {"ro": "INSTRUCTIUNE DE LUCRU",         "en": "WORK INSTRUCTION"},
    "proc": {"ro": "PROCEDURA DE SISTEM",           "en": "SYSTEM PROCEDURE"},
    "csr":  {"ro": "CERINTE SPECIFICE CLIENT",      "en": "CUSTOMER SPECIFIC REQUIREMENTS"},
    "mc":   {"ro": "MANUAL CALITATE SECTIUNEA 8",   "en": "QUALITY MANUAL SECTION 8"},
}

DOC_STRUCTS = {
    "fi": """1. HEADER: Cod, Titlu, Revizie, Data, Responsabil
2. SCOP SI DOMENIU (2-3 propozitii)
3. REFERINTE DOCUMENTE (4-5 documente cu cod si revizie)
4. DESCRIERE PROCES (6-8 pasi numerotati)
5. PARAMETRI DE CONTROL (tabel: parametru / target / limite / frecventa)
6. INREGISTRARI GENERATE
7. ISTORIC REVIZII (ultimele 3 revizii)""",

    "il": """1. HEADER COMPLET cu date aprobare si distributie
2. SCOP
3. DOMENIU DE APLICARE
4. DEFINITII SI ABREVIERI (8-10 termeni)
5. REFERINTE (6-7 documente cu cod si revizie)
6. RESPONSABILITATI (tabel pe roluri)
7. PROCEDURA DETALIATA:
   7.1 Initierea comenzii
   7.2 Selectarea furnizorului
   7.3 Emiterea comenzii
   7.4 Receptia si verificarea
   7.5 Gestionarea neconformitatilor
8. INREGISTRARI
9. ISTORIC REVIZII""",

    "proc": """1. HEADER: cod, titlu, rev.3, aprobare Director Calitate si Director General
2. SCOP SI OBIECTIVE MASURABILE
3. DOMENIU DE APLICARE
4. DEFINITII SI ABREVIERI (15-20 termeni)
5. REFERINTE NORMATIVE SI INTERNE (10-15 documente)
6. RESPONSABILITATI - matrice RACI
7. CALIFICAREA FURNIZORILOR NOI (7.1-7.4 subsectiuni detaliate)
8. REEVALUAREA FURNIZORILOR ACTIVI (8.1-8.3 subsectiuni)
9. CLASIFICAREA FURNIZORILOR
10. GESTIONAREA PERFORMANTEI SLABE
11. INDICATORI KPI cu valori target numerice
12. INREGISTRARI SI TRASABILITATE
13. ISTORIC REVIZII""",

    "csr": """1. COVER PAGE: versiune, data intrare in vigoare, lista distributie
2. INTRODUCERE SI SCOP
3. CERINTE GENERALE
4. CERINTE CALIFICARE FURNIZORI (4.1-4.3)
5. CERINTE CALITATE (SPC, MSA, FAI)
6. LIVRARE SI LOGISTICA
7. MANAGEMENTUL SCHIMBARILOR
8. MONITORIZAREA PERFORMANTEI FURNIZORILOR (8.1-8.3 cu KPI numerici)
9. MANAGEMENTUL NECONFORMITATILOR
10. CERINTE DOCUMENTARE
11. DREPTURI DE AUDIT
12. SANCTIUNI SI CONSECINTE
13. ISTORIC REVIZII""",

    "mc": """8.1 Planificarea si controlul operational
8.2 Cerinte pentru produse si servicii (8.2.1-8.2.4)
8.3 Proiectare si dezvoltare
8.4 Controlul furnizorilor externi (8.4.1-8.4.3)
8.5 Productie si furnizare de servicii (8.5.1-8.5.6)
8.6 Eliberarea produselor
8.7 Controlul iesirilor neconforme""",
}

LANG_NOTES = {
    "ro":    "Limba: EXCLUSIV romana tehnica. Nu folosi cuvinte in engleza in afara termenilor standard ISO/IATF (SPC, MSA, FAI, PPAP, FMEA).",
    "en":    "Language: EXCLUSIVELY English technical language throughout.",
    "mixed": "Limba: mixta realist - titlurile sectiunilor in engleza, corpul textului in romana tehnica. Ca un document real trimis de un client german unui furnizor roman.",
}

CT_DATA = {
    "C01": {
        "sections": {"fi": "sectiunea 3 (REFERINTE DOCUMENTE)", "il": "sectiunea 7.2 (Selectarea furnizorului)", "proc": "sectiunea 7.1 (Calificarea furnizorilor noi)", "csr": "sectiunea 4.1 (CERINTE CALIFICARE FURNIZORI)", "mc": "sectiunea 8.4.1 (Controlul furnizorilor externi)"},
        "ro": 'Specifica explicit: "Pentru aprobarea oricarui furnizor nou, sunt obligatorii MINIMUM TREI (3) oferte competitive, indiferent de valoarea comenzii." Scrie aceasta ca politica oficiala, FARA nicio nota sau explicatie.',
        "en": 'State explicitly: "A minimum of THREE (3) competitive quotes is mandatory for any new supplier approval, regardless of order value." Write as official policy only, no explanatory notes.',
    },
    "C02": {
        "sections": {"fi": "sectiunea 3 (REFERINTE DOCUMENTE)", "il": "sectiunea 7.3 (Emiterea comenzii)", "proc": "sectiunea 7.3 (Aprobarea furnizorului)", "csr": "sectiunea 4.1 (CERINTE CALIFICARE FURNIZORI)", "mc": "sectiunea 8.4.2 (Tipul si amploarea controlului)"},
        "ro": 'Specifica explicit: "Auditul on-site la sediul furnizorului este OBLIGATORIU si trebuie finalizat inainte de orice aprobare a unui furnizor nou." Scrie aceasta ca politica oficiala, FARA nicio nota sau explicatie.',
        "en": 'State explicitly: "An on-site audit at the supplier premises is MANDATORY and must be completed prior to any new supplier approval." Write as official policy only, no explanatory notes.',
    },
    "C03": {
        "sections": {"fi": "sectiunea 5 (PARAMETRI DE CONTROL)", "il": "sectiunea 7.2 (Selectarea furnizorului)", "proc": "sectiunea 8.1 (Frecventa reevaluarii)", "csr": "sectiunea 8.1 (MONITORIZAREA PERFORMANTEI)", "mc": "sectiunea 8.4.1 (Controlul furnizorilor externi)"},
        "ro": 'Specifica explicit: "Reevaluarea furnizorilor activi se realizeaza SEMESTRIAL." Scrie aceasta ca politica oficiala, FARA nicio nota sau explicatie.',
        "en": 'State explicitly: "Active supplier re-evaluation shall be conducted SEMI-ANNUALLY." Write as official policy only, no explanatory notes.',
    },
    "C04": {
        "sections": {"fi": "sectiunea 3 (REFERINTE DOCUMENTE)", "il": "sectiunea 7.4 (Receptia si verificarea)", "proc": "sectiunea 5 (REFERINTE NORMATIVE)", "csr": "sectiunea 10 (CERINTE DOCUMENTARE)", "mc": "sectiunea 8.5.2 (Identificare si trasabilitate)"},
        "ro": 'Referentiaza explicit "IL-INS-03 rev.2" ca document de referinta pentru inspectia la receptie. Nu adauga nicio nota ca aceasta versiune ar fi depasita.',
        "en": 'Reference explicitly "IL-INS-03 rev.2" as the reference document for incoming inspection. Do not add any note that this revision is obsolete.',
    },
    "C05": {
        "sections": {"fi": "sectiunea 5 (PARAMETRI DE CONTROL)", "il": "sectiunea 7.2 (Selectarea furnizorului)", "proc": "sectiunea 8.1 (Frecventa reevaluarii)", "csr": "sectiunea 8.2 (MONITORIZAREA PERFORMANTEI - furnizori critici)", "mc": "sectiunea 8.4.1 (Controlul furnizorilor externi)"},
        "ro": 'Specifica explicit: "Furnizorii clasificati ca CRITICI sunt supusi reevaluarii SEMESTRIALE." Scrie aceasta ca politica oficiala, FARA nicio nota sau explicatie.',
        "en": 'State explicitly: "Suppliers classified as CRITICAL are subject to SEMI-ANNUAL re-evaluation." Write as official policy only, no explanatory notes.',
    },
}


def build_metadata_block(industry, doctype, lang, name, contradictions):
    cfg = CFGS[industry]
    dm = DOC_META[doctype]
    return (
        f"=== METADATA ===\n"
        f"doc_id: {name}\n"
        f"industry: {industry}\n"
        f"doc_type: {dm['dtype']}\n"
        f"language: {lang}\n"
        f"standard: {cfg['std']}\n"
        f"contradiction: {','.join(contradictions)}\n"
        f"save_path: data/{industry}/{dm['dtype']}/{name}.txt\n"
        f"save_command: python manage_corpus.py save --file {name}.txt "
        f"--name {name} --industry {industry} --lang {lang}\n"
        f"================"
    )


def build_prompt(industry, doctype, lang, name, contradictions):
    cfg = CFGS[industry]
    profile = LENGTH_PROFILES[doctype]
    target_words = profile["target_words"]
    min_words = profile["min_words"]
    title_key = "en" if lang == "en" else "ro"
    doc_title = DOC_TITLES[doctype][title_key]
    required_structure = DOC_STRUCTS[doctype]
    if doctype in ("csr", "proc", "mc"):
        required_structure = f"{required_structure}\n{LONG_APPENDICES}"

    ct_blocks = ""
    if contradictions:
        blocks = []
        for cid in contradictions:
            ct = CT_DATA[cid]
            sec = ct["sections"][doctype]
            txt = ct["en"] if lang == "en" else ct["ro"]
            blocks.append(f"{cid} - introdu la {sec}:\n{txt}")
        ct_blocks = (
            "CONTRADICTII DELIBERATE - scrie-le EXCLUSIV ca politica oficiala, "
            "FARA nicio nota, explicatie sau mentiune ca ar fi o contradictie:\n\n"
            + "\n\n".join(blocks) + "\n\n"
        )

    return (
        f"You are an expert in {cfg['std']} quality management systems.\n"
        f"Generate a complete and realistic {doc_title} for the fictitious company "
        f"\"{cfg['co']}\", main client \"{cfg['cl']}\".\n\n"
        f"DOCUMENT NAME: {name}\n"
        f"INDUSTRY: {industry}\n"
        f"STANDARD: {cfg['std']}\n\n"
        f"{LANG_NOTES[lang]}\n\n"
        f"Target length: approximately {target_words} words.\n"
        f"The document is incomplete if shorter than {min_words} words.\n"
        f"Nu prescurta, nu rezuma, nu sari nicio sectiune.\n\n"
        f"REQUIRED STRUCTURE:\n{required_structure}\n\n"
        f"{ct_blocks}"
        f"IMPORTANT: Scrie TOATE sectiunile complet si detaliat. "
        f"Continua fara intrerupere pana la finalul documentului.\n\n"
        f"Do not generate the metadata block. It will be appended programmatically.\n\n"
        f"START GENERATING NOW:"
    )


def detect_provider(model):
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    elif model.startswith("gemini"):
        return "gemini"
    elif model.startswith("llama") or model.startswith("mixtral") or model.startswith("groq"):
        return "groq"
    else:
        return "claude"


def call_once(provider, model, messages):
    if provider == "openai":
        from openai import OpenAI
        oa = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        r = oa.chat.completions.create(
            model=model, max_tokens=4096, messages=messages
        )
        return r.choices[0].message.content, r.choices[0].finish_reason

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gm = genai.GenerativeModel(model)
        combined = "\n".join(m["content"] for m in messages)
        r = gm.generate_content(
            combined,
            generation_config={"max_output_tokens": 4096, "temperature": 0.7}
        )
        return r.text, "stop"

    elif provider == "groq":
        try:
            from groq import Groq
        except ImportError:
            print("Instaleaza: pip install groq")
            import sys; sys.exit(1)
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        r = client.chat.completions.create(
            model=model, max_tokens=4096, messages=messages
        )
        return r.choices[0].message.content, r.choices[0].finish_reason

    else:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        r = client.messages.create(
            model=model, max_tokens=4096, messages=messages
        )
        return r.content[0].text, r.stop_reason


def generate_with_continuation(prompt, provider, model, target_words, min_words, metadata):
    full_text = ""
    messages = [{"role": "user", "content": prompt}]
    current_word_count = 0
    rnd = 0

    while current_word_count < min_words and rnd < 20:
        print(f"  Round {rnd + 1}...", end=" ", flush=True)
        chunk, finish = call_once(provider, model, messages)
        full_text += chunk
        current_word_count = len(full_text.split())
        print(f"{len(chunk.split())} cuv (total: {current_word_count})")
        print(f"  Target words: {target_words}")
        print(f"  Current words: {current_word_count}")
        print(f"  Remaining words: {max(target_words - current_word_count, 0)}")

        has_metadata = "=== METADATA ===" in full_text

        if has_metadata and current_word_count >= min_words:
            print("  Document complet.")
            break

        if has_metadata and current_word_count < min_words:
            print(f"  Prea scurt ({current_word_count}/{min_words} cuv) - sterg metadata si continua...")
            full_text = full_text[:full_text.index("=== METADATA ===")].rstrip()
            current_word_count = len(full_text.split())
            messages.append({"role": "assistant", "content": chunk})
            messages.append({"role": "user", "content": (
                "Continue the document from the exact point where it stopped.\n\n"
                f"Current length: {current_word_count}\n"
                f"Target length: {target_words}\n\n"
                "Do not summarize.\n"
                "Do not conclude.\n"
                "Do not repeat previous content.\n"
                "Continue with the next unfinished sections.\n"
                "Add substantial new content.\n"
                "Do not add metadata."
            )})
            rnd += 1
            continue

        messages.append({"role": "assistant", "content": chunk})
        messages.append({"role": "user", "content": (
            "Continue the document from the exact point where it stopped.\n\n"
            f"Current length: {current_word_count}\n"
            f"Target length: {target_words}\n\n"
            "Do not summarize.\n"
            "Do not conclude.\n"
            "Do not repeat previous content.\n"
            "Continue with the next unfinished sections.\n"
            "Add substantial new content."
        )})
        rnd += 1

    if "=== METADATA ===" in full_text:
        full_text = full_text[:full_text.index("=== METADATA ===")].rstrip()
    return f"{full_text.rstrip()}\n\n{metadata}\n"


def save_document(text, industry, doctype, name, lang, contradictions):
    dm = DOC_META[doctype]
    out_dir = BASE / industry / dm["dtype"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.txt"
    out_path.write_text(text, encoding="utf-8")

    meta_dir = BASE / "metadata" / industry
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{name}.json"
    meta = {
        "doc_id":    name,
        "industry":  industry,
        "doc_type":  dm["dtype"],
        "language":  lang,
        "saved_at":  datetime.now().isoformat(),
        "word_count": len(text.split()),
        "char_count": len(text),
        "detected_contradictions": contradictions,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generator documente demo ISO/IATF")
    parser.add_argument("--industry",       choices=list(CFGS.keys()),     required=False)
    parser.add_argument("--doctype",        choices=list(DOC_META.keys()),  required=False)
    parser.add_argument("--lang",           choices=["ro", "en", "mixed"],  default="ro")
    parser.add_argument("--name",           help="Numele documentului")
    parser.add_argument("--contradictions", nargs="*", choices=list(CT_DATA.keys()), default=[])
    parser.add_argument("--model",          default="claude-haiku-4-5-20251001",
                        help="Model: claude-haiku-4-5-20251001 | claude-sonnet-4-6 | gpt-4o-mini | gpt-4o | gemini-1.5-flash | gemini-1.5-pro")
    parser.add_argument("--list-doctypes",  action="store_true")
    parser.add_argument("--dry-run",        action="store_true")
    args = parser.parse_args()

    if args.list_doctypes:
        print("\nTipuri de documente:")
        for k, v in DOC_META.items():
            print(f"  {k:<6} {DOC_TITLES[k]['ro']:<35} ({v['len']} cuv) -> {v['dtype']}/")
        print("\nModele disponibile:")
        print("  claude-haiku-4-5-20251001  (Anthropic - ieftin)")
        print("  claude-sonnet-4-6          (Anthropic - calitate inalta)")
        print("  gpt-4o-mini                (OpenAI - ieftin, recomandat)")
        print("  gpt-4o                     (OpenAI - calitate inalta)")
        print("  gemini-1.5-flash           (Google - cel mai ieftin)")
        print("  gemini-1.5-pro             (Google - calitate inalta)")
        return

    if not args.industry or not args.doctype:
        parser.print_help()
        return

    name = args.name or DOC_META[args.doctype]["defaultName"]
    prompt = build_prompt(args.industry, args.doctype, args.lang, name, args.contradictions)

    if args.dry_run:
        print(prompt)
        return

    provider = detect_provider(args.model)
    key_map = {"openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY", "claude": "ANTHROPIC_API_KEY"}
    if not os.environ.get(key_map[provider]):
        print(f"ERROR: seteaza {key_map[provider]}")
        sys.exit(1)

    dm = DOC_META[args.doctype]
    print(f"\nGenerez: {name}")
    print(f"Tip:     {DOC_TITLES[args.doctype]['ro']} ({dm['len']} cuv)")
    print(f"Limba:   {args.lang}")
    print(f"Model:   {args.model} ({provider})")
    print(f"Contradictii: {args.contradictions or 'niciuna'}")
    print("Se apeleaza API-ul (cu continuare automata)...\n")

    profile = LENGTH_PROFILES[args.doctype]
    metadata = build_metadata_block(args.industry, args.doctype, args.lang, name, args.contradictions)
    text = generate_with_continuation(
        prompt,
        provider,
        args.model,
        profile["target_words"],
        profile["min_words"],
        metadata,
    )
    wc = len(text.split())
    out_path = save_document(text, args.industry, args.doctype, name, args.lang, args.contradictions)

    print(f"\nGata!")
    print(f"Cuvinte:  {wc} (target: {dm['len']})")
    print(f"Salvat:   {out_path}")
    print(f"\nVerifica cu:")
    print(f"  python manage_corpus.py list --industry {args.industry}")

    if wc < dm["min_words"] // 2:
        print(f"\nATENTION: document prea scurt ({wc} cuv). Incearca cu --model gpt-4o sau claude-sonnet-4-6")


if __name__ == "__main__":
    main()
