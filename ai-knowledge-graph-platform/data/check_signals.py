import re
from pathlib import Path

checks = [
    ('C01', 'automotive/long/CSR-CLIENT-2023.txt',    ['ofert', 'quotes', 'trei', 'three', 'minim']),
    ('C01', 'automotive/medium/IL-PROC-12-rev1.txt',  ['ofert', 'doua', 'two', 'minim']),
    ('C02', 'automotive/long/CSR-CLIENT-2023.txt',    ['on-site', 'audit', 'obligator', 'mandatory']),
    ('C03', 'automotive/long/PQ-07-rev3.txt',         ['semestri', 'anual', 'semi', 'reevaluar']),
    ('C03', 'automotive/long/MC-01-S8-rev6.txt',      ['semestri', 'anual', 'semi', 'reevaluar']),
    ('C04', 'automotive/medium/PC-COMP-07-rev3.txt',  ['rev.2', 'rev.4', 'revision', 'revizie']),
    ('C04', 'automotive/medium/IL-INS-03-rev2.txt',   ['rev.2', 'rev.4', 'revision', 'revizie']),
    ('C05', 'automotive/long/CSR-CLIENT-2023.txt',    ['critic', 'semestri', 'semi', 'anual']),
    ('C05', 'automotive/medium/RFA-REG-01-rev5.txt',  ['critic', 'semestri', 'anual', 'reevaluar']),
]

for cid, fname, keywords in checks:
    p = Path(fname)
    if not p.exists():
        print(f'LIPSA: {fname}')
        continue
    text = p.read_text(encoding='utf-8').lower()
    print(f'\n{cid} - {p.name}:')
    for kw in keywords:
        m = re.search(r'.{0,50}' + re.escape(kw) + r'.{0,50}', text)
        print(f'  [{kw}]: {m.group().strip()[:90] if m else "NEGASIT"}')
