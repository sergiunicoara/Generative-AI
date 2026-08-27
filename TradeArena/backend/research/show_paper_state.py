"""Afiseaza starea curenta a celor 5 conturi paper trading."""
import sqlite3
from pathlib import Path
import json

DB   = Path(__file__).resolve().parents[2] / "backend" / "environment" / "paper_trading.db"
OPTS = Path(__file__).resolve().parents[2] / "backend" / "environment" / "paper_options.json"
MG   = Path(__file__).resolve().parents[2] / "backend" / "environment" / "paper_mg.json"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

print("=" * 60)
print("  PAPER TRADER — stare curenta")
print("=" * 60)

print("\nCONTURI:")
for r in conn.execute("SELECT trader_id, cash, initial_balance FROM accounts ORDER BY trader_id"):
    pnl = r["cash"] - r["initial_balance"]
    pct = pnl / r["initial_balance"] * 100
    print(f"  {r['trader_id']:<15}  cash=${r['cash']:>8,.2f}  P&L ${pnl:>+7,.2f} ({pct:>+.1f}%)")

print("\nHOLDINGS (pozitii deschise):")
rows = list(conn.execute("SELECT trader_id, ticker, quantity, avg_cost FROM holdings"))
if rows:
    for r in rows:
        print(f"  {r['trader_id']:<15}  {r['ticker']}  {r['quantity']:.4f}sh @ ${r['avg_cost']:.2f}")
else:
    print("  (nicio pozitie activa)")

print("\nTRANZACTII (toate):")
trades = list(conn.execute("SELECT trader_id, ticker, quantity, price, ts FROM trades ORDER BY id"))
if trades:
    for r in trades:
        side = "BUY " if r["quantity"] > 0 else "SELL"
        print(f"  {r['trader_id']:<15}  {side} {r['ticker']}  "
              f"{abs(r['quantity']):.4f}sh @ ${r['price']:.2f}  {r['ts'][:16]}")
else:
    print("  (nicio tranzactie inca)")

opts = json.loads(OPTS.read_text()) if OPTS.exists() else {}
if any(opts.values()):
    print("\nOPTIUNI DESCHISE (S3/S4):")
    for tid, positions in opts.items():
        for ticker, o in positions.items():
            print(f"  {tid:<15}  {ticker} K=${o['strike']}  exp={o['expiry_date']}  "
                  f"qty={o['qty']:.4f}  entry=${o['entry_px']:.2f}")

mg = json.loads(MG.read_text()) if MG.exists() else {}
if any(mg.values()):
    print("\nPOZITII MG INTRADAY (S5 — exit 22:55):")
    for tid, positions in mg.items():
        for ticker, m in positions.items():
            print(f"  {tid:<15}  {ticker}  {m['qty']:.4f}sh @ ${m['entry_price']:.2f}"
                  f"  gap {m['gap_pct']:+.1f}%  data={m['entry_date']}")

conn.close()

