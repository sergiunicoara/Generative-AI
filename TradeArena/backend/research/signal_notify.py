"""
Signal Notifier — run daily (e.g. 09:35 ET on weekdays).

Checks the pre-earnings D-20 window using the combined_strategy.md S2-Safer rules:
  - Universe: GOOGL, NVDA, AMZN, MSFT, META, AMD
  - Regime:   QQQ > 150dma
  - Score:    >= 1.05

Sends an email to NOTIFY_EMAIL when:
  - A new ticker enters the D-20 window with score >= 1.05
  - A held position reaches D-1 (exit signal)
  - Regime flips OFF (QQQ breaks below 150dma)

State is persisted to signal_notify_state.json so repeated runs
don't re-alert on the same signal.

Usage:
    uv run python -m backend.research.signal_notify

Setup (one time):
    Set NOTIFY_EMAIL and SMTP_* in your .env file, or it prints to console only.
    See SMTP_CONFIG section below.
"""

from __future__ import annotations

import json
import os
import smtplib
import sys
import time
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(override=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "mail4sergiu@gmail.com")
SMTP_HOST    = os.getenv("SMTP_HOST",    "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER",    "")
SMTP_PASS    = os.getenv("SMTP_PASS",    "")

WATCH_TICKERS    = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
SCORE_THRESHOLD  = 1.05
REGIME_MA        = 150
STATE_FILE       = Path(__file__).parent / "signal_notify_state.json"

BASE_QUALITY: dict[str, float] = {
    "GOOGL": 1.40, "NVDA": 1.50, "AMZN": 1.20,
    "MSFT":  1.10, "META": 1.10, "AMD":  1.00,
}

# ---------------------------------------------------------------------------
# Helpers (same logic as tools.py)
# ---------------------------------------------------------------------------

def _bounded(x: float) -> float:
    return max(-20.0, min(20.0, x)) / 100.0


def check_regime() -> dict:
    hist  = yf.Ticker("QQQ").history(period="1y", interval="1d")
    close = hist["Close"]
    ma    = float(close.rolling(REGIME_MA).mean().iloc[-1])
    cur   = float(close.iloc[-1])
    return {"ok": cur > ma, "qqq": round(cur, 2), "ma150": round(ma, 2)}


def compute_scores() -> dict[str, float]:
    raw = yf.download(WATCH_TICKERS + ["QQQ"], period="90d",
                      interval="1d", progress=False, auto_adjust=True)
    close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
    qqq_20d = (close["QQQ"].iloc[-1] / close["QQQ"].iloc[-21] - 1) * 100
    scores: dict[str, float] = {}
    for t in WATCH_TICKERS:
        if t not in close.columns:
            continue
        col  = close[t].dropna()
        if len(col) < 62:
            continue
        m20  = (col.iloc[-1] / col.iloc[-21] - 1) * 100
        m60  = (col.iloc[-1] / col.iloc[-62] - 1) * 100
        rs   = m20 - float(qqq_20d)
        base = BASE_QUALITY.get(t, 1.0)
        scores[t] = round(base + 1.20*_bounded(m20) + 0.80*_bounded(m60) + 1.50*_bounded(rs), 3)
    return scores


def get_upcoming_earnings(ticker: str) -> tuple[str, int] | None:
    today = date.today().isoformat()
    dates: list[str] = []
    try:
        obj = yf.Ticker(ticker)
        ed  = obj.earnings_dates
        if ed is not None and not ed.empty:
            for d in ed.index:
                ds = str(d.date()) if hasattr(d, "date") else str(d)[:10]
                if ds >= today:
                    dates.append(ds)
        try:
            cal = obj.calendar
            if cal:
                el = cal.get("Earnings Date", [])
                if hasattr(el, "tolist"): el = el.tolist()
                for dv in (el if isinstance(el, list) else [el]):
                    ds = str(pd.Timestamp(dv).date())
                    if ds >= today and ds not in dates:
                        dates.append(ds)
        except Exception:
            pass
    except Exception:
        pass
    if not dates:
        return None
    ann    = sorted(set(dates))[0]
    target = date.fromisoformat(ann)
    cur    = date.today()
    td     = sum(1 for i in range(1, (target - cur).days + 2)
                 if (cur + pd.Timedelta(days=i)).weekday() < 5
                 and (cur + pd.Timedelta(days=i)).date() <= target)
    return (ann, td)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "last_run":           None,
        "active_signals":     {},   # ticker → {"ann": date, "score": float, "alerted": date}
        "last_regime_ok":     None,
        "regime_alert_sent":  None,
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def send_email(subject: str, body_text: str, body_html: str | None = None) -> bool:
    if not SMTP_USER or not SMTP_PASS:
        print(f"\n{'='*60}")
        print(f"[EMAIL — would send to {NOTIFY_EMAIL}]")
        print(f"Subject: {subject}")
        print(body_text)
        print("="*60)
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = NOTIFY_EMAIL
        msg.attach(MIMEText(body_text, "plain"))
        if body_html:
            msg.attach(MIMEText(body_html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, NOTIFY_EMAIL, msg.as_string())
        print(f"  ✓ Email sent to {NOTIFY_EMAIL}")
        return True
    except Exception as e:
        print(f"  ✗ Email failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Notification builders
# ---------------------------------------------------------------------------

def notify_new_signal(ticker: str, ann: str, td: int, score: float) -> None:
    subject = f"[TradeArena] BUY signal: {ticker} — D-{td} (score {score:.2f})"
    text = f"""
PRE-EARNINGS BUY SIGNAL
=======================
Ticker:         {ticker}
Earnings date:  {ann}
Trading days:   {td} days until earnings (D-{td})
Dynamic score:  {score:.3f} (threshold 1.05)
Action:         BUY at 35% of portfolio today
Exit:           1 trading day before {ann}
Stop:           -5% from entry price

Strategy: S2-Safer pre-earnings drift
Expected: +27.3%/yr avg, worst year -3.1%

Run the arena or execute manually.
""".strip()
    html = f"""
<h2>🟢 Pre-Earnings Buy Signal: {ticker}</h2>
<table border="1" cellpadding="6" style="border-collapse:collapse;font-family:monospace">
<tr><td><b>Ticker</b></td><td>{ticker}</td></tr>
<tr><td><b>Earnings date</b></td><td>{ann}</td></tr>
<tr><td><b>Trading days left</b></td><td>D-{td}</td></tr>
<tr><td><b>Dynamic score</b></td><td>{score:.3f} ✓ (≥1.05)</td></tr>
<tr><td><b>Action</b></td><td>BUY at 35% of portfolio</td></tr>
<tr><td><b>Exit</b></td><td>D-1 before {ann}</td></tr>
<tr><td><b>Stop</b></td><td>-5% from entry</td></tr>
</table>
<p style="color:#888;font-size:12px">TradeArena signal monitor — {date.today()}</p>
"""
    send_email(subject, text, html)


def notify_exit_signal(ticker: str, ann: str, pnl_hint: str = "") -> None:
    subject = f"[TradeArena] EXIT signal: {ticker} — earnings tomorrow"
    text = f"""
PRE-EARNINGS EXIT SIGNAL
========================
Ticker:    {ticker}
Earnings:  {ann} (TOMORROW)
Action:    SELL full position today (D-1 exit)
Reason:    Do not hold through earnings — binary event risk
{pnl_hint}

Strategy rule: always exit 1 trading day before earnings announcement.
""".strip()
    send_email(subject, text)


def notify_regime_change(ok: bool, qqq: float, ma150: float) -> None:
    if ok:
        subject = f"[TradeArena] Regime ON — QQQ {qqq:.0f} back above 150dma {ma150:.0f}"
        text = f"QQQ {qqq:.2f} has crossed ABOVE 150dma {ma150:.2f}.\nRegime filter is ON. Resume new entries."
    else:
        subject = f"[TradeArena] Regime OFF — QQQ {qqq:.0f} below 150dma {ma150:.0f}"
        text = f"QQQ {qqq:.2f} has dropped BELOW 150dma {ma150:.2f}.\nRegime filter is OFF. Hold cash, no new entries.\nClose open positions if they trigger stops."
    send_email(subject, text)


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def run() -> None:
    today   = date.today().isoformat()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\nSignal monitor — {now_str}")
    print("="*50)

    state = load_state()

    # 1. Regime check
    print("Checking regime (QQQ vs 150dma)...")
    regime = check_regime()
    print(f"  QQQ {regime['qqq']}  150dma {regime['ma150']}  ok={regime['ok']}")

    last_regime = state.get("last_regime_ok")
    if last_regime is not None and last_regime != regime["ok"]:
        print(f"  ⚡ Regime CHANGED: {last_regime} → {regime['ok']}")
        notify_regime_change(regime["ok"], regime["qqq"], regime["ma150"])
        state["regime_alert_sent"] = today
    state["last_regime_ok"] = regime["ok"]

    # 2. Dynamic scores
    print("Computing dynamic scores...")
    try:
        scores = compute_scores()
        for t, s in sorted(scores.items(), key=lambda x: -x[1]):
            flag = " ✓" if s >= SCORE_THRESHOLD else ""
            print(f"  {t:<6} {s:.3f}{flag}")
    except Exception as e:
        print(f"  Score computation failed: {e}")
        scores = {}

    # 3. Earnings window check
    print("Checking D-20 windows...")
    current_signals: dict[str, dict] = {}

    for ticker in WATCH_TICKERS:
        result = get_upcoming_earnings(ticker)
        if not result:
            continue
        ann, td = result
        score   = scores.get(ticker, BASE_QUALITY.get(ticker, 1.0))
        in_win  = 10 <= td <= 25
        passes  = in_win and regime["ok"] and score >= SCORE_THRESHOLD

        if in_win:
            current_signals[ticker] = {"ann": ann, "td": td, "score": score, "passes": passes}
            status = "✓ BUY" if passes else f"skip (score {score:.2f})" if score < SCORE_THRESHOLD else "skip (regime off)"
            print(f"  {ticker:<6} D-{td:<3} {ann}  score {score:.3f}  {status}")
        elif td <= 1:
            print(f"  {ticker:<6} D-{td}  EXIT DUE  {ann}")

    # 4. New signal alerts
    prev_signals = state.get("active_signals", {})
    for ticker, sig in current_signals.items():
        if not sig["passes"]:
            continue
        prev = prev_signals.get(ticker, {})
        already_alerted = prev.get("alerted_date") == today
        if not already_alerted:
            print(f"  → NEW SIGNAL: {ticker} D-{sig['td']} score {sig['score']:.3f}")
            notify_new_signal(ticker, sig["ann"], sig["td"], sig["score"])
            current_signals[ticker]["alerted_date"] = today
        else:
            print(f"  → {ticker} already alerted today")
            current_signals[ticker]["alerted_date"] = prev["alerted_date"]

    # 5. Exit alerts for tickers that have LEFT the window (td < 10 = approaching earnings)
    for ticker in WATCH_TICKERS:
        result = get_upcoming_earnings(ticker)
        if not result:
            continue
        ann, td = result
        if td <= 1:
            prev = prev_signals.get(ticker, {})
            if prev.get("alerted_date") and prev.get("exit_alerted_date") != today:
                print(f"  → EXIT ALERT: {ticker} earnings {ann} is tomorrow")
                notify_exit_signal(ticker, ann)
                current_signals[ticker] = {**prev, "exit_alerted_date": today}

    # 6. Summary
    state["active_signals"] = current_signals
    state["last_run"]       = now_str
    save_state(state)

    active = [t for t, s in current_signals.items() if s.get("passes")]
    print()
    if active:
        print(f"✓ Active buy signals: {', '.join(active)}")
    elif not regime["ok"]:
        print("⚠ Regime OFF — hold cash")
    else:
        print("○ No signals in window today")
    print(f"State saved → {STATE_FILE}")


if __name__ == "__main__":
    run()

