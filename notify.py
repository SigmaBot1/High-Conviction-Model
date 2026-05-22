"""
Notification layer for the High-Conviction live scanner.

Supports two delivery channels:
  - Email via Gmail SMTP  (always attempted when credentials are present)
  - Slack webhook         (optional; only used when SLACK_WEBHOOK_URL is set)

Environment variables
---------------------
  GMAIL_ADDRESS       Sender / recipient address (e.g. you@gmail.com)
  GMAIL_APP_PASSWORD  Gmail App Password (NOT your account password).
                      Enable at: https://myaccount.google.com/apppasswords
  SLACK_WEBHOOK_URL   Incoming-webhook URL.  Optional.

Usage
-----
  from notify import Signal, Notifier

  notifier = Notifier()
  notifier.notify(signals)          # list of Signal objects (may be empty)
"""
import json
import logging
import os
import smtplib
from dataclasses import dataclass, field
from datetime import date as dt_date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filter display labels (same order as signals.py)
# ---------------------------------------------------------------------------

_FILTER_LABELS: Dict[str, str] = {
    "f1_rev_growth":   "F1  Revenue growth >15% YoY",
    "f2_gross_margin": "F2  Gross margin >40%",
    "f3_roic":         "F3  ROIC >15%",
    "f4_de_ratio":     "F4  D/E ratio <1×",
    "f5_fcf":          "F5  Positive FCF / cash runway",
    "f6_peg_ps":       "F6  PEG <2 or P/S <15",
    "f7_fcf_yield":    "F7  FCF yield >1.5%",
    "f8_fear_regime":  "F8  Fear regime (VIX / SPY drawdown)",
    "f9_price_disloc": "F9  Price dislocation",
    "f10_momentum":    "F10 Momentum not bottom decile",
    "f11_ps_quartile": "F11 P/S in bottom quartile (3yr range)",
    "f12_above_ema50": "F12 Near 50-day EMA",
}

_FILTER_ORDER = list(_FILTER_LABELS.keys())


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """All information about a single entry signal."""
    ticker: str
    scan_date: dt_date

    # Price
    price: float
    pct_off_52w_high: Optional[float] = None    # e.g. 0.25 → 25% below high
    price_zscore: Optional[float] = None

    # Fundamental snapshot
    ps_ratio: Optional[float] = None
    roic: Optional[float] = None
    gross_margin: Optional[float] = None

    # Market context
    vix: Optional[float] = None

    # All 12 filter results  {filter_key: True | False | None}
    filters: Dict[str, Optional[bool]] = field(default_factory=dict)

    @property
    def conviction(self) -> str:
        """
        Conviction level based on how strongly the fear regime and
        price dislocation conditions are triggered.

        EXTREME  – VIX ≥ 35 AND price >30% off 52-week high
        HIGH     – VIX ≥ 25 AND price >20% off 52-week high (normal signal)
        MODERATE – Fear regime met via SPY drawdown only (VIX < 25)
        """
        vix = self.vix or 0.0
        off_high = self.pct_off_52w_high or 0.0

        if vix >= 35 and off_high >= 0.30:
            return "EXTREME"
        if vix >= 25 and off_high >= 0.20:
            return "HIGH"
        return "MODERATE"

    @property
    def n_filters_passed(self) -> int:
        return sum(1 for v in self.filters.values() if v is True)

    @property
    def n_filters_total(self) -> int:
        return len(self.filters)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(val: Optional[float], precision: int = 1) -> str:
    if val is None:
        return "n/a"
    return f"{val * 100:.{precision}f}%"


def _fmt_float(val: Optional[float], precision: int = 2) -> str:
    if val is None:
        return "n/a"
    return f"{val:.{precision}f}"


def _filter_row_text(key: str, result: Optional[bool]) -> str:
    label = _FILTER_LABELS.get(key, key)
    if result is True:
        icon = "[PASS]"
    elif result is False:
        icon = "[FAIL]"
    else:
        icon = "[ -- ]"
    return f"  {icon}  {label}"


def _filter_row_html(key: str, result: Optional[bool]) -> str:
    label = _FILTER_LABELS.get(key, key)
    if result is True:
        icon, color = "&#10003;", "#2e7d32"   # checkmark, dark green
    elif result is False:
        icon, color = "&#10007;", "#c62828"   # cross, dark red
    else:
        icon, color = "&#8211;",  "#757575"   # en-dash, grey
    return (
        f'<tr>'
        f'<td style="padding:2px 8px;font-weight:bold;color:{color};font-size:14px">{icon}</td>'
        f'<td style="padding:2px 4px;color:#333;font-size:13px">{label}</td>'
        f'</tr>'
    )


def _signal_text_block(sig: Signal) -> str:
    lines = [
        f"{'='*56}",
        f"  {sig.ticker}  |  ${sig.price:.2f}  |  {sig.conviction} CONVICTION",
        f"  Scan date: {sig.scan_date}",
        f"{'='*56}",
        "",
        "  Market context",
        f"    VIX:              {_fmt_float(sig.vix, 1)}",
        f"    % off 52-week hi: {_fmt_pct(sig.pct_off_52w_high)}",
        f"    Price z-score:    {_fmt_float(sig.price_zscore, 2)}",
        "",
        "  Fundamentals",
        f"    P/S ratio:        {_fmt_float(sig.ps_ratio, 1)}×",
        f"    ROIC:             {_fmt_pct(sig.roic)}",
        f"    Gross margin:     {_fmt_pct(sig.gross_margin)}",
        "",
        f"  Filters  ({sig.n_filters_passed}/{sig.n_filters_total} passed)",
    ]
    for key in _FILTER_ORDER:
        if key in sig.filters:
            lines.append(_filter_row_text(key, sig.filters[key]))
    lines.append("")
    return "\n".join(lines)


def _signal_html_card(sig: Signal) -> str:
    conviction_color = {
        "EXTREME": "#b71c1c",
        "HIGH":    "#1565c0",
        "MODERATE":"#e65100",
    }.get(sig.conviction, "#333")

    filter_rows_html = "\n".join(
        _filter_row_html(key, sig.filters.get(key))
        for key in _FILTER_ORDER
        if key in sig.filters
    )

    return f"""
<div style="border:2px solid {conviction_color};border-radius:8px;
            padding:16px 20px;margin:16px 0;font-family:Arial,sans-serif;
            background:#fafafa;max-width:560px">
  <div style="display:flex;align-items:baseline;gap:12px">
    <span style="font-size:22px;font-weight:bold;color:#111">{sig.ticker}</span>
    <span style="font-size:18px;color:#333">${sig.price:.2f}</span>
    <span style="font-size:13px;font-weight:bold;color:{conviction_color};
                 border:1px solid {conviction_color};border-radius:4px;
                 padding:1px 6px">{sig.conviction}</span>
  </div>
  <div style="color:#555;font-size:12px;margin-top:2px">{sig.scan_date}</div>

  <table style="margin:12px 0;border-collapse:collapse;width:100%">
    <tr>
      <td style="width:50%;vertical-align:top">
        <div style="font-size:11px;font-weight:bold;color:#888;
                    text-transform:uppercase;margin-bottom:4px">Market</div>
        <table style="font-size:13px;color:#333;border-collapse:collapse">
          <tr><td style="padding:1px 8px 1px 0;color:#666">VIX</td>
              <td style="padding:1px 0"><b>{_fmt_float(sig.vix, 1)}</b></td></tr>
          <tr><td style="padding:1px 8px 1px 0;color:#666">Off 52-week high</td>
              <td style="padding:1px 0"><b>{_fmt_pct(sig.pct_off_52w_high)}</b></td></tr>
          <tr><td style="padding:1px 8px 1px 0;color:#666">Price z-score</td>
              <td style="padding:1px 0"><b>{_fmt_float(sig.price_zscore, 2)}</b></td></tr>
        </table>
      </td>
      <td style="width:50%;vertical-align:top">
        <div style="font-size:11px;font-weight:bold;color:#888;
                    text-transform:uppercase;margin-bottom:4px">Fundamentals</div>
        <table style="font-size:13px;color:#333;border-collapse:collapse">
          <tr><td style="padding:1px 8px 1px 0;color:#666">P/S ratio</td>
              <td style="padding:1px 0"><b>{_fmt_float(sig.ps_ratio, 1)}×</b></td></tr>
          <tr><td style="padding:1px 8px 1px 0;color:#666">ROIC</td>
              <td style="padding:1px 0"><b>{_fmt_pct(sig.roic)}</b></td></tr>
          <tr><td style="padding:1px 8px 1px 0;color:#666">Gross margin</td>
              <td style="padding:1px 0"><b>{_fmt_pct(sig.gross_margin)}</b></td></tr>
        </table>
      </td>
    </tr>
  </table>

  <div style="font-size:11px;font-weight:bold;color:#888;
              text-transform:uppercase;margin-bottom:4px">
    Filters &nbsp;<span style="color:{conviction_color}">{sig.n_filters_passed}/{sig.n_filters_total}</span>
  </div>
  <table style="border-collapse:collapse">
    {filter_rows_html}
  </table>
</div>
"""


def build_email(
    signals: List[Signal],
    notify_on_empty: bool = False,
    scan_date: Optional[dt_date] = None,
) -> Optional[tuple]:
    """
    Build (subject, text_body, html_body).
    Returns None if there are no signals and notify_on_empty is False.
    """
    today = scan_date or dt_date.today()

    if not signals:
        if not notify_on_empty:
            return None
        subject   = f"High-Conviction Scanner – No signals [{today}]"
        text_body = f"No entry signals fired on {today}.\n\nAll filters reviewed. No stocks passed all 12 criteria today.\n"
        html_body = f"""
<html><body style="font-family:Arial,sans-serif;color:#333;max-width:600px">
  <h2 style="color:#555">High-Conviction Scanner</h2>
  <p style="color:#777">No entry signals on <b>{today}</b>.</p>
  <p style="color:#aaa;font-size:12px">All 12 filters reviewed. No stocks passed all criteria today.</p>
</body></html>"""
        return subject, text_body, html_body

    n = len(signals)
    tickers_str = ", ".join(s.ticker for s in signals)
    subject = f"High-Conviction Scanner – {n} signal{'s' if n > 1 else ''}: {tickers_str} [{today}]"

    # ---- Plain text ----
    text_lines = [
        f"High-Conviction Entry Signal{'s' if n > 1 else ''}",
        f"Scan date: {today}",
        f"{'='*56}",
        "",
    ]
    for sig in signals:
        text_lines.append(_signal_text_block(sig))

    text_lines += [
        "---",
        "This is an automated alert from the High-Conviction Stock Scanner.",
        "Not financial advice. Always do your own research.",
    ]
    text_body = "\n".join(text_lines)

    # ---- HTML ----
    cards_html = "\n".join(_signal_html_card(s) for s in signals)
    html_body = f"""
<html>
<body style="font-family:Arial,sans-serif;color:#333;max-width:620px;
             margin:0 auto;padding:16px">
  <h2 style="color:#111;margin-bottom:4px">
    High-Conviction Entry Signal{'s' if n > 1 else ''}
  </h2>
  <p style="color:#777;margin-top:0">Scan date: {today}</p>
  {cards_html}
  <hr style="border:none;border-top:1px solid #eee;margin:24px 0">
  <p style="color:#aaa;font-size:11px">
    Automated alert &mdash; not financial advice.
  </p>
</body>
</html>"""

    return subject, text_body, html_body


def build_slack_payload(
    signals: List[Signal],
    notify_on_empty: bool = False,
    scan_date: Optional[dt_date] = None,
) -> Optional[dict]:
    """
    Build a Slack Block Kit payload dict.
    Returns None if there are no signals and notify_on_empty is False.
    """
    today = scan_date or dt_date.today()

    if not signals:
        if not notify_on_empty:
            return None
        return {
            "text": f"High-Conviction Scanner: no signals on {today}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":white_check_mark: *High-Conviction Scanner* — no signals on {today}",
                    },
                }
            ],
        }

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f":bell: High-Conviction Signal{'s' if len(signals) > 1 else ''}  —  {today}",
            },
        }
    ]

    for sig in signals:
        conviction_emoji = {
            "EXTREME": ":rotating_light:",
            "HIGH":    ":large_blue_circle:",
            "MODERATE":":large_yellow_circle:",
        }.get(sig.conviction, ":white_circle:")

        passed = [
            _FILTER_LABELS.get(k, k)
            for k, v in sig.filters.items()
            if v is True
        ]
        passed_str = "\n".join(f"  • {lbl}" for lbl in passed)

        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{conviction_emoji} *{sig.ticker}*   `${sig.price:.2f}`   "
                        f"*{sig.conviction}* conviction\n"
                        f">VIX *{_fmt_float(sig.vix, 1)}*  |  "
                        f"Off 52w high *{_fmt_pct(sig.pct_off_52w_high)}*  |  "
                        f"P/S *{_fmt_float(sig.ps_ratio, 1)}×*  |  "
                        f"ROIC *{_fmt_pct(sig.roic)}*  |  "
                        f"Gross margin *{_fmt_pct(sig.gross_margin)}*\n"
                        f"Filters passed ({sig.n_filters_passed}/{sig.n_filters_total}):\n"
                        f"{passed_str}"
                    ),
                },
            }
        )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "_Automated alert — not financial advice._",
                }
            ],
        }
    )

    tickers_str = ", ".join(s.ticker for s in signals)
    return {
        "text": f"High-Conviction signal(s): {tickers_str} on {today}",
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# Notifier – reads credentials from environment and dispatches
# ---------------------------------------------------------------------------

class Notifier:
    """
    Sends notifications via email (Gmail SMTP) and optionally Slack.

    Credentials are read from environment variables at construction time;
    no credentials are stored in code.

    Parameters
    ----------
    notify_on_empty : bool
        If True, send a short "no signals today" message when the signal
        list is empty.  Default False.
    """

    def __init__(self, notify_on_empty: bool = False):
        self.notify_on_empty = notify_on_empty
        self.gmail_address   = os.environ.get("GMAIL_ADDRESS", "")
        self.gmail_password  = os.environ.get("GMAIL_APP_PASSWORD", "")
        self.slack_url       = os.environ.get("SLACK_WEBHOOK_URL", "")

    @property
    def email_configured(self) -> bool:
        return bool(self.gmail_address and self.gmail_password)

    @property
    def slack_configured(self) -> bool:
        return bool(self.slack_url)

    # ------------------------------------------------------------------ #
    # Email
    # ------------------------------------------------------------------ #

    def send_email(
        self,
        signals: List[Signal],
        scan_date: Optional[dt_date] = None,
    ) -> bool:
        """
        Send a signal email via Gmail SMTP.
        Returns True on success, False on failure.
        """
        if not self.email_configured:
            logger.warning(
                "Email not configured. Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD."
            )
            return False

        result = build_email(signals, self.notify_on_empty, scan_date)
        if result is None:
            logger.debug("No signals and notify_on_empty=False; skipping email.")
            return True

        subject, text_body, html_body = result

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self.gmail_address
        msg["To"]      = self.gmail_address   # send to self; forward as desired
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.gmail_address, self.gmail_password)
                server.sendmail(
                    self.gmail_address,
                    self.gmail_address,
                    msg.as_string(),
                )
            logger.info(f"Email sent: {subject}")
            return True
        except smtplib.SMTPAuthenticationError:
            logger.error(
                "Gmail authentication failed. Make sure you are using an App Password "
                "(not your account password). "
                "Generate one at: https://myaccount.google.com/apppasswords"
            )
            return False
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Slack
    # ------------------------------------------------------------------ #

    def send_slack(
        self,
        signals: List[Signal],
        scan_date: Optional[dt_date] = None,
    ) -> bool:
        """
        Post to Slack via incoming webhook.
        Returns True on success, False on failure.
        """
        if not self.slack_configured:
            logger.debug("SLACK_WEBHOOK_URL not set; skipping Slack notification.")
            return True

        payload = build_slack_payload(signals, self.notify_on_empty, scan_date)
        if payload is None:
            logger.debug("No signals and notify_on_empty=False; skipping Slack.")
            return True

        try:
            resp = requests.post(
                self.slack_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("Slack notification sent.")
            return True
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Combined dispatch
    # ------------------------------------------------------------------ #

    def notify(
        self,
        signals: List[Signal],
        scan_date: Optional[dt_date] = None,
    ) -> None:
        """
        Send all configured notifications (email + Slack).
        Logs errors but never raises — the scanner must not crash on notify failure.
        """
        if not signals and not self.notify_on_empty:
            logger.info("No signals today and notify_on_empty=False. Nothing sent.")
            return

        n = len(signals)
        if n:
            tickers = ", ".join(s.ticker for s in signals)
            logger.info(f"Notifying: {n} signal(s) — {tickers}")
        else:
            logger.info("Sending empty-run notification.")

        if self.email_configured:
            self.send_email(signals, scan_date)
        else:
            logger.warning(
                "Email credentials not set (GMAIL_ADDRESS / GMAIL_APP_PASSWORD). "
                "Skipping email."
            )

        if self.slack_configured:
            self.send_slack(signals, scan_date)
