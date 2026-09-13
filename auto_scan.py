"""
AUTO-SCAN: Runs watchlist through diagnose.py and alerts ONLY on changes
============================================================================
Place in your high-conviction-model/ folder.

Manual run:     python auto_scan.py
Force alert:    python auto_scan.py --force   (sends alert even if no change)
Scheduled run:  Use Windows Task Scheduler (see instructions at bottom)

ALERTS ONLY FIRE WHEN SOMETHING CHANGES:
  - A stock newly crosses 11/12 or 12/12
  - A stock's score changes (e.g., 11→12 or 12→11)
  - A stock drops below threshold (removal notice)
  - Use --force to override and send regardless

Set these environment variables for alerts:
  GMAIL_ADDRESS=your_email@gmail.com       (for email alerts)
  GMAIL_APP_PASSWORD=your_app_password     (for email alerts)
  SLACK_WEBHOOK_URL=https://hooks.slack... (for Slack alerts, see bottom of file)
"""

import subprocess
import os
import sys
import re
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIAGNOSE_SCRIPT = os.path.join(BASE_DIR, 'diagnose.py')
WATCHLIST_FILE = os.path.join(BASE_DIR, 'watchlist.txt')
ALERT_THRESHOLD = 11
LOG_FILE = os.path.join(BASE_DIR, 'output', 'scan_log.txt')
ALERT_STATE_FILE = os.path.join(BASE_DIR, 'data_cache', 'alert_state.json')


def load_watchlist():
    """Load tickers from watchlist.txt."""
    if not os.path.exists(WATCHLIST_FILE):
        print(f"ERROR: {WATCHLIST_FILE} not found.")
        return []
    with open(WATCHLIST_FILE, 'r') as f:
        tickers = [line.strip().upper() for line in f
                   if line.strip() and not line.startswith('#')]
    return tickers


def load_last_alert_state():
    """Load the previous scan's alert state."""
    if os.path.exists(ALERT_STATE_FILE):
        try:
            with open(ALERT_STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_alert_state(state):
    """Save current alert state for next comparison."""
    os.makedirs(os.path.dirname(ALERT_STATE_FILE), exist_ok=True)
    with open(ALERT_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def run_diagnose(tickers):
    """Run diagnose.py on EACH ticker individually."""
    results = []
    for ticker in tickers:
        try:
            result = subprocess.run(
                ['python', DIAGNOSE_SCRIPT, ticker],
                capture_output=True, text=True, timeout=60,
                cwd=BASE_DIR, encoding='utf-8', errors='replace'
            )
            output = result.stdout if result.stdout else ""
            score, total, conviction = parse_single_ticker(output)

            lines = [l for l in output.split('\n') if l.strip()]
            detail = '\n'.join(lines[-25:])

            results.append({
                'ticker': ticker,
                'score': score,
                'total': total,
                'conviction': conviction,
                'detail': detail,
                'full_output': output,
            })
        except subprocess.TimeoutExpired:
            results.append({
                'ticker': ticker, 'score': 0, 'total': 12,
                'conviction': 'TIMEOUT', 'detail': 'Timed out', 'full_output': ''
            })
        except Exception as e:
            results.append({
                'ticker': ticker, 'score': 0, 'total': 12,
                'conviction': 'ERROR', 'detail': str(e), 'full_output': ''
            })
    return results


def parse_single_ticker(output):
    """Parse diagnose output for a SINGLE ticker."""
    score = 0
    total = 12
    conviction = 'UNKNOWN'

    for line in output.split('\n'):
        m = re.search(r'PASSES\s+(\d+)\s+of\s+(\d+)', line, re.IGNORECASE)
        if m:
            score = int(m.group(1))
            total = int(m.group(2))
        m2 = re.search(r'(\d+)\s*/\s*(\d+)\s*filter', line, re.IGNORECASE)
        if m2:
            score = int(m2.group(1))
            total = int(m2.group(2))
        m3 = re.search(r'(\d+)\s+of\s+(\d+)\s*filter', line, re.IGNORECASE)
        if m3:
            score = int(m3.group(1))
            total = int(m3.group(2))
        m4 = re.search(r'Score[:\s]+(\d+)\s*/\s*(\d+)', line, re.IGNORECASE)
        if m4:
            score = int(m4.group(1))
            total = int(m4.group(2))

        upper = line.upper()
        if 'DO NOT ENTER' in upper:
            conviction = 'DO NOT ENTER'
        elif 'HIGH' in upper and ('CONVICTION' in upper or 'SIGNAL' in upper):
            conviction = 'HIGH'
        elif 'MEDIUM' in upper and 'CONVICTION' in upper:
            conviction = 'MEDIUM'

    if conviction == 'UNKNOWN':
        if score >= 11:
            conviction = 'HIGH'
        elif score >= 9:
            conviction = 'MEDIUM'
        else:
            conviction = 'DO NOT ENTER'

    return score, total, conviction


def detect_changes(results, last_state):
    """Compare current results to last alert state. Return only changes."""
    current_state = {}
    new_alerts = []
    upgraded = []
    removed = []

    for r in results:
        if r['score'] >= ALERT_THRESHOLD:
            current_state[r['ticker']] = r['score']

            prev_score = last_state.get(r['ticker'], 0)

            if prev_score == 0:
                # New: wasn't at threshold before
                new_alerts.append(r)
            elif r['score'] != prev_score:
                # Score changed (e.g., 11→12 or 12→11)
                upgraded.append(r)
            # else: same score as last time → no alert

    # Check for stocks that dropped below threshold
    for ticker, prev_score in last_state.items():
        if ticker not in current_state:
            removed.append({'ticker': ticker, 'prev_score': prev_score})

    return new_alerts, upgraded, removed, current_state


def build_alert_body(new_alerts, upgraded, removed):
    """Build detailed alert body for both email and Slack."""
    lines = []
    lines.append(f"HIGH-CONVICTION MODEL — SIGNAL UPDATE")
    lines.append(f"{'='*55}")
    lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
    lines.append("")

    if new_alerts:
        lines.append(f"NEW SIGNALS ({len(new_alerts)}):")
        lines.append(f"{'='*55}")
        for a in new_alerts:
            lines.append(f"\n{'⚡' if a['score'] >= 12 else '🔥'} {a['ticker']}: {a['score']}/{a['total']} ({a['conviction']})")
            lines.append(f"{'-'*45}")
            lines.append(a['detail'])
            lines.append("")

    if upgraded:
        lines.append(f"\nSCORE CHANGES ({len(upgraded)}):")
        lines.append(f"{'='*55}")
        for a in upgraded:
            lines.append(f"\n📊 {a['ticker']}: now {a['score']}/{a['total']} ({a['conviction']})")
            lines.append(f"{'-'*45}")
            lines.append(a['detail'])
            lines.append("")

    if removed:
        lines.append(f"\nDROPPED BELOW {ALERT_THRESHOLD}/12 ({len(removed)}):")
        lines.append(f"{'='*55}")
        for r in removed:
            lines.append(f"  ⬇️  {r['ticker']} (was {r['prev_score']}/12, now below threshold)")
        lines.append("")

    if new_alerts or upgraded:
        lines.append(f"{'='*55}")
        lines.append("ACTION: Run full manual review (megatrend + dip/crash framework)")
        lines.append("If 12/12: read last 2 earnings transcripts, make buy/no-buy decision.")

    return '\n'.join(lines)


def send_alerts(new_alerts, upgraded, removed, all_threshold_results):
    """Send alert via both email and Slack with identical detail."""
    body = build_alert_body(new_alerts, upgraded, removed)

    # Build subject
    all_changes = new_alerts + upgraded
    if all_changes:
        tickers_str = ', '.join([a['ticker'] for a in all_changes])
        top_score = max(a['score'] for a in all_changes)
        subject = f"HIGH-CONVICTION {'⚡ SIGNAL' if top_score >= 12 else '🔥 UPDATE'}: {tickers_str}"
    elif removed:
        tickers_str = ', '.join([r['ticker'] for r in removed])
        subject = f"HIGH-CONVICTION: {tickers_str} dropped below threshold"
    else:
        return

    # Slack
    slack_url = os.environ.get('SLACK_WEBHOOK_URL', '')
    if slack_url:
        send_slack(slack_url, subject, body, new_alerts, upgraded, removed)

    # Email (same detail as Slack)
    gmail_addr = os.environ.get('GMAIL_ADDRESS', '')
    gmail_pass = os.environ.get('GMAIL_APP_PASSWORD', '')

    if gmail_addr and gmail_pass:
        try:
            msg = MIMEMultipart()
            msg['From'] = gmail_addr
            msg['To'] = gmail_addr
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(gmail_addr, gmail_pass)
                server.sendmail(gmail_addr, gmail_addr, msg.as_string())

            print(f"ALERT EMAIL SENT to {gmail_addr}")
        except Exception as e:
            print(f"ERROR sending email: {e}")

    if not slack_url and not gmail_addr:
        print("WARNING: No alert channels configured.")


def send_slack(webhook_url, subject, body, new_alerts, upgraded, removed):
    """Send detailed alert to Slack."""
    import requests

    blocks = [{
        "type": "header",
        "text": {"type": "plain_text", "text": subject[:150]}
    }]

    # New signals
    for a in new_alerts:
        emoji = ":zap:" if a['score'] >= 12 else ":fire:"
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{emoji} *{a['ticker']}*: {a['score']}/{a['total']} ({a['conviction']})\n```{a['detail'][:800]}```"
            }
        })

    # Score changes
    for a in upgraded:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":chart_with_upwards_trend: *{a['ticker']}* score changed → {a['score']}/{a['total']} ({a['conviction']})\n```{a['detail'][:800]}```"
            }
        })

    # Removals
    if removed:
        removal_text = '\n'.join([f"• {r['ticker']} (was {r['prev_score']}/12)" for r in removed])
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":arrow_down: *Dropped below threshold:*\n{removal_text}"
            }
        })

    # Action footer
    if new_alerts or upgraded:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Action:* Run full manual review (megatrend + dip/crash framework)"
            }
        })

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"Scan time: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"}]
    })

    try:
        resp = requests.post(webhook_url, json={"blocks": blocks}, timeout=10)
        if resp.status_code == 200:
            print("ALERT SENT TO SLACK")
        else:
            print(f"Slack error: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"ERROR sending Slack: {e}")


def log_scan(results):
    """Append scan results to log file."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"SCAN: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}\n")
        f.write(f"{'='*60}\n")
        for r in results:
            flag = " <-- ALERT" if r['score'] >= ALERT_THRESHOLD else ""
            f.write(f"  {r['ticker']}: {r['score']}/{r['total']} ({r['conviction']}){flag}\n")
        f.write(f"\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    force_alert = '--force' in sys.argv

    print()
    print("=" * 60)
    print(f"  HIGH-CONVICTION AUTO-SCAN  |  {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
    print("=" * 60)
    print()

    # Load watchlist
    tickers = load_watchlist()
    if not tickers:
        print("No tickers to scan. Add tickers to watchlist.txt.")
        sys.exit(1)

    print(f"Scanning {len(tickers)} tickers: {', '.join(tickers)}")
    print()

    # Run diagnose on each ticker
    results = run_diagnose(tickers)

    if not results:
        print("ERROR: No results from diagnose.")
        sys.exit(1)

    # Display results
    print(f"\n{'TICKER':<8} {'SCORE':>6} {'CONVICTION':<15} {'STATUS'}")
    print("-" * 50)

    threshold_results = []
    for r in results:
        status = ""
        if r['score'] >= 12:
            status = "*** CRASH SIGNAL ACTIVE ***"
        elif r['score'] >= 11:
            status = "** NEAR SIGNAL **"
        elif r['score'] >= 9:
            status = "Watch"
        else:
            status = "-"

        print(f"{r['ticker']:<8} {r['score']:>2}/{r['total']:<3} {r['conviction']:<15} {status}")

        if r['score'] >= ALERT_THRESHOLD:
            threshold_results.append(r)

    print()

    # Log results
    log_scan(results)
    print(f"Scan logged to {LOG_FILE}")

    # Load previous alert state
    last_state = load_last_alert_state()

    # Detect changes
    new_alerts, upgraded, removed, current_state = detect_changes(results, last_state)

    # Save new state
    save_alert_state(current_state)

    has_changes = len(new_alerts) > 0 or len(upgraded) > 0 or len(removed) > 0

    if has_changes:
        print(f"\n{'!'*50}")
        if new_alerts:
            print(f"  {len(new_alerts)} NEW STOCK(S) AT {ALERT_THRESHOLD}/12+")
        if upgraded:
            print(f"  {len(upgraded)} STOCK(S) CHANGED SCORE")
        if removed:
            print(f"  {len(removed)} STOCK(S) DROPPED BELOW THRESHOLD")
        print(f"{'!'*50}\n")

        send_alerts(new_alerts, upgraded, removed, threshold_results)

    elif force_alert and threshold_results:
        print(f"\n[--force] Sending alert for {len(threshold_results)} stocks at threshold...")
        # Treat all threshold results as new for force mode
        send_alerts(threshold_results, [], [], threshold_results)

    elif threshold_results:
        print(f"\n{len(threshold_results)} stock(s) at {ALERT_THRESHOLD}/12+ (unchanged from last scan — no alert sent)")
        for t in threshold_results:
            print(f"  {t['ticker']}: {t['score']}/{t['total']} ({t['conviction']})")
        print("Use --force to send alert regardless.")

    else:
        print(f"No stocks at {ALERT_THRESHOLD}/12 threshold. Market calm.")

    print("\nDone.")


# ---------------------------------------------------------------------------
# WINDOWS TASK SCHEDULER SETUP
# ---------------------------------------------------------------------------
# To run this automatically during market hours:
#
# 1. Open Task Scheduler (search "Task Scheduler" in Start menu)
# 2. Click "Create Basic Task"
# 3. Name: "High-Conviction Auto-Scan"
# 4. Trigger: Daily
# 5. Start time: 9:45 AM (15 min after market open)
# 6. In Advanced settings, check "Repeat task every: 30 minutes"
# 7. For a duration of: 7 hours (covers 9:45 AM to 4:45 PM)
# 8. Action: Start a program
# 9. Program: python
# 10. Arguments: C:\Users\mattp\Projects\my-project\high-conviction-model\auto_scan.py
# 11. Start in: C:\Users\mattp\Projects\my-project\high-conviction-model
#
# This will scan every 30 minutes during market hours and alert you
# ONLY when something changes (new signal, score change, or stock drops).
#
# ---------------------------------------------------------------------------
# SLACK SETUP (optional, easier than Gmail)
# ---------------------------------------------------------------------------
# 1. Go to api.slack.com/apps → Create New App → From scratch
# 2. Name it "High-Conviction Alerts", pick your workspace
# 3. Click "Incoming Webhooks" → Activate → "Add New Webhook to Workspace"
# 4. Pick the channel you want alerts in → Allow
# 5. Copy the webhook URL (starts with https://hooks.slack.com/services/...)
# 6. Set it permanently:
#    [System.Environment]::SetEnvironmentVariable("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/YOUR/WEBHOOK/URL", "User")
# 7. Restart VS Code. Alerts will go to both Slack AND email if both are set.
