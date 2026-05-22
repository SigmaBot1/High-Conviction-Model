"""
AUTO-SCAN: Runs watchlist through diagnose.py and alerts on 11+ scores
============================================================================
Place in your high-conviction-model/ folder.

Manual run:     python auto_scan.py
Scheduled run:  Use Windows Task Scheduler (see instructions at bottom)

Set these environment variables for alerts:
  GMAIL_ADDRESS=your_email@gmail.com       (for email alerts)
  GMAIL_APP_PASSWORD=your_app_password     (for email alerts)
  SLACK_WEBHOOK_URL=https://hooks.slack... (for Slack alerts, see bottom of file)
"""

import subprocess
import os
import sys
import re
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
ALERT_THRESHOLD = 11  # Alert when score >= this (out of 12)
LOG_FILE = os.path.join(BASE_DIR, 'output', 'scan_log.txt')


def load_watchlist():
    """Load tickers from watchlist.txt."""
    if not os.path.exists(WATCHLIST_FILE):
        print(f"ERROR: {WATCHLIST_FILE} not found.")
        return []
    with open(WATCHLIST_FILE, 'r') as f:
        tickers = [line.strip().upper() for line in f 
                   if line.strip() and not line.startswith('#')]
    return tickers


def run_diagnose(tickers):
    """Run diagnose.py on EACH ticker individually to guarantee correct attribution."""
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
            
            # Get last 25 lines for detail
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
    """Parse diagnose output for a SINGLE ticker. Returns (score, total, conviction)."""
    score = 0
    total = 12
    conviction = 'UNKNOWN'
    
    for line in output.split('\n'):
        # Match patterns like "PASSES 9 of 12" or "9/12 filter" or "9 of 12 filter"
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
        
        # Match "Score: X/Y" pattern
        m4 = re.search(r'Score[:\s]+(\d+)\s*/\s*(\d+)', line, re.IGNORECASE)
        if m4:
            score = int(m4.group(1))
            total = int(m4.group(2))
        
        # Conviction
        upper = line.upper()
        if 'DO NOT ENTER' in upper:
            conviction = 'DO NOT ENTER'
        elif 'HIGH' in upper and ('CONVICTION' in upper or 'SIGNAL' in upper):
            conviction = 'HIGH'
        elif 'MEDIUM' in upper and ('CONVICTION' in upper or score >= 9):
            conviction = 'MEDIUM'
    
    # Fallback conviction from score
    if conviction == 'UNKNOWN':
        if score >= 11:
            conviction = 'HIGH'
        elif score >= 9:
            conviction = 'MEDIUM'
        else:
            conviction = 'DO NOT ENTER'
    
    return score, total, conviction


def send_alert(alerts, full_output=""):
    """Send alert via email and/or Slack."""
    subject_tickers = ', '.join([a['ticker'] for a in alerts])
    
    body = f"""HIGH-CONVICTION MODEL - SIGNAL ALERT
{'='*50}
Time: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}

STOCKS AT {ALERT_THRESHOLD}/12 OR HIGHER:
{'='*50}
"""
    for alert in alerts:
        body += f"\n{alert['ticker']}: {alert['score']}/{alert['total']} ({alert['conviction']})\n"
        body += f"{'-'*40}\n"
        body += alert['detail'] + "\n"
    
    body += f"\n{'='*50}\nAction: Run full manual review (megatrend assessment + dip/crash framework)\n"
    
    # Try Slack first
    slack_url = os.environ.get('SLACK_WEBHOOK_URL', '')
    if slack_url:
        send_slack(slack_url, subject_tickers, alerts)
    
    # Then email
    gmail_addr = os.environ.get('GMAIL_ADDRESS', '')
    gmail_pass = os.environ.get('GMAIL_APP_PASSWORD', '')
    
    if gmail_addr and gmail_pass:
        subject = f"HIGH-CONVICTION ALERT: {subject_tickers} at {alerts[0]['score']}/12+"
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
        print("Set SLACK_WEBHOOK_URL or GMAIL_ADDRESS + GMAIL_APP_PASSWORD")


def send_slack(webhook_url, tickers_str, alerts):
    """Send alert to Slack via webhook."""
    import requests
    
    blocks = [{
        "type": "header",
        "text": {"type": "plain_text", "text": f"\u26A1 HIGH-CONVICTION ALERT: {tickers_str}"}
    }]
    
    for alert in alerts:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{alert['ticker']}*: {alert['score']}/{alert['total']} ({alert['conviction']})\n```{alert['detail'][:500]}```"
            }
        })
    
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Action:* Run full manual review (megatrend + dip/crash framework)"}
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
    
    # Run diagnose on each ticker individually
    results = run_diagnose(tickers)
    
    if not results:
        print("ERROR: No results from diagnose.")
        sys.exit(1)
    
    # Display results
    print(f"\n{'TICKER':<8} {'SCORE':>6} {'CONVICTION':<15} {'STATUS'}")
    print("-" * 50)
    
    alerts = []
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
            alerts.append(r)
    
    print()
    
    # Log results
    log_scan(results)
    print(f"Scan logged to {LOG_FILE}")
    
    # Send alerts if any
    if alerts:
        print(f"\n{'!'*50}")
        print(f"  {len(alerts)} STOCK(S) AT {ALERT_THRESHOLD}/12 OR HIGHER!")
        print(f"{'!'*50}\n")
        send_alert(alerts)
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
# This will scan every 30 minutes during market hours and email you
# if any watchlist stock hits 11/12 or higher.
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
