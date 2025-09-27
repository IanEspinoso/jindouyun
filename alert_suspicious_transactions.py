"""
Monitors a SQLite database of card-not-present transactions and emits e-mail
alerts for suspicious activity based on three rules:

1) Velocity: too many attempts by the same user in a short time window
2) High amount: transaction (or rolling sum) exceeds a threshold in a window
3) Prior chargeback: user has historical chargeback => auto-flag new tx

Assumptions:
- SQLite DB file: cc_transactions.db
- Table `transactions` has at least:
    id (INTEGER PRIMARY KEY),
    time (TEXT, 'YYYY-MM-DD HH:MM:SS'),
    user_id (INTEGER),
    device_id (INTEGER NULL),
    merchant_id (INTEGER),
    transaction_amount (REAL),
    has_cbk (INTEGER or BOOLEAN 0/1)  -- historical info may arrive later
- Existing helper: email_alert.send_email_alert(message: str)

Run:
    python alert_suspicious_transactions.py
"""

import sqlite3
import threading
import time as t
import datetime as dt

import email_alert  # must expose: send_email_alert(str)

# --------------------------- Config ---------------------------

DB_PATH = "cc_transactions.db"

# Velocity rule: e.g., > 5 transactions in 2 minutes (per user)
VELOCITY_WINDOW_MIN = 2
VELOCITY_MAX_TX = 5

# High-amount rule: flag any single transaction over this amount
SINGLE_AMOUNT_THRESHOLD = 1500.00
# Optional rolling sum threshold per user within a window (set None to disable)
ROLLING_WINDOW_MIN = 10
ROLLING_SUM_THRESHOLD = 3000.00

# Polling interval (seconds)
POLL_INTERVAL_SEC = 30

# ------------------------- DB Helpers -------------------------

def get_conn():
    return sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)  # autocommit

def ensure_alerts_table():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                tx_id INTEGER NOT NULL,
                user_id INTEGER,
                device_id INTEGER,
                rule TEXT NOT NULL,
                severity TEXT NOT NULL,
                details TEXT NOT NULL
            )
        """)

def get_last_processed_id() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT MAX(tx_id) FROM transactions_alerts").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

def fetch_new_transactions(last_id: int):
    """
    Returns rows as dicts ordered by id.
    """
    q = """
        SELECT id, time, user_id, device_id, merchant_id, transaction_amount, has_cbk
        FROM transactions
        WHERE id > ?
        ORDER BY id ASC
    """
    with get_conn() as conn:
        cur = conn.execute(q, (last_id,))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows

# ---------------------- Rule Implementations ------------------

def parse_time(ts: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def has_prior_chargeback(user_id: int, at_time: dt.datetime) -> bool:
    """
    Any historical chargeback for this user strictly before current tx time?
    """
    q = """
        SELECT 1
        FROM transactions
        WHERE user_id = ?
          AND has_cbk IN (1, TRUE)
          AND time < ?
        LIMIT 1
    """
    with get_conn() as conn:
        row = conn.execute(q, (user_id, at_time.strftime("%Y-%m-%d %H:%M:%S"))).fetchone()
    return row is not None

def velocity_exceeded(user_id: int, at_time: dt.datetime) -> tuple[bool, int]:
    """
    Count tx by user within [at_time - window, at_time] inclusive.
    """
    start = at_time - dt.timedelta(minutes=VELOCITY_WINDOW_MIN)
    q = """
        SELECT COUNT(*)
        FROM transactions
        WHERE user_id = ?
          AND time >= ?
          AND time <= ?
    """
    with get_conn() as conn:
        (cnt,) = conn.execute(q, (
            user_id,
            start.strftime("%Y-%m-%d %H:%M:%S"),
            at_time.strftime("%Y-%m-%d %H:%M:%S"),
        )).fetchone()
    return (cnt > VELOCITY_MAX_TX, cnt)

def high_amount_single(amount: float) -> bool:
    return amount is not None and amount >= SINGLE_AMOUNT_THRESHOLD

def rolling_sum_exceeded(user_id: int, at_time: dt.datetime) -> tuple[bool, float]:
    """
    Sum transaction_amount for user in [at_time - ROLLING_WINDOW_MIN, at_time]
    """
    if ROLLING_SUM_THRESHOLD is None:
        return (False, 0.0)
    start = at_time - dt.timedelta(minutes=ROLLING_WINDOW_MIN)
    q = """
        SELECT COALESCE(SUM(transaction_amount), 0.0)
        FROM transactions
        WHERE user_id = ?
          AND time >= ?
          AND time <= ?
    """
    with get_conn() as conn:
        (total,) = conn.execute(q, (
            user_id,
            start.strftime("%Y-%m-%d %H:%M:%S"),
            at_time.strftime("%Y-%m-%d %H:%M:%S"),
        )).fetchone()
    return (total >= ROLLING_SUM_THRESHOLD, float(total))

# --------------------------- Alerts ---------------------------

def record_alert(tx_id: int, user_id: int | None, device_id: int | None,
                 rule: str, severity: str, details: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO transactions_alerts (created_at, tx_id, user_id, device_id, rule, severity, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            tx_id, user_id, device_id, rule, severity, details
        ))

def send_alert_email(rule: str, tx: dict, details: str, severity: str = "medium"):
    # Keep the message short and actionable; enrich as needed
    msg = (
        f"[Fraud Alert][{severity.upper()}] {rule}\n"
        f"tx_id={tx.get('id')} user_id={tx.get('user_id')} device_id={tx.get('device_id')} "
        f"merchant_id={tx.get('merchant_id')} amount={tx.get('transaction_amount')}\n"
        f"time={tx.get('time')}\n"
        f"details: {details}"
    )
    try:
        email_alert.send_email_alert(msg)
    except Exception as e:
        # Fallback: log to stdout; in real systems, log to file/observability stack
        print(f"[WARN] Failed to send email: {e}\n{msg}")

# ------------------------- Main Monitor -----------------------

def process_transaction(tx: dict):
    """
    Apply rules to a single transaction and emit alerts as needed.
    """
    tx_id = int(tx["id"])
    user_id = int(tx["user_id"]) if tx.get("user_id") is not None else None
    device_id = int(tx["device_id"]) if tx.get("device_id") is not None else None
    amount = float(tx["transaction_amount"]) if tx.get("transaction_amount") is not None else None
    when = parse_time(tx.get("time") or "")

    if when is None:
        # Invalid timestamp -> low-confidence but worth flagging
        details = "Invalid transaction time format"
        record_alert(tx_id, user_id, device_id, rule="data_integrity", severity="low", details=details)
        send_alert_email("data_integrity", tx, details, severity="low")
        return

    # Rule 1: Prior chargeback history
    if user_id is not None and has_prior_chargeback(user_id, when):
        details = "User has historical chargeback before this transaction"
        record_alert(tx_id, user_id, device_id, rule="prior_chargeback", severity="high", details=details)
        send_alert_email("prior_chargeback", tx, details, severity="high")

    # Rule 2: Velocity
    if user_id is not None:
        vel_flag, cnt = velocity_exceeded(user_id, when)
        if vel_flag:
            details = f"Velocity exceeded: {cnt} tx within last {VELOCITY_WINDOW_MIN} min (threshold={VELOCITY_MAX_TX})"
            record_alert(tx_id, user_id, device_id, rule="velocity", severity="medium", details=details)
            send_alert_email("velocity", tx, details, severity="medium")

    # Rule 3: High amount (single)
    if high_amount_single(amount):
        details = f"High single amount: {amount:.2f} >= {SINGLE_AMOUNT_THRESHOLD:.2f}"
        record_alert(tx_id, user_id, device_id, rule="high_amount_single", severity="medium", details=details)
        send_alert_email("high_amount_single", tx, details, severity="medium")

    # Rule 4 (optional): Rolling sum in window
    if user_id is not None and ROLLING_SUM_THRESHOLD is not None:
        roll_flag, total = rolling_sum_exceeded(user_id, when)
        if roll_flag:
            details = (
                f"Rolling sum exceeded in {ROLLING_WINDOW_MIN} min: "
                f"{total:.2f} >= {ROLLING_SUM_THRESHOLD:.2f}"
            )
            record_alert(tx_id, user_id, device_id, rule="rolling_sum", severity="medium", details=details)
            send_alert_email("rolling_sum", tx, details, severity="medium")

def monitor_loop():
    ensure_alerts_table()
    last_id = get_last_processed_id()
    print(f"[monitor] starting from last processed tx_id = {last_id}")

    while True:
        try:
            new_txs = fetch_new_transactions(last_id)
            if new_txs:
                for tx in new_txs:
                    process_transaction(tx)
                last_id = new_txs[-1]["id"]
            t.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\n[monitor] stopped by user")
            break
        except Exception as e:
            print(f"[monitor][ERROR] {e}")
            # brief backoff on error
            t.sleep(5)

# --------------------------- Entrypoint -----------------------

if __name__ == "__main__":
    # Run in a thread to mirror your style; could also run directly.
    th = threading.Thread(target=monitor_loop, daemon=True)
    th.start()

    # Keep the main thread alive
    try:
        while True:
            t.sleep(1)
    except KeyboardInterrupt:
        print("\n[main] shutting down...")
