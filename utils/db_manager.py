"""
db_manager.py - إدارة قاعدة البيانات v17.0
- SQLite خفيف
- تتبع أحداث شامل بالوقت
- حفظ القرارات والتاريخ
"""
import sqlite3, json, os
from datetime import datetime

DB_PATH = "pricing_v17.db"


def get_db():
    """الحصول على اتصال قاعدة البيانات"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """تهيئة قاعدة البيانات"""
    conn = get_db()
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        page TEXT,
        event_type TEXT NOT NULL,
        details TEXT,
        product_name TEXT,
        action_taken TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        product_name TEXT NOT NULL,
        old_status TEXT,
        new_status TEXT,
        reason TEXT,
        decided_by TEXT DEFAULT 'user'
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS ai_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        prompt_hash TEXT UNIQUE,
        response TEXT,
        source TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        our_file TEXT,
        comp_file TEXT,
        total_products INTEGER,
        matched INTEGER,
        missing INTEGER,
        summary TEXT
    )""")

    conn.commit()
    conn.close()


def log_event(page, event_type, details="", product_name="", action=""):
    """تسجيل حدث"""
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO events (timestamp, page, event_type, details, product_name, action_taken) VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), page, event_type, details, product_name, action)
        )
        conn.commit()
        conn.close()
    except:
        pass


def log_decision(product_name, old_status, new_status, reason="", decided_by="user"):
    """تسجيل قرار"""
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO decisions (timestamp, product_name, old_status, new_status, reason, decided_by) VALUES (?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), product_name, old_status, new_status, reason, decided_by)
        )
        conn.commit()
        conn.close()
    except:
        pass


def log_analysis(our_file, comp_file, total, matched, missing, summary=""):
    """تسجيل تحليل"""
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO analysis_history (timestamp, our_file, comp_file, total_products, matched, missing, summary) VALUES (?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), our_file, comp_file, total, matched, missing, summary)
        )
        conn.commit()
        conn.close()
    except:
        pass


def get_events(page=None, limit=50):
    """جلب الأحداث"""
    try:
        conn = get_db()
        if page:
            rows = conn.execute(
                "SELECT * FROM events WHERE page=? ORDER BY id DESC LIMIT ?", (page, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except:
        return []


def get_decisions(product_name=None, limit=50):
    """جلب القرارات"""
    try:
        conn = get_db()
        if product_name:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE product_name LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{product_name}%", limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except:
        return []


def get_analysis_history(limit=10):
    """جلب تاريخ التحليلات"""
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except:
        return []


# تهيئة تلقائية
init_db()
