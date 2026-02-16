"""
db_manager.py - قاعدة بيانات SQLite خفيفة v16.0
"""
import sqlite3, pandas as pd, json
from datetime import datetime
from config import DB_PATH


class DatabaseManager:
    def __init__(self, path=None):
        self.path = path or DB_PATH
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS results(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product TEXT, our_price REAL, comp_name TEXT, comp_price REAL,
                diff REAL, match_score REAL, decision TEXT, competitor TEXT,
                brand TEXT, size REAL, risk TEXT, reasoning TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
            c.execute("""CREATE TABLE IF NOT EXISTS audit_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT, details TEXT, page TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP)""")
            c.execute("""CREATE TABLE IF NOT EXISTS settings(
                key TEXT PRIMARY KEY, value TEXT)""")

    def save_results(self, df):
        col_map = {
            "المنتج":"product","السعر":"our_price","اسم المنافس":"comp_name",
            "أقل سعر منافس":"comp_price","الفرق":"diff","نسبة التطابق":"match_score",
            "القرار":"decision","المنافس":"competitor","الماركة":"brand",
            "الحجم":"size","الخطورة":"risk","التفسير":"reasoning",
        }
        with self._conn() as c:
            for _, row in df.iterrows():
                vals = {en: row.get(ar, "") for ar, en in col_map.items()}
                cols = ",".join(vals.keys())
                phs = ",".join(["?" for _ in vals])
                c.execute(f"INSERT INTO results({cols}) VALUES({phs})", list(vals.values()))

    def get_all_results(self, limit=500):
        with self._conn() as c:
            return pd.read_sql(f"SELECT * FROM results ORDER BY id DESC LIMIT {limit}", c)

    def get_statistics(self):
        with self._conn() as c:
            cur = c.execute("SELECT COUNT(*), AVG(diff), AVG(match_score) FROM results")
            r = cur.fetchone()
            return {"total": r[0] or 0, "avg_price_diff": r[1] or 0, "avg_match_score": r[2] or 0}

    def clear_results(self):
        with self._conn() as c:
            c.execute("DELETE FROM results")

    def log_action(self, action, details="", page=""):
        with self._conn() as c:
            c.execute("INSERT INTO audit_log(action,details,page,timestamp) VALUES(?,?,?,?)",
                      (action, details, page, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def get_audit_log(self, limit=50):
        with self._conn() as c:
            return pd.read_sql(f"SELECT * FROM audit_log ORDER BY id DESC LIMIT {limit}", c)

    def save_setting(self, key, value):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, value))

    def get_setting(self, key, default=""):
        with self._conn() as c:
            cur = c.execute("SELECT value FROM settings WHERE key=?", (key,))
            r = cur.fetchone()
            return r[0] if r else default
