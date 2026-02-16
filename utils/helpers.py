"""
helpers.py - أدوات مساعدة v17.0
- فلاتر متقدمة
- أزرار ذكية لكل قسم
- خاصية لصق
- تصدير Excel
- عمل في الخلفية
"""
import pandas as pd, io, threading, time
from datetime import datetime


# ===== فلاتر متقدمة =====
def apply_filters(df, filters):
    """تطبيق فلاتر متعددة على DataFrame"""
    result = df.copy()
    if not filters:
        return result

    # فلتر الماركة
    if filters.get("brand") and filters["brand"] != "الكل":
        result = result[result.get("الماركة", pd.Series(dtype=str)).str.contains(filters["brand"], case=False, na=False)]

    # فلتر المنافس
    if filters.get("competitor") and filters["competitor"] != "الكل":
        result = result[result.get("المنافس", pd.Series(dtype=str)).str.contains(filters["competitor"], case=False, na=False)]

    # فلتر نطاق السعر
    if filters.get("price_min") is not None:
        result = result[result.get("السعر", pd.Series(dtype=float)) >= filters["price_min"]]
    if filters.get("price_max") is not None:
        result = result[result.get("السعر", pd.Series(dtype=float)) <= filters["price_max"]]

    # فلتر نسبة التطابق
    if filters.get("match_min") is not None:
        result = result[result.get("نسبة التطابق", pd.Series(dtype=float)) >= filters["match_min"]]

    # فلتر الفرق
    if filters.get("diff_min") is not None:
        result = result[result.get("الفرق", pd.Series(dtype=float)).abs() >= filters["diff_min"]]

    # فلتر النوع
    if filters.get("type") and filters["type"] != "الكل":
        result = result[result.get("النوع", pd.Series(dtype=str)).str.contains(filters["type"], case=False, na=False)]

    # فلتر الحجم
    if filters.get("size") and filters["size"] != "الكل":
        result = result[result.get("الحجم", pd.Series(dtype=str)).str.contains(filters["size"], case=False, na=False)]

    # فلتر القرار
    if filters.get("decision") and filters["decision"] != "الكل":
        result = result[result.get("القرار", pd.Series(dtype=str)).str.contains(filters["decision"], case=False, na=False)]

    # بحث نصي
    if filters.get("search"):
        search = filters["search"].lower()
        mask = result.apply(lambda row: any(search in str(v).lower() for v in row.values), axis=1)
        result = result[mask]

    return result


def get_filter_options(df):
    """استخراج خيارات الفلاتر من البيانات"""
    options = {"brands": ["الكل"], "competitors": ["الكل"], "types": ["الكل"], "sizes": ["الكل"], "decisions": ["الكل"]}

    if "الماركة" in df.columns:
        brands = df["الماركة"].dropna().unique().tolist()
        options["brands"].extend(sorted(set(b for b in brands if b)))

    if "المنافس" in df.columns:
        comps = df["المنافس"].dropna().unique().tolist()
        options["competitors"].extend(sorted(set(c for c in comps if c)))

    if "النوع" in df.columns:
        types = df["النوع"].dropna().unique().tolist()
        options["types"].extend(sorted(set(t for t in types if t)))

    if "الحجم" in df.columns:
        sizes = df["الحجم"].dropna().unique().tolist()
        options["sizes"].extend(sorted(set(s for s in sizes if s)))

    if "القرار" in df.columns:
        decisions = df["القرار"].dropna().unique().tolist()
        options["decisions"].extend(sorted(set(d for d in decisions if d)))

    return options


# ===== تصدير Excel =====
def export_to_excel(df, sheet_name="البيانات"):
    """تصدير DataFrame إلى Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output


def export_multiple_sheets(data_dict):
    """تصدير عدة أوراق في ملف Excel واحد"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in data_dict.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0)
    return output


# ===== عمل في الخلفية =====
class BackgroundTask:
    """إدارة المهام في الخلفية"""
    _tasks = {}

    @classmethod
    def start(cls, task_id, func, *args, **kwargs):
        """بدء مهمة في الخلفية"""
        cls._tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "result": None,
            "error": None,
            "started": datetime.now().strftime("%H:%M:%S")
        }

        def wrapper():
            try:
                result = func(*args, **kwargs)
                cls._tasks[task_id]["result"] = result
                cls._tasks[task_id]["status"] = "done"
                cls._tasks[task_id]["progress"] = 100
            except Exception as e:
                cls._tasks[task_id]["error"] = str(e)
                cls._tasks[task_id]["status"] = "error"

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        return task_id

    @classmethod
    def get_status(cls, task_id):
        return cls._tasks.get(task_id, {"status": "not_found"})

    @classmethod
    def update_progress(cls, task_id, progress):
        if task_id in cls._tasks:
            cls._tasks[task_id]["progress"] = progress

    @classmethod
    def get_result(cls, task_id):
        task = cls._tasks.get(task_id)
        if task and task["status"] == "done":
            return task["result"]
        return None


# ===== لصق ومعالجة =====
def parse_pasted_text(text):
    """تحليل نص ملصوق وتحويله إلى بيانات"""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return None, "لا يوجد محتوى"

    # محاولة تحليل كـ CSV/TSV
    if '\t' in lines[0] or ',' in lines[0]:
        sep = '\t' if '\t' in lines[0] else ','
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            return df, f"تم تحليل {len(df)} صف"
        except:
            pass

    # محاولة تحليل كقائمة
    products = []
    for line in lines:
        products.append({"المنتج": line})
    return pd.DataFrame(products), f"تم تحليل {len(products)} عنصر"


def process_ai_commands(text, products_df=None):
    """معالجة أوامر AI على البيانات"""
    commands = {
        "حذف": "remove",
        "إزالة": "remove",
        "ازالة": "remove",
        "نقل": "move",
        "تأجيل": "defer",
        "تاجيل": "defer",
        "موافقة": "approve",
        "رفض": "reject"
    }

    detected = []
    for keyword, action in commands.items():
        if keyword in text:
            detected.append(action)

    return detected if detected else ["analyze"]


# ===== أدوات مساعدة =====
def format_price(price):
    """تنسيق السعر"""
    try:
        return f"{float(price):,.2f}"
    except:
        return "0.00"


def format_diff(diff):
    """تنسيق الفرق مع لون"""
    try:
        d = float(diff)
        if d > 0:
            return f"🔴 +{d:,.2f}"
        elif d < 0:
            return f"🟢 {d:,.2f}"
        return "⚪ 0.00"
    except:
        return "0.00"


def get_color_for_diff(diff):
    """الحصول على لون بناءً على الفرق"""
    try:
        d = float(diff)
        if d > 10:
            return "#ff4444"
        elif d > 0:
            return "#ff8800"
        elif d < -10:
            return "#00cc00"
        elif d < 0:
            return "#44aa44"
        return "#888888"
    except:
        return "#888888"


def safe_float(val, default=0.0):
    """تحويل آمن إلى float"""
    try:
        return float(val)
    except:
        return default


def log_event(event_type, details=""):
    """تسجيل حدث مع الوقت"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"time": timestamp, "type": event_type, "details": details}
