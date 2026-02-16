"""
helpers.py - أدوات مساعدة v16.0
يشمل: فلاتر، تصفح، تصدير، جداول مقارنة، عمل في الخلفية
"""
import streamlit as st
import pandas as pd
import threading, time
from io import BytesIO


# ===== العمل في الخلفية =====
class BackgroundTask:
    """تشغيل مهام في الخلفية بدون تجميد الواجهة"""
    def __init__(self):
        self._tasks = {}

    def run(self, task_id, func, *args, **kwargs):
        def wrapper():
            try:
                result = func(*args, **kwargs)
                self._tasks[task_id] = {"status": "done", "result": result}
            except Exception as e:
                self._tasks[task_id] = {"status": "error", "error": str(e)}

        self._tasks[task_id] = {"status": "running"}
        t = threading.Thread(target=wrapper, daemon=True)
        t.start()
        return task_id

    def get_status(self, task_id):
        return self._tasks.get(task_id, {"status": "not_found"})

    def is_running(self, task_id):
        return self._tasks.get(task_id, {}).get("status") == "running"

    def get_result(self, task_id):
        task = self._tasks.get(task_id, {})
        if task.get("status") == "done":
            return task.get("result")
        return None

bg_tasks = BackgroundTask()


# ===== فلاتر =====
def render_filters(df, prefix):
    filters = {}
    cols = st.columns(4)
    with cols[0]:
        search = st.text_input("🔍 بحث", key=f"{prefix}_search", placeholder="ابحث بالاسم...")
        if search: filters["search"] = search
    with cols[1]:
        if "المنافس" in df.columns:
            opts = ["الكل"] + sorted(df["المنافس"].dropna().unique().tolist())
            v = st.selectbox("المنافس", opts, key=f"{prefix}_comp")
            if v != "الكل": filters["المنافس"] = v
    with cols[2]:
        if "الخطورة" in df.columns:
            opts = ["الكل"] + sorted(df["الخطورة"].dropna().unique().tolist())
            v = st.selectbox("الخطورة", opts, key=f"{prefix}_risk")
            if v != "الكل": filters["الخطورة"] = v
    with cols[3]:
        if "الماركة" in df.columns:
            brands = df["الماركة"].dropna()
            brands = brands[brands != ""]
            if len(brands) > 0:
                opts = ["الكل"] + sorted(brands.unique().tolist())
                v = st.selectbox("الماركة", opts, key=f"{prefix}_brand")
                if v != "الكل": filters["الماركة"] = v
    return filters


def apply_filters(df, filters):
    if not filters: return df
    r = df.copy()
    if "search" in filters and "المنتج" in r.columns:
        r = r[r["المنتج"].str.lower().str.contains(filters["search"].lower(), na=False)]
    for col in ["المنافس", "الخطورة", "الماركة"]:
        if col in filters and col in r.columns:
            r = r[r[col] == filters[col]]
    return r


# ===== تصفح الصفحات =====
def paginate_df(df, per_page, key):
    if df.empty: return df
    total_pages = max(1, (len(df) - 1) // per_page + 1)
    page = st.number_input("الصفحة", 1, total_pages, 1, key=key)
    start = (page - 1) * per_page
    end = start + per_page
    st.caption(f"صفحة {page} من {total_pages} | عرض {min(per_page, len(df)-start)} من {len(df)}")
    return df.iloc[start:end]


# ===== تصدير Excel =====
def export_to_excel(df, filename="export.xlsx"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name="البيانات")
    return output.getvalue()


# ===== جدول المقارنة البصري =====
def render_comparison_table(df, section_type="raise"):
    if df.empty:
        st.info("لا توجد بيانات")
        return

    color_map = {"raise": "#dc3545", "lower": "#ffc107", "approved": "#28a745", "review": "#ff9800"}
    sec_color = color_map.get(section_type, "#6C63FF")

    html = '<table class="cmp-table"><thead><tr>'
    html += '<th style="width:30px">#</th>'
    html += '<th>منتجنا 🟣</th><th>سعرنا</th>'
    html += '<th>منتج المنافس 🟠</th><th>سعر المنافس</th>'
    html += '<th>الفرق</th><th>التطابق</th><th>الخطورة</th><th>المنافس</th>'
    html += '</tr></thead><tbody>'

    for i, (_, row) in enumerate(df.iterrows(), 1):
        our_name = row.get("المنتج", "")
        our_price = row.get("السعر", 0)
        comp_name = row.get("اسم المنافس", "")
        comp_price = row.get("أقل سعر منافس", 0)
        diff = row.get("الفرق", 0)
        score = row.get("نسبة التطابق", 0)
        risk = row.get("الخطورة", "")
        source = row.get("المنافس", "")

        # لون الفرق
        if diff > 0: dc = "#FF1744"
        elif diff < 0: dc = "#00C853"
        else: dc = "#FFD600"

        # لون الخطورة
        if risk == "حرج": rc, rb = "#FF1744", "b-high"
        elif risk == "متوسط": rc, rb = "#FFD600", "b-med"
        else: rc, rb = "#00C853", "b-low"

        # لون التطابق
        if score >= 95: sc = "#00C853"
        elif score >= 85: sc = "#FFD600"
        else: sc = "#FF9800"

        html += f'<tr>'
        html += f'<td style="color:{sec_color};font-weight:700">{i}</td>'
        html += f'<td class="td-our">{our_name}</td>'
        html += f'<td style="font-weight:700;color:#6C63FF">{our_price:.0f}</td>'
        html += f'<td class="td-comp">{comp_name}</td>'
        html += f'<td style="font-weight:700;color:#ff9800">{comp_price:.0f}</td>'
        html += f'<td style="font-weight:900;color:{dc}">{diff:+.0f}</td>'
        html += f'<td><div class="conf-bar"><div class="conf-fill" style="width:{score}%;background:{sc}"></div></div><span style="font-size:.75rem;color:{sc}">{score:.0f}%</span></td>'
        html += f'<td><span class="badge {rb}">{risk}</span></td>'
        html += f'<td style="font-size:.8rem;color:#8B8B8B">{source}</td>'
        html += '</tr>'

    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)
