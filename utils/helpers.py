"""
helpers.py - أدوات مساعدة وإدارة المهام الخلفية v17.2
- إدارة المهام (Threading) لمنع تجمد الواجهة.
- فلاتر متقدمة تدعم هيكلية البيانات الجديدة.
- دوال تصدير Excel محسنة.
"""
import pandas as pd
import io
import threading
import uuid
import time
from datetime import datetime

# ===== 1. إدارة المهام في الخلفية (Background Task Manager) =====

class TaskManager:
    """يدير العمليات الطويلة في خيوط منفصلة (Threads)"""
    _tasks = {}

    @classmethod
    def start_task(cls, func, *args, **kwargs):
        """بدء مهمة جديدة وإرجاع معرفها (ID)"""
        task_id = str(uuid.uuid4())
        cls._tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'result': None,
            'error': None,
            'start_time': datetime.now(),
            'message': 'جاري البدء...'
        }
        
        def task_wrapper():
            try:
                # دالة تحديث التقدم التي سيستخدمها المحرك
                def update_progress(p, msg=""):
                    cls._tasks[task_id]['progress'] = int(p * 100)
                    if msg: cls._tasks[task_id]['message'] = msg

                # استدعاء الدالة الأصلية (المحرك) مع تمرير دالة التقدم
                # نفترض أن الدالة المستقبلة تقبل معامل progress_callback
                result = func(*args, progress_callback=update_progress, **kwargs)
                
                cls._tasks[task_id]['result'] = result
                cls._tasks[task_id]['status'] = 'completed'
                cls._tasks[task_id]['progress'] = 100
                cls._tasks[task_id]['message'] = 'تم الانتهاء بنجاح'
            except Exception as e:
                cls._tasks[task_id]['error'] = str(e)
                cls._tasks[task_id]['status'] = 'failed'
                cls._tasks[task_id]['message'] = f"خطأ: {str(e)}"
        
        # تشغيل في Thread منفصل
        thread = threading.Thread(target=task_wrapper, daemon=True)
        thread.start()
        return task_id

    @classmethod
    def get_status(cls, task_id):
        """جلب حالة المهمة الحالية"""
        return cls._tasks.get(task_id, {'status': 'not_found'})

    @classmethod
    def clear_task(cls, task_id):
        """تنظيف الذاكرة بعد الانتهاء"""
        if task_id in cls._tasks:
            del cls._tasks[task_id]

# ===== 2. الفلاتر المتقدمة =====

def apply_filters(df, filters):
    """تطبيق فلاتر متعددة على DataFrame"""
    if df is None or df.empty: return df
    
    result = df.copy()
    
    # 1. فلتر البحث النصي (شامل)
    if filters.get("search"):
        search_term = filters["search"].lower()
        # دمج كل الأعمدة النصية للبحث فيها
        mask = result.astype(str).apply(
            lambda x: x.str.lower().str.contains(search_term, na=False)
        ).any(axis=1)
        result = result[mask]

    # 2. فلتر الماركة
    if filters.get("brand") and filters["brand"] != "الكل":
        result = result[result["الماركة"] == filters["brand"]]

    # 3. فلتر المنافس
    if filters.get("competitor") and filters["competitor"] != "الكل":
        result = result[result["المنافس"] == filters["competitor"]]

    # 4. فلتر فرق السعر (المدى)
    if filters.get("diff_min") is not None:
        result = result[result["الفرق"].abs() >= filters["diff_min"]]

    # 5. فلتر نسبة التطابق
    if filters.get("match_min") is not None:
        result = result[result["نسبة التطابق"] >= filters["match_min"]]
        
    # 6. فلتر السعر (Range)
    if filters.get("price_min") is not None:
        result = result[result["السعر"] >= filters["price_min"]]
    if filters.get("price_max") is not None and filters["price_max"] > 0:
        result = result[result["السعر"] <= filters["price_max"]]

    return result

def get_filter_options(df):
    """استخراج خيارات القوائم المنسدلة من البيانات"""
    options = {
        "brands": ["الكل"],
        "competitors": ["الكل"],
        "types": ["الكل"]
    }
    
    if df is None or df.empty: return options

    if "الماركة" in df.columns:
        brands = sorted(df["الماركة"].dropna().unique().astype(str).tolist())
        options["brands"].extend([b for b in brands if b])

    if "المنافس" in df.columns:
        comps = sorted(df["المنافس"].dropna().unique().astype(str).tolist())
        options["competitors"].extend([c for c in comps if c])
        
    if "النوع" in df.columns:
        types = sorted(df["النوع"].dropna().unique().astype(str).tolist())
        options["types"].extend([t for t in types if t])

    return options

# ===== 3. أدوات التنسيق والتصدير =====

def format_price(val):
    try: return f"{float(val):,.2f}"
    except: return "0.00"

def format_diff(val):
    try:
        v = float(val)
        if v > 0: return f"🔴 +{v:,.2f}" # أغلى من المنافس
        if v < 0: return f"🟢 {v:,.2f}"  # أرخص من المنافس
        return "⚪ 0.00"
    except: return "0.00"

def export_to_excel(df, sheet_name="Sheet1"):
    """تصدير سريع لملف Excel"""
    output = io.BytesIO()
    # إزالة الأعمدة التقنية قبل التصدير
    export_df = df.copy()
    cols_to_drop = ['norm_name', 'vector_id', 'جميع المنافسين']
    export_df = export_df.drop(columns=[c for c in cols_to_drop if c in export_df.columns], errors='ignore')
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name=sheet_name[:30], index=False) # Excel limit 31 chars
    output.seek(0)
    return output

def export_multiple_sheets(data_dict):
    """تصدير عدة شيتات في ملف واحد"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in data_dict.items():
            if df is not None and not df.empty:
                # تنظيف
                export_df = df.copy()
                cols_to_drop = ['norm_name', ' جميع المنافسين']
                export_df = export_df.drop(columns=[c for c in cols_to_drop if c in export_df.columns], errors='ignore')
                export_df.to_excel(writer, sheet_name=name[:30], index=False)
    output.seek(0)
    return output

# ===== 4. معالجة النصوص الملصوقة (Paste) =====
def parse_pasted_text(text):
    """تحويل النص المنسوخ من Excel/Sheets إلى DataFrame"""
    try:
        # محاولة قراءة كـ Tab-separated (Excel default copy)
        df = pd.read_csv(io.StringIO(text), sep='\t')
        if len(df.columns) < 2:
            # محاولة قراءة كـ CSV عادي
            df = pd.read_csv(io.StringIO(text), sep=',')
        return df, f"تم استيراد {len(df)} صف بنجاح"
    except Exception as e:
        return None, f"فشل تحليل النص: {str(e)}"
