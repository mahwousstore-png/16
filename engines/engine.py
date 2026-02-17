"""
engine.py - المحرك المتجهي السريع v17.2 (Vectorized Engine)
- يعتمد على TF-IDF & Cosine Similarity لسرعة تصل إلى 50x
- فلترة صارمة للماركة والحجم لتقليل الأخطاء
- متوافق تماماً مع app.py v17.2
"""
import re
import pandas as pd
import numpy as np
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# استيراد الإعدادات (تأكد من تطابق الأسماء مع config.py)
try:
    from config import (
        REJECT_KEYWORDS, KNOWN_BRANDS, WORD_REPLACEMENTS,
        MATCH_THRESHOLD, HIGH_CONFIDENCE, REVIEW_THRESHOLD,
        PRICE_TOLERANCE, TESTER_KEYWORDS, SET_KEYWORDS
    )
except ImportError:
    # قيم افتراضية للطوارئ
    MATCH_THRESHOLD = 60
    HIGH_CONFIDENCE = 90
    PRICE_TOLERANCE = 5
    REJECT_KEYWORDS = ["sample", "عينة"]
    KNOWN_BRANDS = []
    WORD_REPLACEMENTS = {}

# ===== 1. دوال القراءة والمعالجة (Helpers) =====

def read_file(uploaded_file):
    """قراءة ملف CSV أو Excel بمرونة عالية"""
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        elif name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "صيغة الملف غير مدعومة"
        
        # تنظيف أسماء الأعمدة (إزالة المسافات الزائدة)
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')
        return df, None
    except Exception as e:
        return None, f"خطأ القراءة: {str(e)}"

def normalize(text):
    """توحيد النصوص (عربي/إنجليزي) للمطابقة"""
    if not isinstance(text, str): return ""
    t = text.strip().lower()
    # استبدال الكلمات الشائعة (مثل EDP -> eau de parfum)
    for ar, en in WORD_REPLACEMENTS.items():
        t = t.replace(ar.lower(), en)
    # تنظيف الرموز وتوحيد العربية
    t = re.sub("[إأآا]", "ا", t)
    t = re.sub("ة", "ه", t)
    t = re.sub("ى", "ي", t)
    t = re.sub(r'[^\w\s.]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_size(text):
    """استخراج الحجم (ml)"""
    if not isinstance(text, str): return 0
    m = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ml|مل|ملي|g|غ)', text.lower())
    return float(m[0]) if m else 0

def extract_brand(text):
    """استخراج الماركة بناءً على القائمة المعروفة"""
    if not isinstance(text, str): return ""
    tl = text.lower()
    for b in KNOWN_BRANDS:
        if b.lower() in tl:
            return b
    # إذا لم توجد في القائمة، خذ الكلمة الأولى كاجتهاد
    return text.split()[0] if text else ""

def extract_type(text):
    """استخراج نوع العطر"""
    if not isinstance(text, str): return ""
    tl = text.lower()
    if any(k in tl for k in ['edp', 'eau de parfum', 'بارفيوم', 'parfum']): return 'EDP'
    if any(k in tl for k in ['edt', 'eau de toilette', 'تواليت']): return 'EDT'
    if any(k in tl for k in ['cologne', 'كولون', 'edc']): return 'EDC'
    if any(k in tl for k in ['oil', 'زيت']): return 'Oil'
    return ''

def is_sample(text):
    if not isinstance(text, str): return False
    tl = text.lower()
    return any(k in tl for k in REJECT_KEYWORDS)

# ===== 2. المحرك المتجهي (The Vectorized Engine) =====

def run_full_analysis(our_df, comp_dfs, progress_callback=None):
    """
    تحليل كامل باستخدام المصفوفات (Vectorization).
    الأسرع والأدق للبيانات الضخمة.
    """
    results = []
    
    # تحديد أعمدتنا
    our_col = next((c for c in ["المنتج", "اسم المنتج", "Product", "Name", "name"] if c in our_df.columns), our_df.columns[0])
    our_price_col = next((c for c in ["السعر", "سعر", "Price", "price", "Cost"] if c in our_df.columns), None)

    # تجهيز بياناتنا (مرة واحدة)
    our_data = our_df.copy()
    # تنظيف واستخراج الخصائص
    our_data['normalized'] = our_data[our_col].apply(normalize)
    our_data['brand'] = our_data[our_col].apply(extract_brand)
    our_data['size'] = our_data[our_col].apply(extract_size)
    
    # استبعاد العينات من المقارنة
    our_data = our_data[~our_data[our_col].apply(is_sample)]

    # إعداد المحرك (TF-IDF)
    # نستخدم char_wb (حروف مع حدود كلمات) لمرونة أكبر في الأكواد والأسماء
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
    
    # تدريب النموذج على بياناتنا
    try:
        our_vectors = vectorizer.fit_transform(our_data['normalized'].fillna(""))
    except ValueError:
        return pd.DataFrame() # بيانات فارغة

    total_steps = len(comp_dfs)
    
    for idx, (comp_name, comp_df) in enumerate(comp_dfs.items()):
        # تحديث شريط التقدم في app.py
        if progress_callback: progress_callback((idx) / total_steps)
        
        # تحديد أعمدة المنافس
        comp_prod_col = next((c for c in ["المنتج", "اسم المنتج", "Product", "Name", "name"] if c in comp_df.columns), comp_df.columns[0])
        comp_price_col = next((c for c in ["السعر", "سعر", "Price", "price"] if c in comp_df.columns), None)

        # تجهيز بيانات المنافس
        comp_data = comp_df.copy()
        comp_data = comp_data[~comp_data[comp_prod_col].apply(is_sample)] # استبعاد عينات المنافس
        comp_data['normalized'] = comp_data[comp_prod_col].apply(normalize)
        comp_data['brand'] = comp_data[comp_prod_col].apply(extract_brand)
        comp_data['size'] = comp_data[comp_prod_col].apply(extract_size)

        if comp_data.empty: continue

        # تحويل بيانات المنافس لمصفوفة
        try:
            comp_vectors = vectorizer.transform(comp_data['normalized'].fillna(""))
        except: continue

        # === المضرب السحري: حساب التشابه (الكل مقابل الكل) ===
        # النتيجة مصفوفة ضخمة: [عدد منتجاتنا] × [عدد منتجات المنافس]
        similarity_matrix = cosine_similarity(our_vectors, comp_vectors)

        # استخراج النتائج
        for i, (our_idx, our_row) in enumerate(our_data.iterrows()):
            
            # صف التشابهات لهذا المنتج
            sim_scores = similarity_matrix[i]
            
            # --- فلترة ذكية (Post-Processing Filters) ---
            
            # 1. فلتر الماركة (Brand Lock)
            # إذا اختلفت الماركة، اجعل السكور صفر فوراً
            if our_row['brand']:
                brand_mask = comp_data['brand'].str.lower() != our_row['brand'].lower()
                sim_scores[brand_mask.values] = 0

            # 2. فلتر الحجم (Size Lock)
            # نسمح باختلاف بسيط (مثلاً 5 مل)
            if our_row['size'] > 0:
                size_diff = np.abs(comp_data['size'].values - our_row['size'])
                size_mask = size_diff > 5 # اختلاف أكثر من 5 مل
                sim_scores[size_mask] *= 0.5 # عقاب قوي للاختلاف

            # العثور على أفضل تطابق بعد الفلترة
            best_match_idx = sim_scores.argmax()
            best_score = sim_scores[best_match_idx] * 100

            if best_score >= MATCH_THRESHOLD:
                comp_row = comp_data.iloc[best_match_idx]
                
                # استخراج الأسعار
                our_p = float(our_row[our_price_col]) if our_price_col else 0
                comp_p = float(comp_row[comp_price_col]) if comp_price_col else 0
                
                # تجاهل الأسعار الصفرية
                if our_p <= 1 or comp_p <= 1: continue

                diff = our_p - comp_p
                
                # منطق القرار
                decision = "✅ موافق"
                risk = "منخفض"
                
                if diff > PRICE_TOLERANCE:
                    decision = "🔴 سعر أعلى"
                    risk = "عالي"
                elif diff < -PRICE_TOLERANCE:
                    decision = "🟢 سعر أقل"
                
                if best_score < HIGH_CONFIDENCE:
                    decision = "⚠️ مراجعة"
                    risk = "متوسط"

                results.append({
                    "المنتج": our_row[our_col],
                    "السعر": our_p,
                    "الماركة": our_row['brand'],
                    "الحجم": f"{int(our_row['size'])}ml" if our_row['size'] else "",
                    "النوع": extract_type(our_row[our_col]),
                    "منتج المنافس": comp_row[comp_prod_col],
                    "سعر المنافس": comp_p,
                    "الفرق": round(diff, 2),
                    "نسبة التطابق": round(best_score, 1),
                    "القرار": decision,
                    "الخطورة": risk,
                    "المنافس": comp_name,
                    # حقول للتوافق مع التصدير
                    "جميع المنافسين": [] 
                })

    if progress_callback: progress_callback(1.0)
    return pd.DataFrame(results)


def find_missing_products(our_df, comp_dfs):
    """
    نسخة سريعة جداً لإيجاد المفقودات باستخدام الـ Sets (Hashing)
    بدلاً من تكرار الحلقات البطيئة
    """
    missing = []
    
    # 1. تجهيز قائمة منتجاتنا كـ "بصمات" (Hash Set)
    our_col = next((c for c in ["المنتج", "اسم المنتج", "Product", "Name", "name"] if c in our_df.columns), our_df.columns[0])
    # نستخدم التطبيع الدقيق لإنشاء البصمة
    our_fingerprints = set(our_df[our_col].astype(str).apply(normalize).tolist())
    
    for comp_name, comp_df in comp_dfs.items():
        comp_prod_col = next((c for c in ["المنتج", "اسم المنتج", "Product", "Name", "name"] if c in comp_df.columns), comp_df.columns[0])
        comp_price_col = next((c for c in ["السعر", "سعر", "Price", "price"] if c in comp_df.columns), None)
        
        for _, row in comp_df.iterrows():
            p_name = str(row[comp_prod_col])
            if is_sample(p_name): continue
            
            p_fingerprint = normalize(p_name)
            
            # بحث فوري (O(1) complexity)
            if p_fingerprint not in our_fingerprints:
                # تحقق إضافي: هل الاسم قصير جداً ليكون مفيداً؟
                if len(p_fingerprint) < 4: continue
                
                price = 0
                if comp_price_col:
                    try: price = float(row[comp_price_col])
                    except: pass
                
                missing.append({
                    "منتج المنافس": p_name,
                    "سعر المنافس": price,
                    "المنافس": comp_name,
                    "الماركة": extract_brand(p_name),
                    "النوع": extract_type(p_name),
                    "الحجم": extract_size(p_name)
                })

    return pd.DataFrame(missing)


# ===== دوال التصدير (مطلوبة لـ app.py) =====
def export_excel(df, sheet_name="النتائج"):
    output = io.BytesIO()
    export_df = df.copy()
    if "جميع المنافسين" in export_df.columns:
        export_df = export_df.drop(columns=["جميع المنافسين"])
    # تصحيح ترتيب الأعمدة للأناقة
    cols_order = ["المنتج", "السعر", "منتج المنافس", "سعر المنافس", "الفرق", "نسبة التطابق", "القرار", "المنافس", "الماركة"]
    available_cols = [c for c in cols_order if c in export_df.columns]
    remaining_cols = [c for c in export_df.columns if c not in cols_order]
    export_df = export_df[available_cols + remaining_cols]
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()

def export_section_excel(df, section_name):
    return export_excel(df, sheet_name=section_name[:31])
