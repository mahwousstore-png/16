"""
engine.py - محرك المطابقة الهجين v17.2 (Hybrid Vectorization)
- الترقية: TF-IDF Vectorization للبحث الأولي (فائق السرعة).
- الدقة: Reranking منطقي يعتمد على الماركة والحجم والنوع.
- الأداء: معالجة مسبقة للبيانات (Vectorized Preprocessing).
"""
import re
import pandas as pd
import numpy as np
import io
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import (REJECT_KEYWORDS, KNOWN_BRANDS, WORD_REPLACEMENTS,
                    MATCH_THRESHOLD, HIGH_CONFIDENCE, REVIEW_THRESHOLD,
                    PRICE_TOLERANCE, TESTER_KEYWORDS, SET_KEYWORDS)

# ===== 1. دوال مساعدة وتحليل (محسنة) =====

def normalize(text):
    if not isinstance(text, str): return ""
    t = text.strip().lower()
    # استبدال سريع باستخدام قاموس
    for ar, en in WORD_REPLACEMENTS.items():
        if ar in t: # تحقق سريع قبل الاستبدال
            t = t.replace(ar.lower(), en)
    # تنظيف الرموز
    t = re.sub(r'[^\w\s.]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def extract_size(text):
    if not isinstance(text, str): return 0
    # تحسين Regex ليكون أدق وأسرع
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:ml|lz|مل|ملي|g|gram)', text.lower())
    return float(m.group(1)) if m else 0

def extract_brand(text):
    if not isinstance(text, str): return ""
    tl = text.lower()
    # البحث عن الماركة (الأطول أولاً لتجنب تداخل الأسماء)
    for b in KNOWN_BRANDS:
        if b.lower() in tl:
            return b
    return ""

def extract_type(text):
    if not isinstance(text, str): return ""
    tl = text.lower()
    if 'edp' in tl or 'parfum' in tl or 'بارفان' in tl: return 'edp'
    if 'edt' in tl or 'toilette' in tl or 'تواليت' in tl: return 'edt'
    if 'edc' in tl or 'cologne' in tl or 'كولون' in tl: return 'edc'
    if 'oil' in tl or 'زيت' in tl: return 'oil'
    if 'tester' in tl or 'تستر' in tl: return 'tester'
    return ''

def is_sample(text):
    if not isinstance(text, str): return False
    tl = text.lower()
    return any(k in tl for k in REJECT_KEYWORDS)

# ===== 2. قراءة الملفات =====
def read_file(uploaded_file):
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
            return None, "صيغة الملف غير مدعومة."
        
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')
        
        # تنظيف أولي لأسماء الأعمدة
        df = df.rename(columns=lambda x: x.lower().replace(' ', '_'))
        return df, None
    except Exception as e:
        return None, f"خطأ في قراءة الملف: {str(e)}"

# ===== 3. فئة المطابقة الذكية (Smart Matcher Class) =====

class SmartMatcher:
    def __init__(self, our_df, comp_dfs):
        self.our_df = our_df.copy()
        self.comp_dfs = comp_dfs
        # إعداد Vectorizer للغة العربية والإنجليزية
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', # تحليل الأحرف (يساعد في الأخطاء الإملائية)
            ngram_range=(2, 4), # يأخذ مقاطع من حرفين إلى 4 أحرف
            min_df=1
        )
        self.prepare_data()

    def get_col_name(self, df, candidates):
        for c in candidates:
            for col in df.columns:
                if c.lower() in col.lower():
                    return col
        return df.columns[0]

    def prepare_data(self):
        # 1. تحديد أعمدة منتجاتنا
        self.our_prod_col = self.get_col_name(self.our_df, ["product", "name", "المنتج", "اسم"])
        self.our_price_col = self.get_col_name(self.our_df, ["price", "سعر", "السعر"])
        
        # معالجة مسبقة لبياناتنا (Vectorized)
        self.our_df['norm_name'] = self.our_df[self.our_prod_col].apply(normalize)
        self.our_df['brand'] = self.our_df[self.our_prod_col].apply(extract_brand)
        self.our_df['size'] = self.our_df[self.our_prod_col].apply(extract_size)
        self.our_df['type'] = self.our_df[self.our_prod_col].apply(extract_type)

        # 2. تجهيز بيانات المنافسين
        self.processed_comps = {}
        for name, df in self.comp_dfs.items():
            df_clean = df.copy()
            p_col = self.get_col_name(df_clean, ["product", "name", "المنتج", "اسم"])
            pr_col = self.get_col_name(df_clean, ["price", "سعر", "السعر"])
            
            # استبعاد العينات مبكراً
            df_clean = df_clean[~df_clean[p_col].apply(is_sample)]
            
            df_clean['norm_name'] = df_clean[p_col].apply(normalize)
            df_clean['brand'] = df_clean[p_col].apply(extract_brand)
            df_clean['size'] = df_clean[p_col].apply(extract_size)
            df_clean['type'] = df_clean[p_col].apply(extract_type)
            
            # تحويل السعر لأرقام
            df_clean[pr_col] = pd.to_numeric(df_clean[pr_col], errors='coerce').fillna(0)
            
            self.processed_comps[name] = {
                'df': df_clean,
                'p_col': p_col,
                'pr_col': pr_col
            }

    def strict_score(self, row_our, row_comp):
        """حساب دقيق جداً بعد الترشيح الأولي"""
        # 1. العلامة الأساسية من الاسم (RapidFuzz)
        base_score = fuzz.token_sort_ratio(row_our['norm_name'], row_comp['norm_name'])
        
        # 2. مكافأة/عقوبة الماركة
        if row_our['brand'] and row_comp['brand']:
            if row_our['brand'].lower() == row_comp['brand'].lower():
                base_score += 5
            else:
                return 0 # ماركة مختلفة = رفض فوري

        # 3. عقوبة الحجم (صارمة)
        if row_our['size'] > 0 and row_comp['size'] > 0:
            if row_our['size'] != row_comp['size']:
                return 0 # حجم مختلف = رفض فوري
            else:
                base_score += 5
        
        # 4. عقوبة النوع (EDP vs EDT)
        if row_our['type'] and row_comp['type']:
            if row_our['type'] != row_comp['type']:
                base_score -= 15

        return min(100, max(0, base_score))

    def run(self, progress_callback=None):
        results = []
        total_items = len(self.our_df)
        
        # حلقة على كل منافس
        for comp_name, comp_data in self.processed_comps.items():
            comp_df = comp_data['df']
            if comp_df.empty: continue
            
            # بناء TF-IDF Matrix للمنافس
            comp_names = comp_df['norm_name'].tolist()
            try:
                tfidf_matrix_comp = self.vectorizer.fit_transform(comp_names)
            except ValueError: continue

            # تجهيز منتجاتنا
            our_names = self.our_df['norm_name'].tolist()
            tfidf_matrix_our = self.vectorizer.transform(our_names)

            # حساب التشابه (Cosine Similarity) - عملية مصفوفات سريعة
            cosine_sim = cosine_similarity(tfidf_matrix_our, tfidf_matrix_comp)

            # معالجة النتائج
            for idx, row_our in self.our_df.iterrows():
                if is_sample(row_our[self.our_prod_col]): continue
                
                # أفضل 5 مرشحين بناءً على TF-IDF
                sim_scores = cosine_sim[idx]
                top_indices = sim_scores.argsort()[-5:][::-1] 
                
                best_match = None
                best_score = 0

                for comp_idx in top_indices:
                    if sim_scores[comp_idx] < 0.3: continue # تجاوز الضعيف جداً
                    
                    row_comp = comp_df.iloc[comp_idx]
                    
                    # التطبيق الصارم
                    score = self.strict_score(row_our, row_comp)
                    
                    if score > best_score and score >= MATCH_THRESHOLD:
                        best_score = score
                        best_match = row_comp

                if best_match is not None:
                    our_price = float(pd.to_numeric(row_our[self.our_price_col], errors='coerce') or 0)
                    comp_price = float(best_match[comp_data['pr_col']])
                    diff = our_price - comp_price if comp_price > 0 else 0
                    
                    decision = "✅ موافق"
                    risk = "منخفض"
                    
                    if diff > PRICE_TOLERANCE:
                        decision = "🔴 سعر أعلى"
                        risk = "عالي" if diff > 20 else "متوسط"
                    elif diff < -PRICE_TOLERANCE:
                        decision = "🟢 سعر أقل"
                    elif best_score < REVIEW_THRESHOLD:
                        decision = "⚠️ مراجعة"
                        risk = "متوسط"

                    results.append({
                        "المنتج": row_our[self.our_prod_col],
                        "السعر": our_price,
                        "الماركة": row_our['brand'],
                        "الحجم": row_our['size'],
                        "النوع": row_our['type'],
                        "منتج المنافس": best_match[comp_data['p_col']],
                        "سعر المنافس": comp_price,
                        "الفرق": round(diff, 2),
                        "نسبة التطابق": round(best_score, 1),
                        "القرار": decision,
                        "الخطورة": risk,
                        "المنافس": comp_name
                    })

                if progress_callback and idx % 50 == 0: # تحديث كل 50 عنصر لتقليل الضغط
                    progress_callback((idx + 1) / total_items)

        # تجميع النتائج
        df_res = pd.DataFrame(results)
        if df_res.empty: return pd.DataFrame()

        final_rows = []
        grouped = df_res.groupby('المنتج')
        
        for name, group in grouped:
            valid_comps = group[group['سعر المنافس'] > 0]
            if not valid_comps.empty:
                best_comp_row = valid_comps.loc[valid_comps['سعر المنافس'].idxmin()].to_dict()
                best_comp_row['جميع المنافسين'] = group[['المنافس', 'منتج المنافس', 'سعر المنافس', 'نسبة التطابق']].rename(
                    columns={'منتج المنافس': 'name', 'سعر المنافس': 'price', 'نسبة التطابق': 'score', 'المنافس': 'competitor'}
                ).to_dict('records')
                final_rows.append(best_comp_row)
            else:
                row = group.iloc[0].to_dict()
                row['جميع المنافسين'] = []
                final_rows.append(row)

        return pd.DataFrame(final_rows)

# ===== 4. دالة الواجهة الرئيسية =====
def run_full_analysis(our_df, comp_dfs, progress_callback=None):
    matcher = SmartMatcher(our_df, comp_dfs)
    return matcher.run(progress_callback)

# ===== 5. إيجاد المفقودات =====
def find_missing_products(our_df, comp_dfs):
    missing = []
    # تحسين السرعة للمفقودات باستخدام Vectorization
    our_clean = our_df.iloc[:, 0].astype(str).apply(normalize).tolist()
    if not our_clean: return pd.DataFrame()

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    try:
        tfidf_our = vectorizer.fit_transform(our_clean)
    except: return pd.DataFrame()

    for comp_name, comp_df in comp_dfs.items():
        c_col = comp_df.columns[0]
        for col in comp_df.columns:
            if "منتج" in col or "name" in col.lower(): c_col = col; break
        p_col = comp_df.columns[1]
        for col in comp_df.columns:
            if "سعر" in col or "price" in col.lower(): p_col = col; break

        comp_names = comp_df[c_col].astype(str).apply(normalize).tolist()
        if not comp_names: continue

        tfidf_comp = vectorizer.transform(comp_names)
        cosine_sim = cosine_similarity(tfidf_comp, tfidf_our)
        max_scores = cosine_sim.max(axis=1)

        for idx, score in enumerate(max_scores):
            if score < 0.65:
                row = comp_df.iloc[idx]
                if is_sample(str(row[c_col])): continue
                missing.append({
                    "منتج المنافس": row[c_col],
                    "سعر المنافس": pd.to_numeric(row[p_col], errors='coerce'),
                    "المنافس": comp_name,
                    "الماركة": extract_brand(str(row[c_col])),
                    "الحجم": extract_size(str(row[c_col])),
                    "النوع": extract_type(str(row[c_col]))
                })
    
    return pd.DataFrame(missing).drop_duplicates(subset=['منتج المنافس'])

# ===== 6. التصدير =====
def export_excel(df, sheet_name="النتائج"):
    output = io.BytesIO()
    export_df = df.copy()
    if "جميع المنافسين" in export_df.columns:
        export_df = export_df.drop(columns=["جميع المنافسين"])
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()

def export_section_excel(df, section_name):
    return export_excel(df, sheet_name=section_name[:31])
