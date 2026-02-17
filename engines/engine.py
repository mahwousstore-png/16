"""
engine.py - محرك v17.7 (Smart Validator)
- ميزة: طبقة تحقق ذكية لرفض التطابقات غير المنطقية (عينات، ماركات مختلفة).
- تحسين: فحص فروقات الأسعار الضخمة.
"""
import re
import pandas as pd
import io
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import (MATCH_THRESHOLD, HIGH_CONFIDENCE, PRICE_TOLERANCE, 
                    REJECT_KEYWORDS, KNOWN_BRANDS, WORD_REPLACEMENTS)

# ===== دوال المعالجة =====
def normalize(text):
    if not isinstance(text, str): return ""
    t = text.strip().lower()
    # إبقاء الأحرف والأرقام فقط
    t = re.sub(r'[^\w\s\u0600-\u06FF.]', ' ', t)
    for ar, en in WORD_REPLACEMENTS.items():
        if ar in t: t = t.replace(ar.lower(), en)
    return re.sub(r'\s+', ' ', t).strip()

def extract_size(text):
    if not isinstance(text, str): return 0
    # استخراج الحجم بدقة (ml, g, oz)
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:ml|lz|oz|مل|ملي|g|gm|gram)', text.lower())
    return float(m.group(1)) if m else 0

def extract_brand(text):
    if not isinstance(text, str): return ""
    tl = text.lower()
    # ترتيب الماركات بالأطول أولاً لتجنب تداخل الأسماء
    sorted_brands = sorted(KNOWN_BRANDS, key=len, reverse=True)
    for b in sorted_brands:
        if b.lower() in tl: return b.lower()
    return ""

def is_sample_or_tester(text):
    """كشف دقيق للعينات والتسترات"""
    if not isinstance(text, str): return False
    t = text.lower()
    sample_words = ['sample', 'vial', 'عينة', 'عيينة', 'تجربة', '2ml', '1ml', '1.5ml']
    return any(w in t for w in sample_words)

# ===== المحرك الذكي =====
class SmartMatcher:
    def __init__(self, our_df, comp_dfs, mapping=None):
        self.our_df = our_df.copy()
        self.comp_dfs = comp_dfs
        self.mapping = mapping
        # زيادة دقة التحليل النصي
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', 
            ngram_range=(3, 5), # زيادتها لتقليل تأثير الكلمات القصيرة الشائعة
            min_df=1
        )
        self.prepare_data()

    def prepare_data(self):
        # تحديد الأعمدة
        if self.mapping:
            self.our_prod_col = self.mapping.get('our_name')
            self.our_price_col = self.mapping.get('our_price')
        else:
            self.our_prod_col = self.our_df.columns[0]
            self.our_price_col = self.our_df.columns[1]

        # معالجة بياناتنا
        self.our_df[self.our_prod_col] = self.our_df[self.our_prod_col].astype(str)
        self.our_df['norm_name'] = self.our_df[self.our_prod_col].apply(normalize)
        self.our_df['brand'] = self.our_df[self.our_prod_col].apply(extract_brand)
        self.our_df['size'] = self.our_df[self.our_prod_col].apply(extract_size)
        self.our_df['is_sample'] = self.our_df[self.our_prod_col].apply(is_sample_or_tester)

        # معالجة المنافسين
        self.processed_comps = {}
        for name, df in self.comp_dfs.items():
            df_clean = df.copy()
            if self.mapping:
                target_p = self.mapping.get('comp_name')
                target_pr = self.mapping.get('comp_price')
                p_col = target_p if target_p in df_clean.columns else df_clean.columns[0]
                pr_col = target_pr if target_pr in df_clean.columns else df_clean.columns[1]
            else:
                p_col = df_clean.columns[0]
                pr_col = df_clean.columns[1]

            # تنظيف السعر
            if pr_col in df_clean.columns:
                df_clean[pr_col] = pd.to_numeric(
                    df_clean[pr_col].astype(str).str.replace(r'[^\d.]', '', regex=True), 
                    errors='coerce'
                ).fillna(0)
            
            df_clean[p_col] = df_clean[p_col].astype(str)
            df_clean['norm_name'] = df_clean[p_col].apply(normalize)
            df_clean['brand'] = df_clean[p_col].apply(extract_brand)
            df_clean['size'] = df_clean[p_col].apply(extract_size)
            df_clean['is_sample'] = df_clean[p_col].apply(is_sample_or_tester)
            
            self.processed_comps[name] = {'df': df_clean, 'p_col': p_col, 'pr_col': pr_col}

    def validate_match(self, row_our, row_comp, price_our, price_comp):
        """التحقق الذكي قبل قبول المطابقة"""
        # 1. اختلاف الماركة (قاتل)
        if row_our['brand'] and row_comp['brand']:
            if row_our['brand'] != row_comp['brand']:
                # تساهل بسيط إذا كانت إحداهما جزء من الأخرى
                if row_our['brand'] not in row_comp['brand'] and row_comp['brand'] not in row_our['brand']:
                    return False, "Brand Mismatch"

        # 2. فخ العينات (Sample Trap)
        # إذا منتجنا ليس عينة والمنافس عينة (أو العكس)
        if row_our['is_sample'] != row_comp['is_sample']:
            return False, "Sample vs Full Mismatch"

        # 3. كارثة السعر (Price Logic)
        # إذا كان الفرق هائل (أكثر من 3 أضعاف أو أقل من الثلث)
        if price_our > 0 and price_comp > 0:
            ratio = price_our / price_comp
            if ratio > 4.0: # منتجنا بـ 400 والمنافس بـ 100 (غالباً خطأ في الحجم أو النوع)
                return False, "Price too low (Likely Sample/Fake)"
            if ratio < 0.25: # منتجنا بـ 100 والمنافس بـ 400 (غالباً خطأ في المجموعة)
                return False, "Price too high (Likely Set/Bundle)"

        # 4. اختلاف الحجم (Size Mismatch)
        if row_our['size'] > 0 and row_comp['size'] > 0:
            if row_our['size'] != row_comp['size']:
                return False, "Size Mismatch"

        return True, "Valid"

    def strict_score(self, row_our, row_comp):
        # حساب التشابه النصي
        score = fuzz.token_sort_ratio(row_our['norm_name'], row_comp['norm_name'])
        
        # تحسينات العلامة
        if row_our['brand'] == row_comp['brand'] and row_our['brand'] != "":
            score += 10 # مكافأة تطابق الماركة
        
        return score

    def run(self, progress_callback=None):
        results = []
        expected_cols = ["المنتج", "السعر", "منتج المنافس", "سعر المنافس", "الفرق", 
                         "نسبة التطابق", "القرار", "المنافس", "جميع المنافسين"]

        if not self.processed_comps: return pd.DataFrame(columns=expected_cols)

        total = len(self.our_df)
        
        for comp_name, comp_data in self.processed_comps.items():
            comp_df = comp_data['df']
            if comp_df.empty: continue
            
            try:
                tfidf_comp = self.vectorizer.fit_transform(comp_df['norm_name'])
                tfidf_our = self.vectorizer.transform(self.our_df['norm_name'])
                cosine_sim = cosine_similarity(tfidf_our, tfidf_comp)
            except: continue

            for idx, row_our in self.our_df.iterrows():
                # تجاهل العينات من طرفنا إذا أردت (اختياري)
                # if row_our['is_sample']: continue
                
                scores = cosine_sim[idx]
                top_idx = scores.argsort()[-5:][::-1]
                
                best_match = None
                best_score = 0
                
                for c_idx in top_idx:
                    if scores[c_idx] < 0.2: continue # عتبة أولية
                    
                    row_comp = comp_df.iloc[c_idx]
                    p_our = float(row_our[self.our_price_col]) if self.our_price_col in row_our else 0
                    p_comp = float(row_comp[comp_data['pr_col']])

                    # --- مرحلة التحقق الذكي (Smart Validation) ---
                    is_valid, reason = self.validate_match(row_our, row_comp, p_our, p_comp)
                    if not is_valid:
                        continue # تخطي هذا المرشح الفاسد

                    # حساب العلامة الدقيقة
                    s = self.strict_score(row_our, row_comp)
                    
                    if s > best_score:
                        best_score = s
                        best_match = row_comp

                if best_match is not None and best_score >= MATCH_THRESHOLD:
                    p_comp = float(best_match[comp_data['pr_col']])
                    p_our = float(row_our[self.our_price_col]) if self.our_price_col in row_our else 0
                    diff = p_our - p_comp
                    
                    decision = "✅ موافق"
                    if diff > PRICE_TOLERANCE: decision = "🔴 سعر أعلى"
                    elif diff < -PRICE_TOLERANCE: decision = "🟢 سعر أقل"
                    elif best_score < HIGH_CONFIDENCE: decision = "⚠️ مراجعة"
                    
                    results.append({
                        "المنتج": row_our[self.our_prod_col],
                        "السعر": p_our,
                        "منتج المنافس": best_match[comp_data['p_col']],
                        "سعر المنافس": p_comp,
                        "الفرق": diff,
                        "نسبة التطابق": best_score,
                        "القرار": decision,
                        "المنافس": comp_name
                    })
                
                if progress_callback and idx % 50 == 0: progress_callback(idx/total)

        if not results: return pd.DataFrame(columns=expected_cols)
        
        # تجميع النتائج واختيار أفضل تطابق لكل منتج
        df = pd.DataFrame(results)
        final = []
        for n, g in df.groupby("المنتج"):
            # نفضل التطابق الأعلى، ثم السعر الأقل للمنافس
            best = g.sort_values(by=['نسبة التطابق', 'سعر المنافس'], ascending=[False, True]).iloc[0].to_dict()
            best['جميع المنافسين'] = g.to_dict('records')
            final.append(best)
            
        return pd.DataFrame(final)

# ===== واجهات القراءة والتصدير =====
def read_file(uploaded_file):
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    df = None
    encodings = ['utf-8', 'utf-8-sig', 'windows-1256', 'cp1256']
    
    if name.endswith('.csv'):
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                if not df.empty: break
            except: continue
    elif name.endswith(('.xlsx', '.xls')):
        try: df = pd.read_excel(uploaded_file)
        except: pass
        
    if df is not None:
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')
    return df, None

def run_full_analysis(our_df, comp_dfs, progress_callback=None, mapping=None):
    matcher = SmartMatcher(our_df, comp_dfs, mapping)
    return matcher.run(progress_callback)

def find_missing_products(our_df, comp_dfs):
    return pd.DataFrame()

def export_excel(df, sheet_name="Sheet1"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as w:
        df.to_excel(w, sheet_name=sheet_name[:30], index=False)
    return output.getvalue()

def export_section_excel(df, name): return export_excel(df, name)
