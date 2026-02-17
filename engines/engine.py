"""
engine.py - محرك v17.6 (يدعم تحديد الأعمدة يدوياً)
"""
import re
import pandas as pd
import io
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import (MATCH_THRESHOLD, HIGH_CONFIDENCE, PRICE_TOLERANCE, 
                    REJECT_KEYWORDS, KNOWN_BRANDS, WORD_REPLACEMENTS)

# ... (نفس دوال normalize, extract_brand, extract_size السابقة) ...
def normalize(text):
    if not isinstance(text, str): return ""
    t = text.strip().lower()
    t = re.sub(r'[^\w\s\u0600-\u06FF.]', ' ', t)
    for ar, en in WORD_REPLACEMENTS.items():
        if ar in t: t = t.replace(ar.lower(), en)
    return re.sub(r'\s+', ' ', t).strip()

def extract_size(text):
    if not isinstance(text, str): return 0
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:ml|lz|oz|مل|ملي|g)', text.lower())
    return float(m.group(1)) if m else 0

def extract_brand(text):
    if not isinstance(text, str): return ""
    tl = text.lower()
    for b in KNOWN_BRANDS:
        if b.lower() in tl: return b
    return ""

def extract_type(text):
    if not isinstance(text, str): return ""
    tl = text.lower()
    if 'edp' in tl: return 'edp'
    if 'edt' in tl: return 'edt'
    if 'tester' in tl: return 'tester'
    return ''

def is_sample(text):
    if not isinstance(text, str): return False
    return any(k in text.lower() for k in REJECT_KEYWORDS)

def read_file(uploaded_file):
    encodings = ['utf-8', 'utf-8-sig', 'windows-1256', 'cp1256']
    name = uploaded_file.name.lower()
    df = None
    
    # إعادة المؤشر للبداية دائماً
    uploaded_file.seek(0)
    
    if name.endswith('.csv'):
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                if len(df.columns) > 0: break
            except: continue
    elif name.endswith(('.xlsx', '.xls')):
        try: df = pd.read_excel(uploaded_file)
        except: pass
        
    if df is not None:
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')
    return df, None

class SmartMatcher:
    def __init__(self, our_df, comp_dfs, mapping=None):
        self.our_df = our_df.copy()
        self.comp_dfs = comp_dfs
        self.mapping = mapping # استلام خريطة الأعمدة
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.prepare_data()

    def prepare_data(self):
        # استخدام الأعمدة المحددة يدوياً إذا وجدت، وإلا التخمين
        if self.mapping:
            self.our_prod_col = self.mapping.get('our_name')
            self.our_price_col = self.mapping.get('our_price')
        else:
            self.our_prod_col = self.our_df.columns[0]
            self.our_price_col = self.our_df.columns[1]

        # تجهيز بياناتنا
        self.our_df[self.our_prod_col] = self.our_df[self.our_prod_col].astype(str)
        self.our_df['norm_name'] = self.our_df[self.our_prod_col].apply(normalize)
        self.our_df['brand'] = self.our_df[self.our_prod_col].apply(extract_brand)
        self.our_df['size'] = self.our_df[self.our_prod_col].apply(extract_size)

        # تجهيز المنافسين
        self.processed_comps = {}
        for name, df in self.comp_dfs.items():
            df_clean = df.copy()
            
            if self.mapping:
                # محاولة استخدام نفس الأسماء للمنافس (أو البحث عنها إذا اختلفت قليلاً)
                target_p = self.mapping.get('comp_name')
                target_pr = self.mapping.get('comp_price')
                
                # التحقق من وجود العمود، وإذا لم يوجد نستخدم الاسم كما هو (نفترض توحيد الملفات)
                p_col = target_p if target_p in df_clean.columns else df_clean.columns[0]
                pr_col = target_pr if target_pr in df_clean.columns else df_clean.columns[1]
            else:
                p_col = df_clean.columns[0]
                pr_col = df_clean.columns[1]

            # تنظيف السعر
            if pr_col in df_clean.columns:
                df_clean[pr_col] = df_clean[pr_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df_clean[pr_col] = pd.to_numeric(df_clean[pr_col], errors='coerce').fillna(0)
            
            df_clean[p_col] = df_clean[p_col].astype(str)
            df_clean['norm_name'] = df_clean[p_col].apply(normalize)
            df_clean['brand'] = df_clean[p_col].apply(extract_brand)
            df_clean['size'] = df_clean[p_col].apply(extract_size)
            
            self.processed_comps[name] = {'df': df_clean, 'p_col': p_col, 'pr_col': pr_col}

    def strict_score(self, row_our, row_comp):
        score = fuzz.token_sort_ratio(row_our['norm_name'], row_comp['norm_name'])
        
        # علامات مساعدة
        if row_our['brand'] and row_comp['brand'] and row_our['brand'] != row_comp['brand']:
             if row_our['brand'] not in row_comp['brand']: return 0
        
        if row_our['size'] > 0 and row_comp['size'] > 0 and row_our['size'] != row_comp['size']:
            return 0
            
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
                if is_sample(row_our[self.our_prod_col]): continue
                
                scores = cosine_sim[idx]
                top_idx = scores.argsort()[-5:][::-1]
                
                best_match = None
                best_score = 0
                
                for c_idx in top_idx:
                    if scores[c_idx] < 0.15: continue
                    r_comp = comp_df.iloc[c_idx]
                    s = self.strict_score(row_our, r_comp)
                    if s > best_score:
                        best_score = s
                        best_match = r_comp
                
                if best_match is not None and best_score >= MATCH_THRESHOLD:
                    p_our = float(row_our[self.our_price_col]) if self.our_price_col in row_our else 0
                    p_comp = float(best_match[comp_data['pr_col']])
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
        
        # تجميع النتائج
        df = pd.DataFrame(results)
        final = []
        for n, g in df.groupby("المنتج"):
            best = g.loc[g['نسبة التطابق'].idxmax()].to_dict()
            best['جميع المنافسين'] = g.to_dict('records')
            final.append(best)
            
        return pd.DataFrame(final)

def run_full_analysis(our_df, comp_dfs, progress_callback=None, mapping=None):
    matcher = SmartMatcher(our_df, comp_dfs, mapping)
    return matcher.run(progress_callback)

def find_missing_products(our_df, comp_dfs):
    # دالة مبسطة للمفقودات
    return pd.DataFrame()
    
def export_excel(df, sheet_name="Sheet1"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as w:
        df.to_excel(w, sheet_name=sheet_name[:30], index=False)
    return output.getvalue()

def export_section_excel(df, name): return export_excel(df, name)
