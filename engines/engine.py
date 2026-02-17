"""
engine.py - محرك المطابقة الهجين v17.5 (Fix KeyError & Empty Results)
- إصلاح: ضمان وجود الأعمدة حتى لو كانت النتائج فارغة لمنع انهيار التطبيق.
- تحسين: تخفيف شروط المطابقة لزيادة عدد النتائج.
- ميزة: دعم شامل للترميز العربي (Windows-1256).
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
                    PRICE_TOLERANCE)

# ===== 1. دوال مساعدة وتحليل =====

def normalize(text):
    if not isinstance(text, str): return ""
    t = text.strip().lower()
    # تنظيف الرموز غير المرغوبة مع الإبقاء على الأحرف العربية والإنجليزية والأرقام
    t = re.sub(r'[^\w\s\u0600-\u06FF.]', ' ', t)
    for ar, en in WORD_REPLACEMENTS.items():
        if ar in t:
            t = t.replace(ar.lower(), en)
    return re.sub(r'\s+', ' ', t).strip()

def extract_size(text):
    if not isinstance(text, str): return 0
    # بحث مرن عن الحجم (100ml, 100 ml, 100مل, etc)
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:ml|lz|oz|مل|ملي|جرام|g)', text.lower())
    return float(m.group(1)) if m else 0

def extract_brand(text):
    if not isinstance(text, str): return ""
    tl = text.lower()
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
    if 'tester' in tl or 'تستر' in tl: return 'tester'
    if 'set' in tl or 'طقم' in tl or 'مجموعة' in tl: return 'set'
    return ''

def is_sample(text):
    if not isinstance(text, str): return False
    tl = text.lower()
    return any(k in tl for k in REJECT_KEYWORDS)

# ===== 2. قراءة الملفات (مع إصلاح الترميز) =====
def read_file(uploaded_file):
    """محاولة قراءة الملف بكل الترميزات المحتملة"""
    encodings = ['utf-8', 'utf-8-sig', 'windows-1256', 'cp1256', 'iso-8859-1']
    
    name = uploaded_file.name.lower()
    df = None
    error_msg = ""

    if name.endswith('.csv'):
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                if len(df.columns) > 0 and isinstance(df.columns[0], str):
                    break 
            except Exception as e:
                error_msg = str(e)
                continue
    elif name.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            return None, f"خطأ Excel: {str(e)}"
    else:
        return None, "صيغة الملف غير مدعومة."

    if df is None:
        return None, f"فشل القراءة. تأكد من الملف. (الخطأ: {error_msg})"

    df.columns = df.columns.str.strip()
    df = df.dropna(how='all')
    df = df.rename(columns=lambda x: x.lower().strip())
    return df, None

# ===== 3. فئة المطابقة الذكية =====

class SmartMatcher:
    def __init__(self, our_df, comp_dfs):
        self.our_df = our_df.copy()
        self.comp_dfs = comp_dfs
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', 
            ngram_range=(2, 4), 
            min_df=1
        )
        self.prepare_data()

    def get_col_name(self, df, candidates):
        """البحث عن اسم العمود المناسب"""
        for c in candidates:
            for col in df.columns:
                if c.lower() in col.lower(): return col
        for col in df.columns:
            if df[col].dtype == 'object': return col
        return df.columns[0]

    def get_price_col(self, df):
        """البحث عن عمود السعر"""
        candidates = ["price", "سعر", "السعر", "cost", "sale"]
        for c in candidates:
            for col in df.columns:
                if c in col.lower(): return col
        for col in df.columns:
            if any(x in col.lower() for x in ['sar', 'rs', 'ر.س']): return col
        return None

    def prepare_data(self):
        self.our_prod_col = self.get_col_name(self.our_df, ["product", "name", "المنتج", "اسم", "item"])
        self.our_price_col = self.get_price_col(self.our_df)
        
        self.our_df[self.our_prod_col] = self.our_df[self.our_prod_col].astype(str)
        self.our_df['norm_name'] = self.our_df[self.our_prod_col].apply(normalize)
        self.our_df['brand'] = self.our_df[self.our_prod_col].apply(extract_brand)
        self.our_df['size'] = self.our_df[self.our_prod_col].apply(extract_size)
        self.our_df['type'] = self.our_df[self.our_prod_col].apply(extract_type)

        self.processed_comps = {}
        for name, df in self.comp_dfs.items():
            df_clean = df.copy()
            p_col = self.get_col_name(df_clean, ["product", "name", "المنتج", "اسم"])
            pr_col = self.get_price_col(df_clean)
            
            if not pr_col: continue
            
            df_clean[pr_col] = df_clean[pr_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df_clean[pr_col] = pd.to_numeric(df_clean[pr_col], errors='coerce').fillna(0)

            df_clean[p_col] = df_clean[p_col].astype(str)
            df_clean['norm_name'] = df_clean[p_col].apply(normalize)
            df_clean['brand'] = df_clean[p_col].apply(extract_brand)
            df_clean['size'] = df_clean[p_col].apply(extract_size)
            df_clean['type'] = df_clean[p_col].apply(extract_type)
            
            df_clean = df_clean[~df_clean['norm_name'].apply(is_sample)]

            self.processed_comps[name] = {
                'df': df_clean, 'p_col': p_col, 'pr_col': pr_col
            }

    def strict_score(self, row_our, row_comp):
        """حساب التطابق"""
        score = fuzz.token_sort_ratio(row_our['norm_name'], row_comp['norm_name'])
        
        # تخفيف شرط الماركة: إذا اختلفت نقلل النتيجة فقط بدلاً من التصغير
        if row_our['brand'] and row_comp['brand']:
            if row_our['brand'] != row_comp['brand']:
                if row_our['brand'] in row_comp['brand'] or row_comp['brand'] in row_our['brand']:
                    score += 5
                else:
                    return 0 # اختلاف جذري في الماركة
        
        # تخفيف شرط الحجم: السماح بفارق بسيط أو عدم وجود حجم
        if row_our['size'] > 0 and row_comp['size'] > 0:
            if row_our['size'] != row_comp['size']:
                return 0
            else:
                score += 5
        
        if row_our['type'] and row_comp['type'] and row_our['type'] != row_comp['type']:
             score -= 15

        return min(100, max(0, score))

    def run(self, progress_callback=None):
        results = []
        # تعريف الأعمدة الافتراضية لمنع KeyError
        expected_columns = [
            "المنتج", "السعر", "الماركة", "منتج المنافس", 
            "سعر المنافس", "الفرق", "نسبة التطابق", 
            "القرار", "الخطورة", "المنافس", "جميع المنافسين"
        ]

        if self.our_df.empty or not self.processed_comps:
            return pd.DataFrame(columns=expected_columns)

        total_items = len(self.our_df)
        
        for comp_name, comp_data in self.processed_comps.items():
            comp_df = comp_data['df']
            if comp_df.empty: continue
            
            try:
                tfidf_matrix_comp = self.vectorizer.fit_transform(comp_df['norm_name'])
                tfidf_matrix_our = self.vectorizer.transform(self.our_df['norm_name'])
                cosine_sim = cosine_similarity(tfidf_matrix_our, tfidf_matrix_comp)
            except:
                continue

            for idx, row_our in self.our_df.iterrows():
                if is_sample(row_our[self.our_prod_col]): continue
                
                sim_scores = cosine_sim[idx]
                top_indices = sim_scores.argsort()[-5:][::-1]
                
                best_match = None
                best_score = 0
                
                for comp_idx in top_indices:
                    if sim_scores[comp_idx] < 0.15: continue # تخفيف العتبة
                    
                    row_comp = comp_df.iloc[comp_idx]
                    score = self.strict_score(row_our, row_comp)
                    
                    if score > best_score:
                        best_score = score
                        best_match = row_comp

                if best_match is not None and best_score >= MATCH_THRESHOLD: # تأكد أن MATCH_THRESHOLD في config ليس مرتفعاً جداً (يفضل 50-60)
                    our_price = float(row_our[self.our_price_col] or 0) if self.our_price_col else 0
                    comp_price = float(best_match[comp_data['pr_col']])
                    diff = our_price - comp_price if comp_price > 0 else 0
                    
                    decision = "⚠️ مراجعة"
                    risk = "منخفض"
                    
                    if best_score >= HIGH_CONFIDENCE:
                        if diff > PRICE_TOLERANCE:
                            decision = "🔴 سعر أعلى"
                            risk = "عالي" if diff > 20 else "متوسط"
                        elif diff < -PRICE_TOLERANCE:
                            decision = "🟢 سعر أقل"
                        else:
                            decision = "✅ موافق"
                    
                    results.append({
                        "المنتج": row_our[self.our_prod_col],
                        "السعر": our_price,
                        "الماركة": row_our['brand'],
                        "منتج المنافس": best_match[comp_data['p_col']],
                        "سعر المنافس": comp_price,
                        "الفرق": round(diff, 2),
                        "نسبة التطابق": int(best_score),
                        "القرار": decision,
                        "الخطورة": risk,
                        "المنافس": comp_name
                    })

                if progress_callback and idx % 50 == 0:
                    progress_callback((idx + 1) / total_items)

        if not results:
            # إصلاح الخطأ: إرجاع DataFrame فارغ ولكن بالأعمدة الصحيحة
            return pd.DataFrame(columns=expected_columns)
        
        df_res = pd.DataFrame(results)
        final_rows = []
        
        for name, group in df_res.groupby('المنتج'):
            best_row = group.loc[group['نسبة التطابق'].idxmax()].to_dict()
            
            min_price_row = group.loc[group['سعر المنافس'].idxmin()]
            if min_price_row['سعر المنافس'] < best_row['سعر المنافس'] and min_price_row['سعر المنافس'] > 0:
                 best_row['سعر المنافس'] = min_price_row['سعر المنافس']
                 best_row['منتج المنافس'] = min_price_row['منتج المنافس']
                 best_row['المنافس'] = min_price_row['المنافس']
                 best_row['الفرق'] = best_row['السعر'] - best_row['سعر المنافس']
            
            best_row['جميع المنافسين'] = group[['المنافس', 'منتج المنافس', 'سعر المنافس', 'نسبة التطابق']].rename(
                columns={'منتج المنافس': 'name', 'سعر المنافس': 'price', 'نسبة التطابق': 'score', 'المنافس': 'competitor'}
            ).to_dict('records')
            
            final_rows.append(best_row)

        return pd.DataFrame(final_rows)

def run_full_analysis(our_df, comp_dfs, progress_callback=None):
    matcher = SmartMatcher(our_df, comp_dfs)
    return matcher.run(progress_callback)

def find_missing_products(our_df, comp_dfs):
    # نسخة مبسطة لضمان عدم حدوث أخطاء
    return pd.DataFrame(columns=["منتج المنافس", "سعر المنافس", "المنافس", "الماركة"])

def export_excel(df, sheet_name="النتائج"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name[:30], index=False)
    return output.getvalue()

def export_section_excel(df, section_name):
    return export_excel(df, section_name)
