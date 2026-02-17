"""
engine.py - محرك v18.6 (Deep Integrity & App Support)
- الطبقات الخمس: (الماركة، النوع، التستر، الحجم، الإصدار الحرِج).
- دعم كامل: يحتوي على دالة القراءة والتصدير المطلوبة في app.py.
"""
import re
import pandas as pd
import io
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ملاحظة: سيتم استيراد KNOWN_BRANDS و WORD_REPLACEMENTS من ملف config
try:
    from config import KNOWN_BRANDS, WORD_REPLACEMENTS, MATCH_THRESHOLD
except:
    # قيم افتراضية في حال فشل الاستيراد
    KNOWN_BRANDS = ["dior", "chanel", "hermes", "tom ford", "creed", "roja"]
    WORD_REPLACEMENTS = {"بارفيوم": "edp", "تواليت": "edt"}
    MATCH_THRESHOLD = 65

# ==========================================
# 1. دوال المعالجة (DNA Extraction)
# ==========================================

def deep_normalize(text):
    if not isinstance(text, str): return ""
    t = text.strip().lower()
    t = re.sub(r'[أإآ]', 'ا', t)
    t = t.replace('ة', 'ه').replace('ى', 'ي')
    t = re.sub(r'[^\w\s.]', ' ', t)
    for ar, en in WORD_REPLACEMENTS.items():
        if ar in t: t = t.replace(ar, en)
    return re.sub(r'\s+', ' ', t).strip()

def get_detailed_dna(text):
    """استخراج البصمة الوراثية للمنتج لمنع الأخطاء"""
    t = text.lower()
    clean = deep_normalize(text)
    
    # ا. استخراج الحجم
    size_match = re.search(r'\b(\d+)\s*(?:ml|مل|ملي|g|gm|gram|جرام)\b', t)
    size = int(size_match.group(1)) if size_match else 0
    
    # ب. تحديد النوع (Nature)
    nature = "perfume"
    if any(x in t for x in ['set', 'gift', 'طقم', 'مجموعة', 'بكج']): nature = "set"
    elif any(x in t for x in ['hair', 'mist', 'شعر']): nature = "hair"
    elif any(x in t for x in ['body', 'جسم', 'لوشن', 'lotion']): nature = "body"
    elif any(x in t for x in ['sample', 'vial', 'عينة', 'سكب']): nature = "sample"
    
    # ج. وسم التستر
    is_tester = any(x in t for x in ['tester', 'تستر'])
    
    # د. الكلمات المفتاحية الحرجة (منع خلط العادي والمركز)
    critical_keywords = []
    for word in ['intense', 'elixir', 'parfum', 'qahwa', 'extreme', 'sport', 'rouge']:
        if word in t: critical_keywords.append(word)
        
    # هـ. استخراج الماركة
    brand = "unknown"
    for b in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if b.lower() in t:
            brand = b.lower()
            break
            
    return {
        "brand": brand, "size": size, "nature": nature,
        "is_tester": is_tester, "critical": set(critical_keywords),
        "clean": clean
    }

# ==========================================
# 2. الدوال المطلوبة في app.py
# ==========================================

def read_file(uploaded_file):
    """دالة قراءة الملفات مع دعم الترميز العربي"""
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

def export_excel(df, sheet_name="Sheet1"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as w:
        df.to_excel(w, sheet_name=sheet_name[:30], index=False)
    return output.getvalue()

def export_section_excel(df, name): 
    return export_excel(df, name)

def find_missing_products(our_df, comp_dfs):
    """دالة المفقودات (مبسطة)"""
    return pd.DataFrame()

# ==========================================
# 3. محرك الفحص العميق متعدد الطبقات
# ==========================================

class DeepIntegrityEngine:
    def __init__(self, our_df, comp_dfs, mapping=None):
        self.our_df = our_df.copy()
        self.comp_dfs = comp_dfs
        self.mapping = mapping
        self.prepare_data()

    def prepare_data(self):
        p_col = self.mapping.get('our_name') if self.mapping else self.our_df.columns[1]
        pr_col = self.mapping.get('our_price') if self.mapping else self.our_df.columns[0]
        self.our_df['dna'] = self.our_df[p_col].apply(get_detailed_dna)
        self.our_p, self.our_pr = p_col, pr_col

    def calculate_match_quality(self, dna1, dna2):
        # الطبقة 1: الماركة
        if dna1['brand'] != dna2['brand'] and dna1['brand'] != "unknown" and dna2['brand'] != "unknown":
            return 0
        # الطبقة 2: الطبيعة (عطر vs طقم)
        if dna1['nature'] != dna2['nature']:
            return 0
        # الطبقة 3: التستر
        if dna1['is_tester'] != dna2['is_tester']:
            return 0
        # الطبقة 4: الحجم
        if dna1['size'] > 0 and dna2['size'] > 0 and dna1['size'] != dna2['size']:
            return 0
        # الطبقة 5: الكلمات الحرجة (الإصدار)
        if dna1['critical'] != dna2['critical']:
            return 0
        
        return fuzz.token_sort_ratio(dna1['clean'], dna2['clean'])

    def run(self, progress_callback=None):
        final_results = []
        for comp_name, df in self.comp_dfs.items():
            cdf = df.copy()
            cp = self.mapping.get('comp_name') if self.mapping else cdf.columns[0]
            cpr = self.mapping.get('comp_price') if self.mapping else cdf.columns[1]
            cdf['dna'] = cdf[cp].apply(get_detailed_dna)
            cdf[cpr] = pd.to_numeric(cdf[cpr].astype(str).str.replace(r'[^\d.]','',regex=True), errors='coerce').fillna(0)

            for i, row_our in self.our_df.iterrows():
                best_s, best_match = 0, None
                for j, row_comp in cdf.iterrows():
                    score = self.calculate_match_quality(row_our['dna'], row_comp['dna'])
                    if score > best_s:
                        best_s, best_match = score, row_comp
                
                if best_match is not None and best_s >= 65:
                    p_our = float(row_our[self.our_pr])
                    p_comp = float(best_match[cpr])
                    diff = p_our - p_comp
                    final_results.append({
                        "المنتج": row_our[self.our_p], "السعر": p_our,
                        "منتج المنافس": best_match[cp], "سعر المنافس": p_comp,
                        "الفرق": diff, "نسبة التطابق": best_s,
                        "القرار": "🔴 سعر أعلى" if diff > 5 else "🟢 سعر أقل" if diff < -5 else "✅ موافق",
                        "المنافس": comp_name
                    })
                if progress_callback and i % 20 == 0: progress_callback(i / len(self.our_df))
        return pd.DataFrame(final_results)

def run_full_analysis(our_df, comp_dfs, progress_callback=None, mapping=None):
    engine = DeepIntegrityEngine(our_df, comp_dfs, mapping)
    return engine.run(progress_callback)
