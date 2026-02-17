import re
import pandas as pd
import io
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import KNOWN_BRANDS, WORD_REPLACEMENTS, MATCH_THRESHOLD

# ==========================================
# الدوال التحليلية العميقة
# ==========================================

def deep_normalize(text):
    if not isinstance(text, str): return ""
    t = text.strip().lower()
    # توحيد الأحرف العربية
    t = re.sub(r'[أإآ]', 'ا', t)
    t = t.replace('ة', 'ه').replace('ى', 'ي')
    # إبقاء الأحرف والأرقام فقط
    t = re.sub(r'[^\w\s.]', ' ', t)
    for ar, en in WORD_REPLACEMENTS.items():
        if ar in t: t = t.replace(ar, en)
    return re.sub(r'\s+', ' ', t).strip()

def get_detailed_dna(text):
    """تحليل الحامض النووي للمنتج لاستخراج كل التفاصيل"""
    t = text.lower()
    clean = deep_normalize(text)
    
    # 1. استخراج الحجم
    size_match = re.search(r'\b(\d+)\s*(?:ml|مل|ملي|g|gm|gram|جرام)\b', t)
    size = int(size_match.group(1)) if size_match else 0
    
    # 2. تحديد النوع (Nature)
    nature = "perfume"
    if any(x in t for x in ['set', 'gift', 'طقم', 'مجموعة', 'بكج']): nature = "set"
    elif any(x in t for x in ['hair', 'mist', 'شعر']): nature = "hair"
    elif any(x in t for x in ['body', 'جسم', 'لوشن', 'lotion']): nature = "body"
    elif any(x in t for x in ['sample', 'vial', 'عينة']): nature = "sample"
    
    # 3. وسم التستر
    is_tester = any(x in t for x in ['tester', 'تستر'])
    
    # 4. استخراج الكلمات المفتاحية الحرجة (المميزة للإصدار)
    critical_keywords = []
    for word in ['intense', 'elixir', 'parfum', 'qahwa', 'edp', 'edt', 'extreme', 'sport', 'black', 'white', 'rouge']:
        if word in t: critical_keywords.append(word)
        
    # 5. استخراج الماركة
    brand = "unknown"
    for b in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if b.lower() in t:
            brand = b.lower()
            break
            
    return {
        "brand": brand,
        "size": size,
        "nature": nature,
        "is_tester": is_tester,
        "critical": set(critical_keywords),
        "clean": clean
    }

# ==========================================
# محرك الطبقات المتعددة
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
        
        # تحليل بياناتنا بالكامل
        self.our_df['dna'] = self.our_df[p_col].apply(get_detailed_dna)
        self.our_p, self.our_pr = p_col, pr_col

    def calculate_match_quality(self, dna1, dna2):
        """الطبقات الخمس للتحقق من الجودة"""
        
        # الطبقة 1: الماركة (إجباري)
        if dna1['brand'] != dna2['brand'] and dna1['brand'] != "unknown" and dna2['brand'] != "unknown":
            return 0, "إختلاف ماركة"

        # الطبقة 2: الطبيعة (إجباري)
        if dna1['nature'] != dna2['nature']:
            return 0, "إختلاف نوع (عطر vs طقم/شعر)"

        # الطبقة 3: فحص التستر
        if dna1['is_tester'] != dna2['is_tester']:
            return 0, "أحدهما تستر والآخر أصلي"

        # الطبقة 4: الحجم (تنبيه)
        if dna1['size'] > 0 and dna2['size'] > 0 and dna1['size'] != dna2['size']:
            return 0, "إختلاف حجم"

        # الطبقة 5: الكلمات الحرجة (Intense, Elixir...)
        # إذا وجد إصدار خاص في طرف ولم يوجد في الآخر، نرفض المطابقة
        if dna1['critical'] != dna2['critical']:
            return 0, "إختلاف إصدار (مركز/عادي)"

        # إذا اجتاز كل الاختبارات المنطقية، نحسب التشابه النصي
        score = fuzz.token_sort_ratio(dna1['clean'], dna2['clean'])
        return score, "OK"

    def run(self, progress_callback=None):
        final_results = []
        
        for comp_name, df in self.comp_dfs.items():
            cdf = df.copy()
            cp = self.mapping.get('comp_name') if self.mapping else cdf.columns[0]
            cpr = self.mapping.get('comp_price') if self.mapping else cdf.columns[1]
            
            # تجهيز بيانات المنافس
            cdf['dna'] = cdf[cp].apply(get_detailed_dna)
            cdf[cpr] = pd.to_numeric(cdf[cpr].astype(str).str.replace(r'[^\d.]','',regex=True), errors='coerce').fillna(0)

            for i, row_our in self.our_df.iterrows():
                best_s = 0
                best_match = None
                
                # البحث في كل منتجات المنافس (بدون Vectorization لضمان عدم إضاعة أي فرصة)
                # هذا أبطأ قليلاً ولكنه يضمن فحص كل الاحتمالات
                for j, row_comp in cdf.iterrows():
                    score, reason = self.calculate_match_quality(row_our['dna'], row_comp['dna'])
                    
                    if score > best_s:
                        best_s = score
                        best_match = row_comp

                if best_match is not None and best_s >= 65: # عتبة قبول ذكية
                    p_our = float(row_our[self.our_pr])
                    p_comp = float(best_match[cpr])
                    diff = p_our - p_comp
                    
                    final_results.append({
                        "المنتج": row_our[self.our_p],
                        "السعر": p_our,
                        "منتج المنافس": best_match[cp],
                        "سعر المنافس": p_comp,
                        "الفرق": diff,
                        "نسبة التطابق": best_s,
                        "القرار": "🔴 سعر أعلى" if diff > 5 else "🟢 سعر أقل" if diff < -5 else "✅ موافق",
                        "المنافس": comp_name
                    })
                
                if progress_callback and i % 20 == 0:
                    progress_callback(i / len(self.our_df))

        return pd.DataFrame(final_results)

def run_full_analysis(our_df, comp_dfs, progress_callback=None, mapping=None):
    engine = DeepIntegrityEngine(our_df, comp_dfs, mapping)
    return engine.run(progress_callback)

def find_missing_products(our_df, comp_dfs): return pd.DataFrame()
