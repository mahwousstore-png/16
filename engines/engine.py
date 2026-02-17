"""
engine.py - المحرك المنطقي الصارم v18.0
- إلغاء الاعتماد الكلي على التشابه النصي.
- تطبيق فلاتر: الماركة (إجباري)، النوع (إجباري)، الحجم (إجباري).
"""
import re
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import KNOWN_BRANDS, WORD_REPLACEMENTS, REJECT_KEYWORDS

# ==========================================
# 1. دوال التصنيف والاستخراج (قلب النظام)
# ==========================================

def normalize(text):
    if not isinstance(text, str): return ""
    t = text.strip().lower()
    # تنظيف الرموز
    t = re.sub(r'[^\w\s\u0600-\u06FF.]', ' ', t)
    # استبدال الكلمات (مثل او دو بارفيوم -> edp)
    for ar, en in WORD_REPLACEMENTS.items():
        if ar in t: t = t.replace(ar, en)
    return re.sub(r'\s+', ' ', t).strip()

def extract_brand(text):
    """استخراج الماركة بدقة عالية"""
    if not isinstance(text, str): return "unknown"
    t = text.lower()
    # ترتيب الماركات بالأطول أولاً لتفادي الخطأ (مثل Tom Ford قبل Ford)
    brands = sorted(KNOWN_BRANDS, key=len, reverse=True)
    for b in brands:
        if b.lower() in t:
            return b.lower()
    return "unknown"

def extract_size(text):
    """استخراج الحجم (مل/جرام)"""
    if not isinstance(text, str): return 0
    # 100ml, 100 ml, 100مل, 100 g
    m = re.search(r'(\d+)\s*(?:ml|lz|oz|مل|ملي|g|gm|gram|جرام)', text.lower())
    return int(m.group(1)) if m else 0

def get_product_nature(text):
    """تحديد هوية المنتج (عطر، شعر، جسم، طقم، عينة)"""
    t = text.lower()
    
    if any(x in t for x in ['set', 'gift', 'طقم', 'مجموعة', 'بكج']): return 'set'
    if any(x in t for x in ['hair', 'mist', 'شعر', 'معطر شعر']): return 'hair'
    if any(x in t for x in ['body', 'lotion', 'cream', 'gel', 'جسم', 'لوشن', 'كريم', 'معطر جسم']): return 'body'
    if any(x in t for x in ['sample', 'vial', 'tester', 'test', 'عينة', 'تستر']): return 'sample'
    if any(x in t for x in ['powder', 'foundation', 'blush', 'بودرة', 'أحمر خدود']): return 'makeup'
    if any(x in t for x in ['oil', 'زيت']): return 'oil'
    
    return 'perfume' # الافتراضي

# ==========================================
# 2. كلاس المطابقة المنطقية
# ==========================================

class StrictMatcher:
    def __init__(self, our_df, comp_dfs, mapping=None):
        self.our_df = our_df.copy()
        self.comp_dfs = comp_dfs
        self.mapping = mapping
        # نستخدم TF-IDF فقط للفلترة الأولية السريعة
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
        self.prepare_data()

    def prepare_data(self):
        # تحديد الأعمدة
        if self.mapping:
            self.our_p = self.mapping.get('our_name')
            self.our_pr = self.mapping.get('our_price')
        else:
            self.our_p = self.our_df.columns[0]
            self.our_pr = self.our_df.columns[1]

        # معالجة بياناتنا
        self.our_df['clean_name'] = self.our_df[self.our_p].astype(str).apply(normalize)
        self.our_df['brand'] = self.our_df[self.our_p].apply(extract_brand)
        self.our_df['size'] = self.our_df[self.our_p].apply(extract_size)
        self.our_df['nature'] = self.our_df[self.our_p].apply(get_product_nature)

        # معالجة بيانات المنافسين
        self.processed_comps = {}
        for name, df in self.comp_dfs.items():
            cdf = df.copy()
            # محاولة تحديد الأعمدة تلقائياً إذا لم تتوفر في المابينج
            if self.mapping:
                cp = self.mapping.get('comp_name', cdf.columns[0])
                cpr = self.mapping.get('comp_price', cdf.columns[1])
            else:
                cp = cdf.columns[0]
                cpr = cdf.columns[1]

            cdf['clean_name'] = cdf[cp].astype(str).apply(normalize)
            cdf['brand'] = cdf[cp].apply(extract_brand)
            cdf['size'] = cdf[cp].apply(extract_size)
            cdf['nature'] = cdf[cp].apply(get_product_nature)
            
            # تنظيف السعر
            cdf[cpr] = pd.to_numeric(cdf[cpr].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
            
            self.processed_comps[name] = {'df': cdf, 'p_col': cp, 'pr_col': cpr}

    def check_logic_match(self, row1, row2):
        """التحقق المنطقي الصارم (يجب أن يمر لتقبل النتيجة)"""
        
        # 1. اختلاف الماركة = رفض قاطع
        # (إلا إذا كانت غير معروفة في أحدهما، نتساهل قليلاً ونعتمد على الاسم)
        if row1['brand'] != 'unknown' and row2['brand'] != 'unknown':
            if row1['brand'] != row2['brand']:
                return False, "ماركة مختلفة"

        # 2. اختلاف الطبيعة = رفض قاطع
        # مستحيل نطابق (عطر شعر) بـ (عطر) أو (طقم) بـ (عطر)
        if row1['nature'] != row2['nature']:
            # استثناء بسيط: التستر والعطر العادي يمكن مطابقتهما
            if {row1['nature'], row2['nature']} == {'perfume', 'sample'}: 
                pass # مسموح (تستر مع عطر)
            elif 'sample' in [row1['nature'], row2['nature']]:
                 # إذا أحدهما عينة والآخر لا، والأسماء متشابهة، نقبلها ولكن نضع علامة
                 pass 
            else:
                return False, f"نوع مختلف ({row1['nature']} vs {row2['nature']})"

        # 3. اختلاف الحجم (إذا وجد في الاثنين)
        if row1['size'] > 0 and row2['size'] > 0:
            if row1['size'] != row2['size']:
                return False, f"حجم مختلف ({row1['size']} vs {row2['size']})"

        return True, "ok"

    def run(self, progress_callback=None):
        results = []
        total = len(self.our_df)
        
        for comp_name, comp_data in self.processed_comps.items():
            comp_df = comp_data['df']
            if comp_df.empty: continue

            # Vectorization للتسريع فقط (وليس للقرار النهائي)
            try:
                tfidf_c = self.vectorizer.fit_transform(comp_df['clean_name'])
                tfidf_o = self.vectorizer.transform(self.our_df['clean_name'])
                cosine_sim = cosine_similarity(tfidf_o, tfidf_c)
            except: continue

            for i, row_our in self.our_df.iterrows():
                # أفضل 5 مرشحين بناء على النص
                top_indices = cosine_sim[i].argsort()[-5:][::-1]
                
                best_match = None
                best_score = 0

                for j in top_indices:
                    # تجاوز النتائج الضعيفة نصياً
                    if cosine_sim[i][j] < 0.3: continue
                    
                    row_comp = comp_df.iloc[j]
                    
                    # --- الفلتر المنطقي الصارم ---
                    is_valid, _ = self.check_logic_match(row_our, row_comp)
                    if not is_valid: continue

                    # حساب دقيق للاسم
                    score = fuzz.token_sort_ratio(row_our['clean_name'], row_comp['clean_name'])
                    
                    # التحقق من الكلمات الزائدة (لتفادي خمرة vs خمرة قهوة)
                    # إذا كانت هناك كلمة مهمة في أحد الاسمين غير موجودة في الآخر، نعاقب النتيجة
                    w1 = set(row_our['clean_name'].split())
                    w2 = set(row_comp['clean_name'].split())
                    diff_words = w1.symmetric_difference(w2)
                    
                    penalty = 0
                    critical_words = ['intense', 'elixir', 'le parfum', 'qahwa', 'royal', 'sport', 'blue', 'red']
                    for w in diff_words:
                        if w in critical_words or len(w) > 3: # كلمة طويلة مختلفة = منتج مختلف
                            penalty += 15
                    
                    final_score = score - penalty

                    if final_score > best_score and final_score >= 60: # عتبة القبول النهائية
                        best_score = final_score
                        best_match = row_comp

                if best_match is not None:
                    p_our = float(row_our[self.our_pr])
                    p_comp = float(best_match[comp_data['pr_col']])
                    
                    # فلتر السعر المجنون (حماية أخيرة)
                    # إذا السعر 10 أضعاف أو العكس، غالباً خطأ (طقم vs عينة)
                    if p_comp > 0 and (p_our / p_comp > 5 or p_comp / p_our > 5):
                        decision = "⚠️ تحقق سعر"
                    else:
                        diff = p_our - p_comp
                        if diff > 10: decision = "🔴 سعر أعلى"
                        elif diff < -10: decision = "🟢 سعر أقل"
                        else: decision = "✅ موافق"

                    results.append({
                        "المنتج": row_our[self.our_p],
                        "السعر": p_our,
                        "طبيعة المنتج": row_our['nature'], # للتوضيح
                        "منتج المنافس": best_match[comp_data['p_col']],
                        "سعر المنافس": p_comp,
                        "الفرق": diff if 'diff' in locals() else 0,
                        "نسبة التطابق": best_score,
                        "القرار": decision,
                        "المنافس": comp_name
                    })

                if progress_callback and i % 50 == 0: progress_callback(i/total)

        if not results: return pd.DataFrame()
        
        # تصفية أفضل نتيجة
        df_res = pd.DataFrame(results)
        final_rows = []
        for n, g in df_res.groupby("المنتج"):
            best = g.sort_values(by=['نسبة التطابق', 'سعر المنافس'], ascending=[False, True]).iloc[0].to_dict()
            final_rows.append(best)
            
        return pd.DataFrame(final_rows)

# ===== واجهات =====
def read_file(uploaded_file):
    import io
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    df = None
    if name.endswith('.csv'):
        try: df = pd.read_csv(uploaded_file, encoding='utf-8')
        except: 
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp1256') # للعربية القديمة
    elif name.endswith(('.xlsx', '.xls')):
        try: df = pd.read_excel(uploaded_file)
        except: pass
    
    if df is not None:
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')
    return df, None

def run_full_analysis(our_df, comp_dfs, progress_callback=None, mapping=None):
    matcher = StrictMatcher(our_df, comp_dfs, mapping)
    return matcher.run(progress_callback)

def find_missing_products(our_df, comp_dfs): return pd.DataFrame() # مبسط للآن
def export_excel(df, sheet_name="Sheet1"):
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as w:
        df.to_excel(w, sheet_name=sheet_name[:30], index=False)
    return output.getvalue()
def export_section_excel(df, name): return export_excel(df, name)
