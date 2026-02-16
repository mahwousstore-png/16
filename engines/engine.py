"""
engine.py - محرك المطابقة الذكي v16.0
خفيف | سريع | دقيق
"""
import re, pandas as pd, numpy as np
from rapidfuzz import fuzz, process
from io import BytesIO
from config import (MATCH_THRESHOLD, HIGH_CONFIDENCE, REVIEW_THRESHOLD,
                    PRICE_TOLERANCE, REJECT_KEYWORDS, TESTER_KEYWORDS,
                    SET_KEYWORDS, KNOWN_BRANDS, WORD_REPLACEMENTS)


# ===== تطبيع الأسماء =====
def normalize_name(name):
    if not isinstance(name, str): return ""
    t = name.strip().lower()
    for ar, en in WORD_REPLACEMENTS.items():
        t = t.replace(ar, en)
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_size(name):
    if not isinstance(name, str): return 0
    m = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ml|مل|ملي)', name.lower())
    return float(m[-1]) if m else 0


def extract_brand(name):
    if not isinstance(name, str): return ""
    nl = name.lower()
    for b in KNOWN_BRANDS:
        if b.lower() in nl: return b
    return ""


def classify_product(name):
    if not isinstance(name, str): return "عادي"
    nl = name.lower()
    for kw in REJECT_KEYWORDS:
        if kw in nl: return "عينة"
    for kw in TESTER_KEYWORDS:
        if kw in nl: return "تستر"
    for kw in SET_KEYWORDS:
        if kw in nl: return "طقم"
    return "عادي"


def get_type_label(t):
    m = {"عادي": "🟢", "تستر": "🟡", "طقم": "📦", "عينة": "🚫"}
    return m.get(t, "")


def is_sample(name):
    if not isinstance(name, str): return False
    nl = name.lower()
    return any(kw in nl for kw in REJECT_KEYWORDS)


# ===== قراءة الملفات =====
def read_file(file_data):
    data = file_data["data"]
    name = file_data["name"].lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(BytesIO(data))
        else:
            df = pd.read_excel(BytesIO(data))
    except Exception:
        return pd.DataFrame()
    df.columns = df.columns.str.strip()
    return df


def detect_columns(df):
    name_col = price_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if not name_col and any(k in cl for k in ['اسم','name','منتج','product','عنوان','title']):
            name_col = c
        if not price_col and any(k in cl for k in ['سعر','price','ثمن','cost']):
            price_col = c
    if not name_col and len(df.columns) >= 1:
        name_col = df.columns[0]
    if not price_col and len(df.columns) >= 2:
        for c in df.columns[1:]:
            if df[c].dtype in ['float64','int64']:
                price_col = c
                break
    return name_col, price_col


# ===== المطابقة الذكية =====
def smart_match(our_name, comp_names, threshold=MATCH_THRESHOLD):
    if not our_name or not comp_names:
        return None, 0, -1
    our_norm = normalize_name(our_name)
    our_size = extract_size(our_name)
    our_brand = extract_brand(our_name).lower()
    our_type = classify_product(our_name)

    best_score = 0
    best_name = None
    best_idx = -1

    for i, cn in enumerate(comp_names):
        cn_norm = normalize_name(cn)
        cn_size = extract_size(cn)
        cn_brand = extract_brand(cn).lower()
        cn_type = classify_product(cn)

        # تخطي إذا الأنواع مختلفة جداً
        if our_type != cn_type:
            continue

        # تخطي إذا الأحجام مختلفة
        if our_size > 0 and cn_size > 0 and our_size != cn_size:
            continue

        # تخطي إذا الماركات مختلفة
        if our_brand and cn_brand and our_brand != cn_brand:
            continue

        # حساب التشابه
        s1 = fuzz.token_sort_ratio(our_norm, cn_norm)
        s2 = fuzz.token_set_ratio(our_norm, cn_norm)
        s3 = fuzz.partial_ratio(our_norm, cn_norm)
        score = max(s1, s2, int(s3 * 0.9))

        # مكافأة تطابق الماركة
        if our_brand and our_brand == cn_brand:
            score = min(100, score + 5)

        # مكافأة تطابق الحجم
        if our_size > 0 and our_size == cn_size:
            score = min(100, score + 5)

        if score > best_score:
            best_score = score
            best_name = cn
            best_idx = i

    if best_score >= threshold:
        return best_name, best_score, best_idx
    return None, best_score, -1


# ===== التحليل الكامل =====
def run_full_analysis(my_file_data, comp_files_data, threshold=MATCH_THRESHOLD, progress_cb=None):
    try:
        if progress_cb: progress_cb(10, "📂 قراءة ملف المتجر...")
        my_df = read_file(my_file_data)
        if my_df.empty:
            return {"error": "ملف المتجر فارغ أو غير صالح"}

        my_name_col, my_price_col = detect_columns(my_df)
        if not my_name_col:
            return {"error": "لم يتم العثور على عمود الأسماء في ملف المتجر"}

        # قراءة المنافسين
        if progress_cb: progress_cb(20, "📂 قراءة ملفات المنافسين...")
        all_comp = []
        for cf in comp_files_data:
            cdf = read_file(cf)
            if cdf.empty: continue
            cn, cp = detect_columns(cdf)
            if not cn: continue
            comp_name = cf["name"].replace(".xlsx","").replace(".csv","").replace("_"," ")
            for _, row in cdf.iterrows():
                pname = str(row.get(cn, "")).strip()
                if not pname: continue
                price = 0
                if cp:
                    try: price = float(row[cp])
                    except: price = 0
                all_comp.append({"name": pname, "price": price, "source": comp_name})

        if not all_comp:
            return {"error": "ملفات المنافسين فارغة"}

        comp_names = [c["name"] for c in all_comp]

        # المطابقة
        results_raise = []
        results_lower = []
        results_approved = []
        results_missing = []
        results_review = []
        all_results = []

        total = len(my_df)
        if progress_cb: progress_cb(30, f"🔍 مطابقة {total} منتج...")

        for idx, row in my_df.iterrows():
            pname = str(row.get(my_name_col, "")).strip()
            if not pname: continue

            # استثناء العينات فقط
            if is_sample(pname):
                continue

            our_price = 0
            if my_price_col:
                try: our_price = float(row[my_price_col])
                except: our_price = 0

            ptype = classify_product(pname)
            psize = extract_size(pname)
            pbrand = extract_brand(pname)

            match_name, match_score, match_idx = smart_match(pname, comp_names, threshold)

            if match_name and match_idx >= 0:
                comp = all_comp[match_idx]
                comp_price = comp["price"]
                diff = our_price - comp_price
                pct = (diff / comp_price * 100) if comp_price > 0 else 0

                # تحديد الخطورة
                if abs(diff) > 50: risk = "حرج"
                elif abs(diff) > 20: risk = "متوسط"
                else: risk = "منخفض"

                # تفسير القرار
                if diff > PRICE_TOLERANCE:
                    decision = "رفع سعر"
                    reason = f"سعرنا أعلى من المنافس بـ {diff:.0f} ر.س ({pct:.1f}%)"
                elif diff < -PRICE_TOLERANCE:
                    decision = "خفض سعر"
                    reason = f"سعرنا أقل من المنافس بـ {abs(diff):.0f} ر.س ({abs(pct):.1f}%)"
                else:
                    decision = "موافق"
                    reason = f"الفرق ضمن الحد المسموح ({diff:+.0f} ر.س)"

                rec = {
                    "المنتج": pname,
                    "السعر": our_price,
                    "اسم المنافس": match_name,
                    "أقل سعر منافس": comp_price,
                    "الفرق": diff,
                    "النسبة": round(pct, 1),
                    "نسبة التطابق": match_score,
                    "المنافس": comp["source"],
                    "النوع": ptype,
                    "الحجم": psize,
                    "الماركة": pbrand,
                    "الخطورة": risk,
                    "القرار": decision,
                    "التفسير": reason,
                }

                all_results.append(rec)

                if match_score < REVIEW_THRESHOLD:
                    results_review.append(rec)
                elif decision == "رفع سعر":
                    results_raise.append(rec)
                elif decision == "خفض سعر":
                    results_lower.append(rec)
                else:
                    results_approved.append(rec)
            else:
                # لم يتم إيجاد مطابقة → مفقود عند المنافس
                pass

            if progress_cb and idx % 50 == 0:
                pct_done = 30 + int((idx / max(total, 1)) * 50)
                progress_cb(pct_done, f"🔍 تحليل {idx}/{total}...")

        # المنتجات المفقودة (موجودة عند المنافس وليست عندنا)
        if progress_cb: progress_cb(82, "📋 تحليل المنتجات المفقودة...")
        my_names = [str(row.get(my_name_col, "")).strip().lower() for _, row in my_df.iterrows()]
        for comp in all_comp:
            if is_sample(comp["name"]): continue
            cn_norm = normalize_name(comp["name"])
            found = False
            for mn in my_names:
                if fuzz.token_sort_ratio(cn_norm, normalize_name(mn)) >= threshold:
                    found = True
                    break
            if not found:
                results_missing.append({
                    "المنتج": comp["name"],
                    "السعر": comp["price"],
                    "المنافس": comp["source"],
                    "النوع": classify_product(comp["name"]),
                    "الحجم": extract_size(comp["name"]),
                    "الماركة": extract_brand(comp["name"]),
                })

        if progress_cb: progress_cb(90, "📊 تجهيز النتائج...")

        # تحويل إلى DataFrames
        df_raise = pd.DataFrame(results_raise)
        df_lower = pd.DataFrame(results_lower)
        df_approved = pd.DataFrame(results_approved)
        df_missing = pd.DataFrame(results_missing)
        df_review = pd.DataFrame(results_review)
        df_all = pd.DataFrame(all_results)

        # ترتيب حسب الفرق
        if not df_raise.empty:
            df_raise = df_raise.sort_values("الفرق", ascending=False)
        if not df_lower.empty:
            df_lower = df_lower.sort_values("الفرق", ascending=True)

        stats = {
            "total": len(all_results),
            "raise_count": len(results_raise),
            "lower_count": len(results_lower),
            "approved_count": len(results_approved),
            "missing_count": len(results_missing),
            "review_count": len(results_review),
            "critical": len([r for r in all_results if r.get("الخطورة") == "حرج"]),
            "avg_diff": np.mean([r["الفرق"] for r in all_results]) if all_results else 0,
            "competitors": len(comp_files_data),
            "threshold": threshold,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        }

        return {
            "raise": df_raise, "lower": df_lower, "approved": df_approved,
            "missing": df_missing, "review": df_review, "all": df_all,
            "stats": stats,
        }

    except Exception as e:
        return {"error": str(e)}


def export_excel(results):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as w:
        for key, label in [("raise","رفع سعر"),("lower","خفض سعر"),("approved","موافق"),("missing","مفقودة"),("review","مراجعة")]:
            df = results.get(key, pd.DataFrame())
            if not df.empty:
                df.to_excel(w, sheet_name=label, index=False)
    return output.getvalue()
