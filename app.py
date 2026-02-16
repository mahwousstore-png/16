"""
نظام التسعير الذكي - مهووس v17.0
نظام متكامل لتسعير العطور بالذكاء الصناعي
"""
import streamlit as st
import pandas as pd
import io, json, time
from datetime import datetime
from config import *
from styles import get_styles
from engines.engine import run_full_analysis, export_excel, is_sample
from engines.ai_engine import (call_ai, chat_with_ai, verify_match, analyze_product,
                                bulk_verify, suggest_price, process_paste, check_duplicate)
from utils.helpers import (apply_filters, get_filter_options, export_to_excel,
                           export_multiple_sheets, BackgroundTask, parse_pasted_text,
                           format_price, format_diff, get_color_for_diff, safe_float, log_event)
from utils.make_helper import (send_price_updates, send_new_products, send_missing_products,
                                send_to_make, send_single_product, test_webhook,
                                verify_webhook_connection, export_to_make_format)
from utils.db_manager import log_event as db_log, log_decision, log_analysis, get_events, get_decisions

# ===== إعداد الصفحة =====
st.set_page_config(page_title=APP_TITLE, page_icon="🧴", layout="wide", initial_sidebar_state="expanded")
st.markdown(get_styles(), unsafe_allow_html=True)

# ===== الحالة =====
if "results" not in st.session_state:
    st.session_state.results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "bg_tasks" not in st.session_state:
    st.session_state.bg_tasks = {}

# ===== الشريط الجانبي =====
with st.sidebar:
    st.markdown(f"## 🧴 {APP_TITLE}")
    st.markdown(f"**الإصدار:** {APP_VERSION}")
    st.markdown("---")
    page = st.radio("📑 الأقسام", SECTIONS, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"⏰ {datetime.now().strftime('%H:%M:%S')}")

# ===== دوال مشتركة =====
def render_filters(df, section_key):
    """عرض فلاتر متقدمة"""
    opts = get_filter_options(df)
    with st.expander("🔍 فلاتر متقدمة", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        filters = {}
        with c1:
            filters["search"] = st.text_input("🔎 بحث", key=f"search_{section_key}")
            filters["brand"] = st.selectbox("الماركة", opts["brands"], key=f"brand_{section_key}")
        with c2:
            filters["competitor"] = st.selectbox("المنافس", opts["competitors"], key=f"comp_{section_key}")
            filters["type"] = st.selectbox("النوع", opts.get("types", ["الكل"]), key=f"type_{section_key}")
        with c3:
            filters["price_min"] = st.number_input("السعر من", 0.0, key=f"pmin_{section_key}")
            filters["price_max"] = st.number_input("السعر إلى", 10000.0, value=10000.0, key=f"pmax_{section_key}")
        with c4:
            filters["match_min"] = st.slider("أقل تطابق %", 0, 100, 0, key=f"match_{section_key}")
            filters["size"] = st.selectbox("الحجم", opts.get("sizes", ["الكل"]), key=f"size_{section_key}")
    return filters


def render_product_table(df, section_key, show_actions=True):
    """عرض جدول منتجات احترافي مع أزرار"""
    if df.empty:
        st.info("لا توجد بيانات")
        return

    # أزرار عامة
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button(f"📥 تصدير Excel", key=f"excel_{section_key}"):
            excel = export_to_excel(df, section_key)
            st.download_button("⬇️ تحميل", excel, f"{section_key}.xlsx", key=f"dl_{section_key}")
    with col_b:
        if st.button(f"🤖 تحقق AI جماعي", key=f"bulk_ai_{section_key}"):
            with st.spinner("جاري التحقق بالذكاء الصناعي..."):
                items = []
                for _, r in df.head(20).iterrows():
                    items.append({"our": str(r.get("المنتج", "")), "comp": str(r.get("اسم المنافس", "")),
                                  "our_price": safe_float(r.get("السعر", 0)), "comp_price": safe_float(r.get("أقل سعر منافس", 0))})
                result = bulk_verify(items, section_key)
                if result["success"]:
                    st.success(f"✅ نتيجة AI ({result.get('source', '')}):")
                    st.markdown(result["response"])
                else:
                    st.error(result["response"])
    with col_c:
        if st.button(f"📤 تصدير Make", key=f"make_{section_key}"):
            products = export_to_make_format(df, section_key)
            wh_type = "new" if section_key == "missing" else "update"
            result = send_to_make(products, wh_type)
            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])
    with col_d:
        st.metric("المنتجات", len(df))

    # عرض الجدول
    st.markdown(f"**إجمالي: {len(df)} منتج**")

    for idx, row in df.iterrows():
        our_name = str(row.get("المنتج", ""))
        comp_name = str(row.get("اسم المنافس", ""))
        our_price = safe_float(row.get("السعر", 0))
        comp_price = safe_float(row.get("أقل سعر منافس", 0))
        diff = safe_float(row.get("الفرق", 0))
        match_pct = safe_float(row.get("نسبة التطابق", 0))
        brand = str(row.get("الماركة", ""))
        competitor = str(row.get("المنافس", ""))

        diff_color = get_color_for_diff(diff)
        match_color = "#00cc00" if match_pct >= 80 else "#ff8800" if match_pct >= 60 else "#ff4444"

        with st.container():
            st.markdown(f"""<div style="border:1px solid #333;border-radius:8px;padding:10px;margin:5px 0;background:#1a1a2e;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="flex:1;padding:5px;">
                    <span style="color:#aaa;font-size:11px;">منتجنا</span><br>
                    <strong style="color:#fff;font-size:13px;">{our_name}</strong><br>
                    <span style="color:#4fc3f7;font-size:14px;font-weight:bold;">{our_price:,.2f} ر.س</span>
                </div>
                <div style="text-align:center;padding:5px;">
                    <span style="color:{match_color};font-size:12px;font-weight:bold;">{match_pct:.0f}%</span><br>
                    <span style="color:{diff_color};font-size:13px;font-weight:bold;">{format_diff(diff)}</span><br>
                    <span style="color:#888;font-size:10px;">{brand} | {competitor}</span>
                </div>
                <div style="flex:1;padding:5px;text-align:left;">
                    <span style="color:#aaa;font-size:11px;">المنافس</span><br>
                    <strong style="color:#fff;font-size:13px;">{comp_name}</strong><br>
                    <span style="color:#ff9800;font-size:14px;font-weight:bold;">{comp_price:,.2f} ر.س</span>
                </div>
            </div></div>""", unsafe_allow_html=True)

            if show_actions:
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    if st.button("🤖 تحقق", key=f"ai_{section_key}_{idx}"):
                        with st.spinner("..."):
                            r = verify_match(our_name, comp_name, our_price, comp_price)
                            if r["success"]:
                                st.info(f"{'✅' if r.get('match') else '❌'} ثقة: {r.get('confidence', 0)}% - {r.get('reason', '')}")
                            else:
                                st.error(r["reason"])
                with c2:
                    if st.button("✅ موافق", key=f"ok_{section_key}_{idx}"):
                        st.session_state.decisions[f"{section_key}_{idx}"] = "approved"
                        log_decision(our_name, section_key, "approved", "موافقة يدوية")
                        st.success("تم النقل للموافق")
                with c3:
                    if st.button("📤 Make", key=f"mk_{section_key}_{idx}"):
                        prod = {"name": our_name, "price": our_price, "comp_name": comp_name, "comp_price": comp_price}
                        r = send_single_product(prod, "update")
                        st.success(r["message"]) if r["success"] else st.error(r["message"])
                with c4:
                    if st.button("🗑️ إزالة", key=f"rm_{section_key}_{idx}"):
                        st.session_state.decisions[f"{section_key}_{idx}"] = "removed"
                        log_decision(our_name, section_key, "removed", "إزالة يدوية")
                        st.warning("تم الإزالة")
                with c5:
                    if st.button("⏸️ تأجيل", key=f"df_{section_key}_{idx}"):
                        st.session_state.decisions[f"{section_key}_{idx}"] = "deferred"
                        log_decision(our_name, section_key, "deferred", "تأجيل")
                        st.info("تم التأجيل")


def render_paste_section(section_key):
    """قسم اللصق في كل صفحة"""
    with st.expander("📋 لصق بيانات / أوامر AI", expanded=False):
        pasted = st.text_area("الصق هنا (بيانات أو أوامر):", key=f"paste_{section_key}", height=100)
        if pasted and st.button("🚀 معالجة", key=f"proc_paste_{section_key}"):
            with st.spinner("جاري المعالجة بالذكاء الصناعي..."):
                result = process_paste(pasted, section_key)
                if result["success"]:
                    st.markdown(result["response"])
                else:
                    st.error(result["response"])


# ==========================================
# ===== الأقسام =====
# ==========================================

# ===== 1. لوحة التحكم =====
if page == "📊 لوحة التحكم":
    st.header("📊 لوحة التحكم")
    db_log("dashboard", "view", "فتح لوحة التحكم")

    if st.session_state.results:
        r = st.session_state.results
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("إجمالي المنتجات", r.get("total_our", 0))
        c2.metric("متطابقة", r.get("matched", 0))
        c3.metric("مفقودة", r.get("missing", 0))
        c4.metric("أعلى سعراً", len(r.get("price_raise", pd.DataFrame())))
        c5.metric("أقل سعراً", len(r.get("price_lower", pd.DataFrame())))

        c1, c2, c3 = st.columns(3)
        c1.metric("بحاجة مراجعة", len(r.get("review", pd.DataFrame())))
        c2.metric("موافق عليها", len(r.get("approved", pd.DataFrame())))
        c3.metric("المنافسين", r.get("total_comp", 0))

        # تصدير شامل
        if st.button("📥 تصدير كل الأقسام Excel"):
            sheets = {}
            for key in ["price_raise", "price_lower", "approved", "missing", "review"]:
                if key in r and not r[key].empty:
                    sheets[key] = r[key]
            if sheets:
                excel = export_multiple_sheets(sheets)
                st.download_button("⬇️ تحميل الملف الشامل", excel, "all_sections.xlsx")

        # سجل الأحداث
        with st.expander("📜 سجل الأحداث"):
            events = get_events(limit=20)
            for e in events:
                st.text(f"[{e['timestamp']}] {e['page']} - {e['event_type']}: {e['details']}")
    else:
        st.info("📂 ارفع ملفات Excel من قسم 'رفع الملفات' للبدء")


# ===== 2. رفع الملفات =====
elif page == "📂 رفع الملفات":
    st.header("📂 رفع وتحليل الملفات")
    db_log("upload", "view", "فتح صفحة الرفع")

    c1, c2 = st.columns(2)
    with c1:
        our_file = st.file_uploader("📄 ملف منتجاتنا (Excel)", type=["xlsx", "xls"], key="our")
    with c2:
        comp_file = st.file_uploader("📄 ملف المنافسين (Excel)", type=["xlsx", "xls"], key="comp")

    bg_mode = st.checkbox("⚡ معالجة في الخلفية", value=False)

    if our_file and comp_file:
        if st.button("🚀 بدء التحليل", type="primary"):
            db_log("upload", "analysis_start", f"بدء تحليل: {our_file.name} vs {comp_file.name}")

            with st.spinner("⏳ جاري التحليل الذكي..."):
                try:
                    our_df = pd.read_excel(our_file)
                    comp_df = pd.read_excel(comp_file)
                    results = run_full_analysis(our_df, comp_df)
                    st.session_state.results = results

                    log_analysis(our_file.name, comp_file.name,
                                 results.get("total_our", 0), results.get("matched", 0),
                                 results.get("missing", 0))

                    st.success(f"✅ تم التحليل! {results.get('matched', 0)} متطابق | {results.get('missing', 0)} مفقود")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ خطأ: {str(e)}")


# ===== 3. سعر أعلى =====
elif page == "🔴 سعر أعلى":
    st.header("🔴 منتجات سعرنا أعلى من المنافسين")
    db_log("price_raise", "view")

    if st.session_state.results and "price_raise" in st.session_state.results:
        df = st.session_state.results["price_raise"]
        if not df.empty:
            filters = render_filters(df, "raise")
            filtered = apply_filters(df, filters)
            render_paste_section("price_raise")
            render_product_table(filtered, "raise")
        else:
            st.success("✅ لا توجد منتجات بسعر أعلى")
    else:
        st.info("ارفع الملفات أولاً")


# ===== 4. سعر أقل =====
elif page == "🟢 سعر أقل":
    st.header("🟢 منتجات سعرنا أقل من المنافسين")
    db_log("price_lower", "view")

    if st.session_state.results and "price_lower" in st.session_state.results:
        df = st.session_state.results["price_lower"]
        if not df.empty:
            filters = render_filters(df, "lower")
            filtered = apply_filters(df, filters)
            render_paste_section("price_lower")
            render_product_table(filtered, "lower")
        else:
            st.info("لا توجد منتجات بسعر أقل")
    else:
        st.info("ارفع الملفات أولاً")


# ===== 5. موافق عليها =====
elif page == "✅ موافق عليها":
    st.header("✅ المنتجات الموافق عليها")
    db_log("approved", "view")

    if st.session_state.results and "approved" in st.session_state.results:
        df = st.session_state.results["approved"]
        if not df.empty:
            filters = render_filters(df, "approved")
            filtered = apply_filters(df, filters)
            render_paste_section("approved")
            render_product_table(filtered, "approved")
        else:
            st.info("لا توجد منتجات موافق عليها")
    else:
        st.info("ارفع الملفات أولاً")


# ===== 6. منتجات مفقودة =====
elif page == "🔍 منتجات مفقودة":
    st.header("🔍 منتجات المنافسين غير الموجودة عندنا")
    db_log("missing", "view")

    if st.session_state.results and "missing" in st.session_state.results:
        df = st.session_state.results["missing"]
        if not df.empty:
            st.warning(f"⚠️ {len(df)} منتج مفقود - تحقق بدقة قبل الإضافة لتجنب التكرار")

            # فلاتر
            opts = get_filter_options(df)
            with st.expander("🔍 فلاتر", expanded=False):
                c1, c2 = st.columns(2)
                search = c1.text_input("🔎 بحث", key="miss_search")
                brand_f = c2.selectbox("الماركة", opts["brands"], key="miss_brand")

            filtered = df.copy()
            if search:
                filtered = filtered[filtered.apply(lambda r: search.lower() in str(r.values).lower(), axis=1)]
            if brand_f != "الكل":
                filtered = filtered[filtered.get("الماركة", pd.Series(dtype=str)).str.contains(brand_f, case=False, na=False)]

            # أزرار عامة
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("📥 تصدير Excel", key="miss_excel"):
                    excel = export_to_excel(filtered, "مفقودة")
                    st.download_button("⬇️ تحميل", excel, "missing.xlsx", key="miss_dl")
            with c2:
                if st.button("🤖 تحقق AI جماعي", key="miss_bulk_ai"):
                    with st.spinner("جاري التحقق..."):
                        items = []
                        for _, r in filtered.head(20).iterrows():
                            items.append({"our": "", "comp": str(r.get("المنتج", "")),
                                          "our_price": 0, "comp_price": safe_float(r.get("السعر", 0))})
                        result = bulk_verify(items, "missing")
                        if result["success"]:
                            st.markdown(result["response"])
            with c3:
                if st.button("📤 تصدير Make", key="miss_make"):
                    products = [{"name": str(r.get("المنتج", "")), "price": safe_float(r.get("السعر", 0)),
                                 "brand": str(r.get("الماركة", "")), "competitor": str(r.get("المنافس", ""))}
                                for _, r in filtered.iterrows()]
                    result = send_missing_products(products)
                    st.success(result["message"]) if result["success"] else st.error(result["message"])

            render_paste_section("missing")

            # عرض المنتجات
            for idx, row in filtered.iterrows():
                name = str(row.get("المنتج", ""))
                price = safe_float(row.get("السعر", 0))
                brand = str(row.get("الماركة", ""))
                comp = str(row.get("المنافس", ""))

                with st.container():
                    st.markdown(f"""<div style="border:1px solid #444;border-radius:8px;padding:10px;margin:5px 0;background:#1a1a2e;">
                    <strong style="color:#ff9800;">{name}</strong><br>
                    <span style="color:#4fc3f7;">{price:,.2f} ر.س</span> | {brand} | {comp}
                    </div>""", unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        if st.button("🤖 تحقق تكرار", key=f"dup_{idx}"):
                            with st.spinner("..."):
                                our_products = []
                                if st.session_state.results:
                                    for key in ["price_raise", "price_lower", "approved"]:
                                        if key in st.session_state.results:
                                            our_products.extend(st.session_state.results[key].get("المنتج", pd.Series()).tolist())
                                r = check_duplicate(name, our_products[:50])
                                if r["success"]:
                                    st.markdown(r["response"])
                    with c2:
                        if st.button("✅ إضافة", key=f"add_{idx}"):
                            log_decision(name, "missing", "to_add", "إضافة للمتجر")
                            st.success("تم وضع علامة للإضافة")
                    with c3:
                        if st.button("📤 Make", key=f"mk_miss_{idx}"):
                            r = send_single_product({"name": name, "price": price, "brand": brand}, "new")
                            st.success(r["message"]) if r["success"] else st.error(r["message"])
                    with c4:
                        if st.button("🗑️ تجاهل", key=f"ign_{idx}"):
                            log_decision(name, "missing", "ignored", "تجاهل")
                            st.warning("تم التجاهل")
        else:
            st.success("✅ لا توجد منتجات مفقودة")
    else:
        st.info("ارفع الملفات أولاً")


# ===== 7. مراجعة =====
elif page == "⚠️ تحت المراجعة":
    st.header("⚠️ منتجات تحتاج مراجعة")
    db_log("review", "view")

    if st.session_state.results and "review" in st.session_state.results:
        df = st.session_state.results["review"]
        if not df.empty:
            st.warning(f"⚠️ {len(df)} منتج يحتاج مراجعة - تطابق غير مؤكد")
            filters = render_filters(df, "review")
            filtered = apply_filters(df, filters)
            render_paste_section("review")

            # أزرار عامة
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("📥 تصدير Excel", key="rev_excel"):
                    excel = export_to_excel(filtered, "مراجعة")
                    st.download_button("⬇️ تحميل", excel, "review.xlsx", key="rev_dl")
            with c2:
                if st.button("🤖 تحقق AI جماعي", key="rev_bulk_ai"):
                    with st.spinner("..."):
                        items = [{"our": str(r.get("المنتج", "")), "comp": str(r.get("اسم المنافس", "")),
                                  "our_price": safe_float(r.get("السعر", 0)), "comp_price": safe_float(r.get("أقل سعر منافس", 0))}
                                 for _, r in filtered.head(20).iterrows()]
                        result = bulk_verify(items, "review")
                        if result["success"]:
                            st.markdown(result["response"])
            with c3:
                if st.button("📤 تصدير Make", key="rev_make"):
                    products = export_to_make_format(filtered, "review")
                    result = send_to_make(products, "update")
                    st.success(result["message"]) if result["success"] else st.error(result["message"])

            # عرض المنتجات مع أزرار قرار
            for idx, row in filtered.iterrows():
                our_name = str(row.get("المنتج", ""))
                comp_name = str(row.get("اسم المنافس", ""))
                our_price = safe_float(row.get("السعر", 0))
                comp_price = safe_float(row.get("أقل سعر منافس", 0))
                match_pct = safe_float(row.get("نسبة التطابق", 0))

                match_color = "#ff8800" if match_pct >= 50 else "#ff4444"

                with st.container():
                    st.markdown(f"""<div style="border:1px solid #ff8800;border-radius:8px;padding:10px;margin:5px 0;background:#2a1a0e;">
                    <div style="display:flex;justify-content:space-between;">
                        <div><span style="color:#aaa;">منتجنا:</span> <strong style="color:#fff;">{our_name}</strong> ({our_price:,.2f} ر.س)</div>
                        <div style="color:{match_color};font-weight:bold;">{match_pct:.0f}%</div>
                        <div><span style="color:#aaa;">المنافس:</span> <strong style="color:#ff9800;">{comp_name}</strong> ({comp_price:,.2f} ر.س)</div>
                    </div></div>""", unsafe_allow_html=True)

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        if st.button("🤖 تحقق AI", key=f"ai_rev_{idx}"):
                            with st.spinner("..."):
                                r = verify_match(our_name, comp_name, our_price, comp_price)
                                if r["success"]:
                                    st.info(f"{'✅' if r.get('match') else '❌'} ثقة: {r.get('confidence', 0)}% - {r.get('reason', '')}")
                    with c2:
                        if st.button("✅ نقل لموافق", key=f"app_rev_{idx}"):
                            log_decision(our_name, "review", "approved")
                            st.success("تم النقل")
                    with c3:
                        if st.button("📉 نقل لمخفض", key=f"low_rev_{idx}"):
                            log_decision(our_name, "review", "price_lower")
                            st.success("تم النقل")
                    with c4:
                        if st.button("📤 Make", key=f"mk_rev_{idx}"):
                            r = send_single_product({"name": our_name, "price": our_price, "comp_name": comp_name, "comp_price": comp_price})
                            st.success(r["message"]) if r["success"] else st.error(r["message"])
                    with c5:
                        if st.button("🗑️ إزالة", key=f"rm_rev_{idx}"):
                            log_decision(our_name, "review", "removed")
                            st.warning("تم الإزالة")
        else:
            st.success("✅ لا توجد منتجات تحتاج مراجعة")
    else:
        st.info("ارفع الملفات أولاً")


# ===== 8. المقارنة البصرية =====
elif page == "📊 مقارنة بصرية":
    st.header("📊 المقارنة البصرية")
    db_log("visual", "view")

    if st.session_state.results:
        r = st.session_state.results
        tab1, tab2, tab3 = st.tabs(["📊 ملخص الأسعار", "📈 توزيع التطابق", "🏷️ حسب الماركة"])

        with tab1:
            data = {"القسم": ["أعلى سعراً", "أقل سعراً", "موافق", "مفقودة", "مراجعة"],
                    "العدد": [len(r.get("price_raise", pd.DataFrame())), len(r.get("price_lower", pd.DataFrame())),
                              len(r.get("approved", pd.DataFrame())), len(r.get("missing", pd.DataFrame())),
                              len(r.get("review", pd.DataFrame()))]}
            st.bar_chart(pd.DataFrame(data).set_index("القسم"))

        with tab2:
            all_matched = pd.DataFrame()
            for key in ["price_raise", "price_lower", "approved", "review"]:
                if key in r and not r[key].empty:
                    all_matched = pd.concat([all_matched, r[key]])
            if not all_matched.empty and "نسبة التطابق" in all_matched.columns:
                st.bar_chart(all_matched["نسبة التطابق"].value_counts().sort_index())

        with tab3:
            if not all_matched.empty and "الماركة" in all_matched.columns:
                brand_counts = all_matched["الماركة"].value_counts().head(15)
                st.bar_chart(brand_counts)
    else:
        st.info("ارفع الملفات أولاً")


# ===== 9. الذكاء الصناعي =====
elif page == "🤖 الذكاء الصناعي":
    st.header("🤖 مساعد الذكاء الصناعي")
    db_log("ai", "view")

    tab1, tab2, tab3 = st.tabs(["💬 دردشة", "🔍 تحقق منتج", "📊 تحليل"])

    with tab1:
        st.markdown("**اسأل أي سؤال عن التسعير والمنافسة:**")
        for h in st.session_state.chat_history[-10:]:
            st.markdown(f"**أنت:** {h['user']}")
            st.markdown(f"**AI ({h.get('source', '')}):** {h['ai']}")
            st.markdown("---")

        user_msg = st.text_input("💬 اكتب رسالتك:", key="chat_input")
        if user_msg and st.button("إرسال", key="chat_send"):
            with st.spinner("🤖 جاري الرد..."):
                result = chat_with_ai(user_msg, st.session_state.chat_history)
                if result["success"]:
                    st.session_state.chat_history.append({"user": user_msg, "ai": result["response"], "source": result["source"]})
                    st.rerun()
                else:
                    st.error(result["response"])

    with tab2:
        st.markdown("**تحقق من تطابق منتجين:**")
        c1, c2 = st.columns(2)
        p1 = c1.text_input("منتجنا:", key="v_our")
        p2 = c2.text_input("المنافس:", key="v_comp")
        c3, c4 = st.columns(2)
        pr1 = c3.number_input("سعرنا:", 0.0, key="v_pr1")
        pr2 = c4.number_input("سعر المنافس:", 0.0, key="v_pr2")

        if st.button("🔍 تحقق", key="verify_btn"):
            if p1 and p2:
                with st.spinner("..."):
                    r = verify_match(p1, p2, pr1, pr2)
                    if r["success"]:
                        col = "🟢" if r.get("match") else "🔴"
                        st.markdown(f"{col} **التطابق:** {'نعم' if r.get('match') else 'لا'}")
                        st.markdown(f"**الثقة:** {r.get('confidence', 0)}%")
                        st.markdown(f"**السبب:** {r.get('reason', '')}")

    with tab3:
        product = st.text_input("اسم المنتج:", key="analyze_name")
        price = st.number_input("السعر:", 0.0, key="analyze_price")
        if st.button("📊 تحليل", key="analyze_btn"):
            if product:
                with st.spinner("..."):
                    r = analyze_product(product, price)
                    if r["success"]:
                        st.markdown(r["response"])


# ===== 10. أتمتة Make =====
elif page == "⚡ أتمتة Make":
    st.header("⚡ أتمتة Make.com")
    db_log("make", "view")

    tab1, tab2, tab3 = st.tabs(["🔗 حالة الاتصال", "📤 إرسال يدوي", "📜 السجل"])

    with tab1:
        if st.button("🔍 فحص الاتصال"):
            with st.spinner("..."):
                results = verify_webhook_connection()
                for name, r in results.items():
                    if name != "all_connected":
                        st.markdown(f"**{name}:** {r['message']}")
                if results["all_connected"]:
                    st.success("✅ جميع الاتصالات تعمل")
                else:
                    st.error("❌ بعض الاتصالات لا تعمل")

    with tab2:
        st.markdown("**إرسال بيانات يدوياً:**")
        wh_type = st.selectbox("نوع الإرسال", ["تحديث أسعار", "منتجات جديدة", "منتجات مفقودة"])

        if st.session_state.results:
            section_map = {"تحديث أسعار": "price_raise", "منتجات جديدة": "price_lower", "منتجات مفقودة": "missing"}
            key = section_map.get(wh_type, "price_raise")
            if key in st.session_state.results and not st.session_state.results[key].empty:
                df = st.session_state.results[key]
                st.info(f"سيتم إرسال {len(df)} منتج")
                if st.button("📤 إرسال الآن"):
                    products = export_to_make_format(df, key)
                    func = {"تحديث أسعار": send_price_updates, "منتجات جديدة": send_new_products, "منتجات مفقودة": send_missing_products}
                    result = func.get(wh_type, send_price_updates)(products)
                    st.success(result["message"]) if result["success"] else st.error(result["message"])

    with tab3:
        events = get_events("make", 20)
        if events:
            for e in events:
                st.text(f"[{e['timestamp']}] {e['event_type']}: {e['details']}")
        else:
            st.info("لا يوجد سجل بعد")


# ===== 11. الإعدادات =====
elif page == "⚙️ الإعدادات":
    st.header("⚙️ الإعدادات")
    db_log("settings", "view")

    tab1, tab2, tab3 = st.tabs(["🔑 المفاتيح", "⚙️ المطابقة", "📜 السجل"])

    with tab1:
        st.markdown("**مفاتيح API:**")
        st.text_input("Gemini API Key", value=GEMINI_API_KEY[:20] + "...", disabled=True)
        st.text_input("OpenRouter API Key", value=OPENROUTER_API_KEY[:20] + "...", disabled=True)
        st.markdown("**Webhooks:**")
        st.text_input("Webhook تحديث الأسعار", value=WEBHOOK_UPDATE_PRICES, disabled=True)
        st.text_input("Webhook منتجات جديدة", value=WEBHOOK_NEW_PRODUCTS, disabled=True)

        if st.button("🔍 اختبار AI"):
            with st.spinner("..."):
                r = call_ai("مرحباً، اختبار اتصال")
                if r["success"]:
                    st.success(f"✅ AI يعمل ({r['source']}): {r['response'][:100]}")
                else:
                    st.error(f"❌ {r['response']}")

    with tab2:
        st.markdown("**إعدادات المطابقة:**")
        st.number_input("حد التطابق الأدنى %", value=MIN_MATCH_SCORE, disabled=True)
        st.number_input("حد التطابق العالي %", value=HIGH_MATCH_SCORE, disabled=True)
        st.number_input("حد فرق السعر (ر.س)", value=PRICE_DIFF_THRESHOLD, disabled=True)

    with tab3:
        decisions = get_decisions(limit=30)
        if decisions:
            for d in decisions:
                st.text(f"[{d['timestamp']}] {d['product_name']}: {d['old_status']} → {d['new_status']} ({d.get('reason', '')})")
        else:
            st.info("لا توجد قرارات مسجلة")


# ===== 12. سجل التحليلات =====
elif page == "📜 السجل":
    st.header("📜 سجل التحليلات والأحداث")
    db_log("log", "view")

    tab1, tab2 = st.tabs(["📊 تحليلات سابقة", "📝 كل الأحداث"])

    with tab1:
        from utils.db_manager import get_analysis_history
        history = get_analysis_history(20)
        if history:
            for h in history:
                st.markdown(f"**[{h['timestamp']}]** {h['our_file']} vs {h['comp_file']} → {h['matched']} متطابق | {h['missing']} مفقود")
        else:
            st.info("لا يوجد تاريخ")

    with tab2:
        events = get_events(limit=50)
        if events:
            df_events = pd.DataFrame(events)
            st.dataframe(df_events, use_container_width=True)
        else:
            st.info("لا توجد أحداث")
