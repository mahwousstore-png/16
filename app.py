"""
نظام التسعير الذكي - مهووس v17.2
- دعم CSV + Excel
- مطابقة بصرية في كل الأقسام
- AI يعمل فعلياً (Gemini + OpenRouter)
- أزرار ذكية + Make + تصدير
"""
import streamlit as st
import pandas as pd
from config import *
from styles import get_styles, stat_card, vs_card
from engines.engine import (read_file, run_full_analysis, find_missing_products,
                            export_excel, export_section_excel, is_sample,
                            extract_brand, extract_size, extract_type)
from engines.ai_engine import (call_ai, chat_with_ai, verify_match, analyze_product,
                               bulk_verify, suggest_price, process_paste, check_duplicate)
from utils.helpers import (apply_filters, get_filter_options, export_to_excel,
                           export_multiple_sheets, parse_pasted_text, safe_float,
                           format_price, format_diff, BackgroundTask)
from utils.make_helper import (send_price_updates, send_new_products, send_missing_products,
                               send_to_make, send_single_product, verify_webhook_connection,
                               export_to_make_format, test_webhook)
from utils.db_manager import (init_db, log_event, log_decision, log_analysis,
                              get_events, get_decisions, get_analysis_history)

# ===== إعداد الصفحة =====
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")
st.markdown(get_styles(), unsafe_allow_html=True)
init_db()

# ===== Session State =====
for key in ["results", "missing_df", "analysis_df", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else None

def db_log(page, action, details=""):
    try: log_event(page, action, details)
    except: pass


# ===== الشريط الجانبي =====
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.caption(f"الإصدار {APP_VERSION}")
    page = st.radio("الأقسام", SECTIONS, label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.results is not None:
        r = st.session_state.results
        st.markdown("**📊 ملخص سريع:**")
        st.caption(f"🔴 أعلى: {len(r.get('price_raise', pd.DataFrame()))}")
        st.caption(f"🟢 أقل: {len(r.get('price_lower', pd.DataFrame()))}")
        st.caption(f"✅ موافق: {len(r.get('approved', pd.DataFrame()))}")
        st.caption(f"🔍 مفقود: {len(r.get('missing', pd.DataFrame()))}")
        st.caption(f"⚠️ مراجعة: {len(r.get('review', pd.DataFrame()))}")


# ===== دوال العرض المشتركة =====
def render_filters(df, prefix):
    """عرض فلاتر متقدمة"""
    opts = get_filter_options(df)
    filters = {}
    with st.expander("🔍 فلاتر متقدمة", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        filters["search"] = c1.text_input("🔎 بحث", key=f"{prefix}_search")
        filters["brand"] = c2.selectbox("الماركة", opts["brands"], key=f"{prefix}_brand")
        filters["competitor"] = c3.selectbox("المنافس", opts["competitors"], key=f"{prefix}_comp")
        filters["type"] = c4.selectbox("النوع", opts["types"], key=f"{prefix}_type")
        c5, c6, c7 = st.columns(3)
        filters["match_min"] = c5.slider("أقل تطابق %", 0, 100, 0, key=f"{prefix}_match")
        filters["price_min"] = c6.number_input("أقل سعر", 0.0, key=f"{prefix}_pmin")
        filters["price_max"] = c7.number_input("أعلى سعر", 0.0, key=f"{prefix}_pmax")
        if filters["price_max"] == 0: filters["price_max"] = None
        if filters["match_min"] == 0: filters["match_min"] = None
    return filters


def render_action_bar(df, prefix, section_type="update"):
    """أزرار عامة لكل قسم: تصدير Excel + تحقق AI جماعي + تصدير Make"""
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📥 تصدير Excel", key=f"{prefix}_excel"):
            excel = export_to_excel(df, prefix)
            st.download_button("⬇️ تحميل", excel, f"{prefix}.xlsx", key=f"{prefix}_dl")
    with c2:
        if st.button("🤖 تحقق AI جماعي", key=f"{prefix}_bulk_ai"):
            with st.spinner("جاري التحقق بالذكاء الصناعي..."):
                items = []
                for _, r in df.head(20).iterrows():
                    items.append({
                        "our": str(r.get("المنتج", "")),
                        "comp": str(r.get("منتج المنافس", r.get("اسم المنافس", ""))),
                        "our_price": safe_float(r.get("السعر", 0)),
                        "comp_price": safe_float(r.get("سعر المنافس", r.get("أقل سعر منافس", 0)))
                    })
                result = bulk_verify(items, prefix)
                if result["success"]:
                    st.markdown(f'<div class="ai-box">{result["response"]}</div>', unsafe_allow_html=True)
                else:
                    st.error(result["response"])
    with c3:
        if st.button("📤 تصدير Make", key=f"{prefix}_make"):
            products = export_to_make_format(df, section_type)
            result = send_to_make(products, section_type)
            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])


def render_paste_section(prefix):
    """خاصية لصق نتائج خارجية مع AI"""
    with st.expander("📋 لصق بيانات / أوامر AI", expanded=False):
        pasted = st.text_area("الصق هنا نتائج من Gemini أو أي مصدر:", key=f"{prefix}_paste", height=100)
        c1, c2 = st.columns(2)
        with c1:
            if pasted and st.button("📊 تحليل", key=f"{prefix}_parse"):
                df, msg = parse_pasted_text(pasted)
                if df is not None:
                    st.success(msg)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error(msg)
        with c2:
            if pasted and st.button("🤖 معالجة AI", key=f"{prefix}_ai_paste"):
                with st.spinner("جاري المعالجة..."):
                    result = process_paste(pasted, prefix)
                    if result["success"]:
                        st.markdown(f'<div class="ai-box">{result["response"]}</div>', unsafe_allow_html=True)
                    else:
                        st.error(result["response"])


def render_vs_table(df, prefix):
    """عرض جدول المقارنة البصرية لكل الأقسام"""
    for idx, row in df.iterrows():
        our_name = str(row.get("المنتج", ""))
        comp_name = str(row.get("منتج المنافس", row.get("اسم المنافس", "")))
        our_price = safe_float(row.get("السعر", 0))
        comp_price = safe_float(row.get("سعر المنافس", row.get("أقل سعر منافس", 0)))
        diff = safe_float(row.get("الفرق", our_price - comp_price))
        match_pct = safe_float(row.get("نسبة التطابق", 0))
        comp_source = str(row.get("المنافس", ""))
        brand = str(row.get("الماركة", ""))
        risk = str(row.get("الخطورة", ""))

        # بطاقة VS بصرية
        st.markdown(vs_card(our_name, our_price, comp_name, comp_price, diff, comp_source), unsafe_allow_html=True)

        # شريط التطابق + معلومات إضافية
        match_color = "#00C853" if match_pct >= 90 else "#FFD600" if match_pct >= 70 else "#FF1744"
        risk_badge = f'<span class="badge b-high">{risk}</span>' if risk == "عالي" else f'<span class="badge b-med">{risk}</span>' if risk == "متوسط" else f'<span class="badge b-low">{risk}</span>'

        st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 12px;font-size:.8rem;">
        <span>🏷️ {brand}</span>
        <span>تطابق: <span style="color:{match_color};font-weight:700">{match_pct:.0f}%</span></span>
        {risk_badge if risk else ""}
        </div>""", unsafe_allow_html=True)

        # عرض المنافسين المتعددين
        all_comps = row.get("جميع المنافسين", [])
        if isinstance(all_comps, list) and len(all_comps) > 1:
            with st.expander(f"👥 {len(all_comps)} منافسين", expanded=False):
                for cm in all_comps:
                    st.markdown(f'<div class="multi-comp">🏪 <strong>{cm.get("competitor", "")}</strong>: {cm.get("name", "")} - <span style="color:#ff9800">{cm.get("price", 0):,.0f} ر.س</span> ({cm.get("score", 0):.0f}%)</div>', unsafe_allow_html=True)

        # أزرار لكل منتج
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if st.button("🤖 تحقق AI", key=f"ai_{prefix}_{idx}"):
                with st.spinner("..."):
                    r = verify_match(our_name, comp_name, our_price, comp_price)
                    if r["success"]:
                        icon = "✅" if r.get("match") else "❌"
                        st.info(f"{icon} ثقة: {r.get('confidence', 0)}% - {r.get('reason', '')}")
                    else:
                        st.error("فشل الاتصال بـ AI")
        with c2:
            if st.button("✅ موافقة", key=f"ok_{prefix}_{idx}"):
                log_decision(our_name, prefix, "approved", "موافقة يدوية")
                st.success("✅ تم")
        with c3:
            if st.button("📤 Make", key=f"mk_{prefix}_{idx}"):
                r = send_single_product({"name": our_name, "price": our_price, "comp_name": comp_name, "comp_price": comp_price, "diff": diff})
                st.success(r["message"]) if r["success"] else st.error(r["message"])
        with c4:
            if st.button("⏸️ تأجيل", key=f"dly_{prefix}_{idx}"):
                log_decision(our_name, prefix, "deferred", "تأجيل")
                st.warning("تم التأجيل")
        with c5:
            if st.button("🗑️ إزالة", key=f"rm_{prefix}_{idx}"):
                log_decision(our_name, prefix, "removed", "إزالة")
                st.warning("تم الإزالة")

        st.markdown("---")


# ============================================================
# ===== 1. لوحة التحكم =====
# ============================================================
if page == "📊 لوحة التحكم":
    st.header("📊 لوحة التحكم")
    db_log("dashboard", "view")

    if st.session_state.results:
        r = st.session_state.results
        cols = st.columns(5)
        data = [
            ("🔴", "سعر أعلى", len(r.get("price_raise", pd.DataFrame())), COLORS["raise"]),
            ("🟢", "سعر أقل", len(r.get("price_lower", pd.DataFrame())), COLORS["lower"]),
            ("✅", "موافق", len(r.get("approved", pd.DataFrame())), COLORS["approved"]),
            ("🔍", "مفقود", len(r.get("missing", pd.DataFrame())), COLORS["missing"]),
            ("⚠️", "مراجعة", len(r.get("review", pd.DataFrame())), COLORS["review"]),
        ]
        for col, (icon, label, val, color) in zip(cols, data):
            col.markdown(stat_card(icon, label, val, color), unsafe_allow_html=True)

        st.markdown("---")

        # تصدير شامل
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📥 تصدير كل الأقسام Excel"):
                sheets = {}
                for key, name in [("price_raise", "سعر_أعلى"), ("price_lower", "سعر_أقل"),
                                  ("approved", "موافق"), ("missing", "مفقود"), ("review", "مراجعة")]:
                    if key in r and not r[key].empty:
                        df = r[key].copy()
                        if "جميع المنافسين" in df.columns:
                            df = df.drop(columns=["جميع المنافسين"])
                        sheets[name] = df
                if sheets:
                    excel = export_multiple_sheets(sheets)
                    st.download_button("⬇️ تحميل الملف الشامل", excel, "all_sections.xlsx")
        with c2:
            if st.button("📤 تصدير كل شيء لـ Make"):
                for key in ["price_raise", "price_lower"]:
                    if key in r and not r[key].empty:
                        products = export_to_make_format(r[key], "update")
                        send_price_updates(products)
                st.success("تم الإرسال!")
    else:
        st.info("👈 ارفع ملفات منتجاتك وملفات المنافسين من قسم 'رفع الملفات' للبدء")


# ============================================================
# ===== 2. رفع الملفات =====
# ============================================================
elif page == "📂 رفع الملفات":
    st.header("📂 رفع الملفات وتحليلها")
    db_log("upload", "view")

    st.markdown("**ارفع ملف منتجاتك وملفات المنافسين (CSV أو Excel)**")

    our_file = st.file_uploader("📦 ملف منتجاتنا", type=["csv", "xlsx", "xls"], key="our_file")
    comp_files = st.file_uploader("🏪 ملفات المنافسين", type=["csv", "xlsx", "xls"],
                                  accept_multiple_files=True, key="comp_files")

    bg_mode = st.checkbox("⚡ معالجة في الخلفية", value=False)

    if st.button("🚀 بدء التحليل", type="primary"):
        if our_file and comp_files:
            our_df, err = read_file(our_file)
            if err:
                st.error(f"❌ خطأ في ملف منتجاتنا: {err}")
            else:
                comp_dfs = {}
                for cf in comp_files:
                    cdf, cerr = read_file(cf)
                    if cerr:
                        st.warning(f"⚠️ خطأ في {cf.name}: {cerr}")
                    else:
                        comp_dfs[cf.name] = cdf

                if comp_dfs:
                    try:
                        progress = st.progress(0, "جاري التحليل...")
                        def update_progress(pct):
                            progress.progress(pct, f"جاري التحليل... {pct*100:.0f}%")

                        analysis_df = run_full_analysis(our_df, comp_dfs, progress_callback=update_progress)
                        missing_df = find_missing_products(our_df, comp_dfs)

                        # تصنيف النتائج
                        results = {
                            "price_raise": analysis_df[analysis_df["القرار"].str.contains("أعلى", na=False)].reset_index(drop=True),
                            "price_lower": analysis_df[analysis_df["القرار"].str.contains("أقل", na=False)].reset_index(drop=True),
                            "approved": analysis_df[analysis_df["القرار"].str.contains("موافق", na=False)].reset_index(drop=True),
                            "review": analysis_df[analysis_df["القرار"].str.contains("مراجعة", na=False)].reset_index(drop=True),
                            "missing": missing_df,
                            "all": analysis_df,
                        }

                        st.session_state.results = results
                        st.session_state.analysis_df = analysis_df
                        st.session_state.missing_df = missing_df

                        total_our = len(our_df)
                        matched = len(analysis_df[analysis_df["نسبة التطابق"] > 0])
                        missing_count = len(missing_df)

                        log_analysis(our_file.name, ",".join([f.name for f in comp_files]),
                                     total_our, matched, missing_count)

                        progress.progress(1.0, "✅ اكتمل التحليل!")
                        st.success(f"✅ تم! {matched} متطابق | {missing_count} مفقود | {len(results['review'])} مراجعة")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ خطأ: {str(e)}")
                else:
                    st.error("❌ لم يتم قراءة أي ملف منافس بنجاح")
        else:
            st.warning("⚠️ ارفع ملف منتجاتنا وملف منافس واحد على الأقل")


# ============================================================
# ===== 3. سعر أعلى =====
# ============================================================
elif page == "🔴 سعر أعلى":
    st.header("🔴 منتجات سعرنا أعلى من المنافسين")
    db_log("price_raise", "view")

    if st.session_state.results and "price_raise" in st.session_state.results:
        df = st.session_state.results["price_raise"]
        if not df.empty:
            st.error(f"⚠️ {len(df)} منتج سعرنا أعلى - يحتاج مراجعة فورية")
            filters = render_filters(df, "raise")
            filtered = apply_filters(df, filters)
            render_action_bar(filtered, "raise", "update")
            render_paste_section("raise")
            st.markdown(f"**عرض {len(filtered)} من {len(df)} منتج**")
            render_vs_table(filtered, "raise")
        else:
            st.success("✅ لا توجد منتجات بسعر أعلى")
    else:
        st.info("ارفع الملفات أولاً من قسم 'رفع الملفات'")


# ============================================================
# ===== 4. سعر أقل =====
# ============================================================
elif page == "🟢 سعر أقل":
    st.header("🟢 منتجات سعرنا أقل من المنافسين")
    db_log("price_lower", "view")

    if st.session_state.results and "price_lower" in st.session_state.results:
        df = st.session_state.results["price_lower"]
        if not df.empty:
            st.info(f"💰 {len(df)} منتج سعرنا أقل - فرصة لرفع السعر")
            filters = render_filters(df, "lower")
            filtered = apply_filters(df, filters)
            render_action_bar(filtered, "lower", "update")
            render_paste_section("lower")
            st.markdown(f"**عرض {len(filtered)} من {len(df)} منتج**")
            render_vs_table(filtered, "lower")
        else:
            st.info("لا توجد منتجات بسعر أقل")
    else:
        st.info("ارفع الملفات أولاً")


# ============================================================
# ===== 5. موافق عليها =====
# ============================================================
elif page == "✅ موافق عليها":
    st.header("✅ المنتجات الموافق عليها")
    db_log("approved", "view")

    if st.session_state.results and "approved" in st.session_state.results:
        df = st.session_state.results["approved"]
        if not df.empty:
            st.success(f"✅ {len(df)} منتج موافق عليه - الأسعار مناسبة")
            filters = render_filters(df, "approved")
            filtered = apply_filters(df, filters)
            render_action_bar(filtered, "approved", "update")
            render_paste_section("approved")
            st.markdown(f"**عرض {len(filtered)} من {len(df)} منتج**")
            render_vs_table(filtered, "approved")
        else:
            st.info("لا توجد منتجات موافق عليها")
    else:
        st.info("ارفع الملفات أولاً")


# ============================================================
# ===== 6. منتجات مفقودة =====
# ============================================================
elif page == "🔍 منتجات مفقودة":
    st.header("🔍 منتجات المنافسين غير الموجودة عندنا")
    db_log("missing", "view")

    if st.session_state.results and "missing" in st.session_state.results:
        df = st.session_state.results["missing"]
        if not df.empty:
            st.warning(f"⚠️ {len(df)} منتج مفقود - تحقق بدقة قبل الإضافة لتجنب التكرار!")

            # فلاتر
            opts = get_filter_options(df)
            with st.expander("🔍 فلاتر", expanded=False):
                c1, c2, c3 = st.columns(3)
                search = c1.text_input("🔎 بحث", key="miss_search")
                brand_f = c2.selectbox("الماركة", opts["brands"], key="miss_brand")
                comp_f = c3.selectbox("المنافس", opts["competitors"], key="miss_comp")

            filtered = df.copy()
            if search:
                filtered = filtered[filtered.apply(lambda r: search.lower() in str(r.values).lower(), axis=1)]
            if brand_f != "الكل" and "الماركة" in filtered.columns:
                filtered = filtered[filtered["الماركة"].str.contains(brand_f, case=False, na=False)]
            if comp_f != "الكل" and "المنافس" in filtered.columns:
                filtered = filtered[filtered["المنافس"].str.contains(comp_f, case=False, na=False)]

            # أزرار عامة
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("📥 تصدير Excel", key="miss_excel"):
                    excel = export_to_excel(filtered, "مفقودة")
                    st.download_button("⬇️ تحميل", excel, "missing.xlsx", key="miss_dl")
            with c2:
                if st.button("🤖 تحقق AI جماعي", key="miss_bulk_ai"):
                    with st.spinner("جاري التحقق من التكرار بالذكاء الصناعي..."):
                        items = [{"our": "", "comp": str(r.get("منتج المنافس", "")),
                                  "our_price": 0, "comp_price": safe_float(r.get("سعر المنافس", 0))}
                                 for _, r in filtered.head(20).iterrows()]
                        result = bulk_verify(items, "missing")
                        if result["success"]:
                            st.markdown(f'<div class="ai-box">{result["response"]}</div>', unsafe_allow_html=True)
                        else:
                            st.error(result["response"])
            with c3:
                if st.button("📤 تصدير Make", key="miss_make"):
                    products = [{"name": str(r.get("منتج المنافس", "")),
                                 "price": safe_float(r.get("سعر المنافس", 0)),
                                 "brand": str(r.get("الماركة", "")),
                                 "competitor": str(r.get("المنافس", ""))}
                                for _, r in filtered.iterrows()]
                    result = send_missing_products(products)
                    st.success(result["message"]) if result["success"] else st.error(result["message"])

            render_paste_section("missing")

            st.markdown(f"**عرض {len(filtered)} من {len(df)} منتج مفقود**")

            # عرض المنتجات المفقودة
            for idx, row in filtered.iterrows():
                name = str(row.get("منتج المنافس", ""))
                price = safe_float(row.get("سعر المنافس", 0))
                brand = str(row.get("الماركة", ""))
                comp = str(row.get("المنافس", ""))
                size = row.get("الحجم", "")
                ptype = row.get("النوع", "")

                st.markdown(f"""<div style="border:1px solid #007bff;border-radius:8px;padding:12px;margin:5px 0;background:#0a1628;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div><strong style="color:#4fc3f7;font-size:1rem;">{name}</strong></div>
                    <div><span style="color:#ff9800;font-weight:700;font-size:1.1rem;">{price:,.0f} ر.س</span></div>
                </div>
                <div style="font-size:.8rem;color:#888;margin-top:4px;">🏷️ {brand} | 📏 {size} | 🏪 {comp} | 🧴 {ptype}</div>
                </div>""", unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if st.button("🤖 تحقق تكرار", key=f"dup_{idx}"):
                        with st.spinner("..."):
                            our_products = []
                            if st.session_state.analysis_df is not None:
                                our_products = st.session_state.analysis_df.get("المنتج", pd.Series()).tolist()
                            r = check_duplicate(name, our_products[:50])
                            if r["success"]:
                                st.markdown(f'<div class="ai-box">{r["response"]}</div>', unsafe_allow_html=True)
                            else:
                                st.error("فشل الاتصال بـ AI")
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
                st.markdown("---")
        else:
            st.success("✅ لا توجد منتجات مفقودة")
    else:
        st.info("ارفع الملفات أولاً")


# ============================================================
# ===== 7. تحت المراجعة =====
# ============================================================
elif page == "⚠️ تحت المراجعة":
    st.header("⚠️ منتجات تحتاج مراجعة")
    db_log("review", "view")

    if st.session_state.results and "review" in st.session_state.results:
        df = st.session_state.results["review"]
        if not df.empty:
            st.warning(f"⚠️ {len(df)} منتج يحتاج مراجعة - تطابق غير مؤكد")
            filters = render_filters(df, "review")
            filtered = apply_filters(df, filters)
            render_action_bar(filtered, "review", "update")
            render_paste_section("review")

            st.markdown(f"**عرض {len(filtered)} من {len(df)} منتج**")

            # عرض المنتجات مع أزرار قرار خاصة بالمراجعة
            for idx, row in filtered.iterrows():
                our_name = str(row.get("المنتج", ""))
                comp_name = str(row.get("منتج المنافس", ""))
                our_price = safe_float(row.get("السعر", 0))
                comp_price = safe_float(row.get("سعر المنافس", 0))
                diff = safe_float(row.get("الفرق", 0))
                match_pct = safe_float(row.get("نسبة التطابق", 0))
                comp_source = str(row.get("المنافس", ""))

                # بطاقة VS بصرية
                st.markdown(vs_card(our_name, our_price, comp_name, comp_price, diff, comp_source), unsafe_allow_html=True)

                match_color = "#FFD600" if match_pct >= 70 else "#FF1744"
                st.markdown(f'<div style="text-align:center;font-size:.85rem;">تطابق: <span style="color:{match_color};font-weight:700">{match_pct:.0f}%</span></div>', unsafe_allow_html=True)

                # أزرار قرار المراجعة
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    if st.button("🤖 تحقق AI", key=f"ai_rev_{idx}"):
                        with st.spinner("..."):
                            r = verify_match(our_name, comp_name, our_price, comp_price)
                            if r["success"]:
                                icon = "✅" if r.get("match") else "❌"
                                st.info(f"{icon} ثقة: {r.get('confidence', 0)}% - {r.get('reason', '')}")
                            else:
                                st.error("فشل الاتصال بـ AI")
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
                st.markdown("---")
        else:
            st.success("✅ لا توجد منتجات تحتاج مراجعة")
    else:
        st.info("ارفع الملفات أولاً")


# ============================================================
# ===== 8. المقارنة البصرية =====
# ============================================================
elif page == "📊 مقارنة بصرية":
    st.header("📊 المقارنة البصرية الشاملة")
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


# ============================================================
# ===== 9. الذكاء الصناعي =====
# ============================================================
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
                    else:
                        st.error("فشل الاتصال بـ AI")

    with tab3:
        product = st.text_input("اسم المنتج:", key="analyze_name")
        price = st.number_input("السعر:", 0.0, key="analyze_price")
        if st.button("📊 تحليل", key="analyze_btn"):
            if product:
                with st.spinner("..."):
                    r = analyze_product(product, price)
                    if r["success"]:
                        st.markdown(f'<div class="ai-box">{r["response"]}</div>', unsafe_allow_html=True)
                    else:
                        st.error(r["response"])


# ============================================================
# ===== 10. أتمتة Make =====
# ============================================================
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


# ============================================================
# ===== 11. الإعدادات =====
# ============================================================
elif page == "⚙️ الإعدادات":
    st.header("⚙️ الإعدادات")
    db_log("settings", "view")

    tab1, tab2, tab3 = st.tabs(["🔑 المفاتيح", "⚙️ المطابقة", "📜 السجل"])

    with tab1:
        st.markdown("**مفاتيح API (محمية):**")
        gemini_status = f"✅ {len(GEMINI_API_KEYS)} مفاتيح مفعلة" if GEMINI_API_KEYS else "❌ غير مفعل"
        openrouter_status = "✅ مفعل" if OPENROUTER_API_KEY else "❌ غير مفعل"
        st.info(f"Gemini: {gemini_status}")
        st.info(f"OpenRouter: {openrouter_status}")

        st.markdown("**Webhooks:**")
        st.info(f"تحديث أسعار: {'✅ مربوط' if WEBHOOK_UPDATE_PRICES else '❌'}")
        st.info(f"منتجات جديدة: {'✅ مربوط' if WEBHOOK_NEW_PRODUCTS else '❌'}")

        if st.button("🔍 اختبار AI"):
            with st.spinner("جاري الاختبار..."):
                r = call_ai("مرحباً، اختبار اتصال. أجب بكلمة واحدة: يعمل")
                if r["success"]:
                    st.success(f"✅ AI يعمل ({r['source']}): {r['response'][:100]}")
                else:
                    st.error(f"❌ {r['response']}")

    with tab2:
        st.markdown("**إعدادات المطابقة:**")
        st.info(f"حد التطابق الأدنى: {MIN_MATCH_SCORE}%")
        st.info(f"حد التطابق العالي: {HIGH_MATCH_SCORE}%")
        st.info(f"حد فرق السعر: {PRICE_DIFF_THRESHOLD} ر.س")

    with tab3:
        decisions = get_decisions(limit=30)
        if decisions:
            for d in decisions:
                st.text(f"[{d['timestamp']}] {d['product_name']}: {d['old_status']} → {d['new_status']} ({d.get('reason', '')})")
        else:
            st.info("لا توجد قرارات مسجلة")


# ============================================================
# ===== 12. السجل =====
# ============================================================
elif page == "📜 السجل":
    st.header("📜 سجل التحليلات والأحداث")
    db_log("log", "view")

    tab1, tab2 = st.tabs(["📊 تحليلات سابقة", "📝 كل الأحداث"])

    with tab1:
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
