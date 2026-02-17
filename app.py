"""
نظام التسعير الذكي - مهووس v17.2
- دعم CSV + Excel
- واجهة محدثة لتدعم العمليات الطويلة (State Management)
- تكامل كامل مع المحرك المتجهي السريع
"""
import streamlit as st
import pandas as pd
import time
from config import *
from styles import get_styles, stat_card, vs_card

# استيراد المحركات (تأكد من وجود ملفات __init__.py)
from engines.engine import (read_file, run_full_analysis, find_missing_products,
                            export_excel, export_section_excel, is_sample,
                            extract_brand, extract_size, extract_type)
from engines.ai_engine import (call_ai, chat_with_ai, verify_match, analyze_product,
                               bulk_verify, suggest_price, process_paste, check_duplicate)

# استيراد الأدوات
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

# ===== إدارة الذاكرة (Session State) =====
if "results" not in st.session_state:
    st.session_state.results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False

def db_log(page, action, details=""):
    try: log_event(page, action, details)
    except: pass


# ===== الشريط الجانبي =====
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.caption(f"الإصدار {APP_VERSION}")
    page = st.radio("الأقسام", SECTIONS, label_visibility="collapsed")
    st.markdown("---")
    
    # ملخص سريع في السايدبار
    if st.session_state.results is not None:
        r = st.session_state.results
        st.markdown("**📊 حالة التحليل:**")
        st.caption(f"🔴 أعلى: {len(r.get('price_raise', pd.DataFrame()))}")
        st.caption(f"🟢 أقل: {len(r.get('price_lower', pd.DataFrame()))}")
        st.caption(f"✅ موافق: {len(r.get('approved', pd.DataFrame()))}")
        st.caption(f"⚠️ مراجعة: {len(r.get('review', pd.DataFrame()))}")
        st.caption(f"🔍 مفقود: {len(r.get('missing', pd.DataFrame()))}")


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
    """أزرار عامة لكل قسم"""
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
    """عرض جدول المقارنة البصرية"""
    display_limit = 50 
    
    for idx, row in df.head(display_limit).iterrows():
        our_name = str(row.get("المنتج", ""))
        comp_name = str(row.get("منتج المنافس", row.get("اسم المنافس", "")))
        our_price = safe_float(row.get("السعر", 0))
        comp_price = safe_float(row.get("سعر المنافس", row.get("أقل سعر منافس", 0)))
        diff = safe_float(row.get("الفرق", our_price - comp_price))
        match_pct = safe_float(row.get("نسبة التطابق", 0))
        comp_source = str(row.get("المنافس", ""))
        brand = str(row.get("الماركة", ""))
        risk = str(row.get("الخطورة", ""))

        st.markdown(vs_card(our_name, our_price, comp_name, comp_price, diff, comp_source), unsafe_allow_html=True)

        match_color = "#00C853" if match_pct >= 90 else "#FFD600" if match_pct >= 70 else "#FF1744"
        risk_badge = f'<span class="badge b-high">{risk}</span>' if risk == "عالي" else f'<span class="badge b-med">{risk}</span>' if risk == "متوسط" else f'<span class="badge b-low">{risk}</span>'

        st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 12px;font-size:.8rem;">
        <span>🏷️ {brand}</span>
        <span>تطابق: <span style="color:{match_color};font-weight:700">{match_pct:.0f}%</span></span>
        {risk_badge if risk else ""}
        </div>""", unsafe_allow_html=True)

        all_comps = row.get("جميع المنافسين", [])
        if isinstance(all_comps, list) and len(all_comps) > 1:
            with st.expander(f"👥 {len(all_comps)} منافسين", expanded=False):
                for cm in all_comps:
                    st.markdown(f'<div class="multi-comp">🏪 <strong>{cm.get("competitor", "")}</strong>: {cm.get("name", "")} - <span style="color:#ff9800">{cm.get("price", 0):,.0f} ر.س</span> ({cm.get("score", 0):.0f}%)</div>', unsafe_allow_html=True)

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
    
    if len(df) > display_limit:
        st.info(f"تم عرض {display_limit} منتج فقط لتسريع الصفحة. حمل ملف Excel لرؤية كافة النتائج ({len(df)} منتج).")


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
        st.info("👈 ابدأ من قسم 'رفع الملفات' للبدء")


# ============================================================
# ===== 2. رفع الملفات (محدث للمعالجة المحسنة) =====
# ============================================================
elif page == "📂 رفع الملفات":
    st.header("📂 رفع الملفات والتحليل")
    db_log("upload", "view")

    st.markdown("**ارفع ملف منتجاتك وملفات المنافسين (CSV أو Excel)**")

    our_file = st.file_uploader("📦 ملف منتجاتنا", type=["csv", "xlsx", "xls"], key="our_file")
    comp_files = st.file_uploader("🏪 ملفات المنافسين", type=["csv", "xlsx", "xls"],
                                  accept_multiple_files=True, key="comp_files")

    # زر بدء التحليل مع إدارة الحالة
    if st.button("🚀 بدء التحليل", type="primary") or st.session_state.analysis_running:
        
        if not (our_file and comp_files):
            st.warning("⚠️ يرجى رفع ملف منتجاتنا وملف منافس واحد على الأقل.")
        else:
            st.session_state.analysis_running = True
            
            # حاوية الحالة للعمليات الطويلة
            with st.status("جاري معالجة البيانات...", expanded=True) as status:
                try:
                    # 1. القراءة
                    status.write("📂 جاري قراءة الملفات...")
                    our_df, err = read_file(our_file)
                    
                    if err:
                        status.update(label="❌ خطأ في الملف", state="error")
                        st.error(f"خطأ: {err}")
                        st.session_state.analysis_running = False
                    else:
                        comp_dfs = {}
                        for cf in comp_files:
                            cdf, cerr = read_file(cf)
                            if not cerr:
                                comp_dfs[cf.name] = cdf
                        
                        if not comp_dfs:
                            status.update(label="❌ فشل قراءة المنافسين", state="error")
                            st.error("لم يتم قراءة أي ملف منافس بنجاح")
                            st.session_state.analysis_running = False
                        else:
                            # 2. التحليل (باستخدام المحرك السريع)
                            status.write(f"⚡ تشغيل المحرك الذكي على {len(our_df)} منتج...")
                            
                            # شريط تقدم داخلي
                            progress_bar = st.progress(0)
                            def update_prog(p): progress_bar.progress(p)
                            
                            analysis_df = run_full_analysis(our_df, comp_dfs, progress_callback=update_prog)
                            
                            status.write("🔍 البحث عن المنتجات المفقودة...")
                            missing_df = find_missing_products(our_df, comp_dfs)

                            # 3. التصنيف وحفظ النتائج
                            status.write("📊 تصنيف النتائج...")
                            results = {
                                "price_raise": analysis_df[analysis_df["القرار"].str.contains("أعلى", na=False)].reset_index(drop=True),
                                "price_lower": analysis_df[analysis_df["القرار"].str.contains("أقل", na=False)].reset_index(drop=True),
                                "approved": analysis_df[analysis_df["القرار"].str.contains("موافق", na=False)].reset_index(drop=True),
                                "review": analysis_df[analysis_df["القرار"].str.contains("مراجعة", na=False)].reset_index(drop=True),
                                "missing": missing_df,
                                "all": analysis_df,
                            }

                            st.session_state.results = results
                            
                            # تسجيل العملية
                            total_our = len(our_df)
                            matched = len(analysis_df[analysis_df["نسبة التطابق"] > 0])
                            missing_count = len(missing_df)
                            log_analysis(our_file.name, str(len(comp_files)), total_our, matched, missing_count)

                            status.update(label="✅ اكتمل التحليل!", state="complete", expanded=False)
                            st.session_state.analysis_running = False
                            
                            st.success(f"تم بنجاح! {matched} متطابق | {missing_count} مفقود")
                            st.balloons()
                            
                except Exception as e:
                    status.update(label="❌ حدث خطأ غير متوقع", state="error")
                    st.error(f"Error details: {str(e)}")
                    st.session_state.analysis_running = False


# ============================================================
# ===== 3. سعر أعلى =====
# ============================================================
elif page == "🔴 سعر أعلى":
    st.header("🔴 منتجات سعرنا أعلى من المنافسين")
    if st.session_state.results and "price_raise" in st.session_state.results:
        df = st.session_state.results["price_raise"]
        if not df.empty:
            filters = render_filters(df, "raise")
            filtered = apply_filters(df, filters)
            render_action_bar(filtered, "raise", "update")
            render_paste_section("raise")
            st.markdown(f"**عرض {len(filtered)} من {len(df)} منتج**")
            render_vs_table(filtered, "raise")
        else:
            st.success("✅ لا توجد منتجات بسعر أعلى")
    else:
        st.info("النتائج غير متوفرة بعد")


# ============================================================
# ===== 4. سعر أقل =====
# ============================================================
elif page == "🟢 سعر أقل":
    st.header("🟢 منتجات سعرنا أقل من المنافسين")
    if st.session_state.results and "price_lower" in st.session_state.results:
        df = st.session_state.results["price_lower"]
        if not df.empty:
            filters = render_filters(df, "lower")
            filtered = apply_filters(df, filters)
            render_action_bar(filtered, "lower", "update")
            render_paste_section("lower")
            st.markdown(f"**عرض {len(filtered)} من {len(df)} منتج**")
            render_vs_table(filtered, "lower")
        else:
            st.info("لا توجد منتجات بسعر أقل")
    else:
        st.info("النتائج غير متوفرة بعد")


# ============================================================
# ===== 5. موافق عليها =====
# ============================================================
elif page == "✅ موافق عليها":
    st.header("✅ المنتجات الموافق عليها")
    if st.session_state.results and "approved" in st.session_state.results:
        df = st.session_state.results["approved"]
        if not df.empty:
            filters = render_filters(df, "approved")
            filtered = apply_filters(df, filters)
            render_action_bar(filtered, "approved", "update")
            render_paste_section("approved")
            st.markdown(f"**عرض {len(filtered)} من {len(df)} منتج**")
            render_vs_table(filtered, "approved")
        else:
            st.info("لا توجد منتجات موافق عليها")
    else:
        st.info("النتائج غير متوفرة بعد")


# ============================================================
# ===== 6. منتجات مفقودة =====
# ============================================================
elif page == "🔍 منتجات مفقودة":
    st.header("🔍 منتجات المنافسين غير الموجودة عندنا")
    if st.session_state.results and "missing" in st.session_state.results:
        df = st.session_state.results["missing"]
        if not df.empty:
            st.warning(f"⚠️ {len(df)} منتج مفقود")

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

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("📥 تصدير Excel", key="miss_excel"):
                    excel = export_to_excel(filtered, "مفقودة")
                    st.download_button("⬇️ تحميل", excel, "missing.xlsx", key="miss_dl")
            with c2:
                if st.button("🤖 تحقق AI جماعي", key="miss_bulk_ai"):
                    with st.spinner("..."):
                        items = [{"our": "", "comp": str(r.get("منتج المنافس", "")),
                                  "our_price": 0, "comp_price": safe_float(r.get("سعر المنافس", 0))}
                                 for _, r in filtered.head(20).iterrows()]
                        result = bulk_verify(items, "missing")
                        if result["success"]:
                            st.markdown(f'<div class="ai-box">{result["response"]}</div>', unsafe_allow_html=True)
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
            st.dataframe(filtered, use_container_width=True)
        else:
            st.success("✅ لا توجد منتجات مفقودة")
    else:
        st.info("النتائج غير متوفرة بعد")


# ============================================================
# ===== 7. تحت المراجعة =====
# ============================================================
elif page == "⚠️ تحت المراجعة":
    st.header("⚠️ منتجات تحتاج مراجعة")
    if st.session_state.results and "review" in st.session_state.results:
        df = st.session_state.results["review"]
        if not df.empty:
            filters = render_filters(df, "review")
            filtered = apply_filters(df, filters)
            render_action_bar(filtered, "review", "update")
            render_paste_section("review")
            st.markdown(f"**عرض {len(filtered)} من {len(df)} منتج**")
            render_vs_table(filtered, "review")
        else:
            st.success("✅ لا توجد منتجات تحتاج مراجعة")
    else:
        st.info("النتائج غير متوفرة بعد")


# ============================================================
# ===== الصفحات الأخرى (بقية الأقسام كما هي) =====
# ============================================================
elif page == "📊 مقارنة بصرية":
    st.header("📊 المقارنة البصرية الشاملة")
    if st.session_state.results:
        r = st.session_state.results
        tab1, tab2 = st.tabs(["📊 ملخص الأسعار", "🏷️ حسب الماركة"])
        with tab1:
            data = {"القسم": ["أعلى سعراً", "أقل سعراً", "موافق", "مفقودة", "مراجعة"],
                    "العدد": [len(r.get("price_raise", pd.DataFrame())), len(r.get("price_lower", pd.DataFrame())),
                              len(r.get("approved", pd.DataFrame())), len(r.get("missing", pd.DataFrame())),
                              len(r.get("review", pd.DataFrame()))]}
            st.bar_chart(pd.DataFrame(data).set_index("القسم"))
        with tab2:
            all_matched = pd.concat([r.get("price_raise", pd.DataFrame()), r.get("price_lower", pd.DataFrame()), r.get("approved", pd.DataFrame())])
            if not all_matched.empty and "الماركة" in all_matched.columns:
                st.bar_chart(all_matched["الماركة"].value_counts().head(15))
    else:
        st.info("ارفع الملفات أولاً")

elif page == "🤖 الذكاء الصناعي":
    st.header("🤖 مساعد الذكاء الصناعي")
    tab1, tab2 = st.tabs(["💬 دردشة", "🔍 تحقق"])
    with tab1:
        for h in st.session_state.chat_history:
            st.chat_message("user").write(h['user'])
            st.chat_message("assistant").write(h['ai'])
        user_msg = st.chat_input("اكتب رسالتك...")
        if user_msg:
            st.chat_message("user").write(user_msg)
            with st.spinner("..."):
                resp = chat_with_ai(user_msg, st.session_state.chat_history)
                st.chat_message("assistant").write(resp["response"])
                st.session_state.chat_history.append({"user": user_msg, "ai": resp["response"]})
    with tab2:
        c1, c2 = st.columns(2)
        p1 = c1.text_input("منتجنا")
        p2 = c2.text_input("المنافس")
        if st.button("تحقق") and p1 and p2:
            with st.spinner("..."):
                r = verify_match(p1, p2)
                st.info(f"{'✅' if r['match'] else '❌'} {r['confidence']}% - {r['reason']}")

elif page == "⚡ أتمتة Make":
    st.header("⚡ أتمتة Make.com")
    if st.button("فحص الاتصال"):
        res = verify_webhook_connection()
        if res["all_connected"]: st.success("✅ متصل")
        else: st.error("❌ غير متصل")

elif page == "⚙️ الإعدادات":
    st.header("⚙️ الإعدادات")
    st.json({"APP_VERSION": APP_VERSION, "SECTIONS": SECTIONS})

elif page == "📜 السجل":
    st.header("📜 السجل")
    st.dataframe(pd.DataFrame(get_analysis_history(20)))
