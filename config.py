"""
نظام التسعير الذكي - مهووس v16.0
خفيف | سريع | AI مباشر | يعمل في الخلفية
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from styles import get_main_css, stat_card, vs_card
from engines.engine import run_full_analysis, export_excel, is_sample
from engines.ai_engine import chat_with_ai, verify_match, analyze_product, bulk_verify, suggest_price
from utils.db_manager import DatabaseManager
from utils.helpers import (render_filters, apply_filters, paginate_df,
                           export_to_excel, render_comparison_table, bg_tasks)
from utils.make_helper import send_to_make, send_price_updates, send_missing_products, test_webhook

# ===== إعداد الصفحة =====
st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")
st.markdown(get_main_css(), unsafe_allow_html=True)

db = DatabaseManager()

# ===== Session State =====
for k, v in [("results", None), ("chat_history", []), ("bg_status", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ===== الشريط الجانبي =====
with st.sidebar:
    st.markdown(f'<div style="text-align:center;padding:15px 0"><div style="font-size:2.2rem">{APP_ICON}</div><h2 style="margin:5px 0;color:#6C63FF">مهووس</h2><p style="color:#8B8B8B;font-size:.8rem">v{APP_VERSION}</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    labels = [f"{i} {n}" for i, n in SIDEBAR_SECTIONS]
    selected = st.radio("التنقل", labels, label_visibility="collapsed")
    page_name = selected.split(" ", 1)[1] if " " in selected else selected

    # حالة سريعة
    r = st.session_state.results
    if r and isinstance(r, dict) and "stats" in r:
        s = r["stats"]
        st.markdown("---")
        st.caption(f"آخر تحليل: {s.get('timestamp','')}")
        st.caption(f"المنتجات: {s.get('total',0)} | حرجة: {s.get('critical',0)}")

    # حالة المهام الخلفية
    for tid, info in st.session_state.bg_status.items():
        if info.get("running"):
            st.markdown(f'<div style="padding:6px;background:rgba(108,99,255,.1);border-radius:6px;font-size:.8rem">⏳ {info.get("label","مهمة")} جارية...</div>', unsafe_allow_html=True)


# =====================================================
# لوحة القيادة
# =====================================================
def page_dashboard():
    st.markdown("## 🏠 لوحة القيادة")
    r = st.session_state.results
    if not r or "stats" not in r:
        st.info("📋 ارفع الملفات من قسم 'رفع الملفات' للبدء.")
        return

    s = r["stats"]
    cards = [
        ("📊","إجمالي",s["total"],"#6C63FF"),("🔴","رفع",s["raise_count"],"#dc3545"),
        ("🟡","خفض",s["lower_count"],"#ffc107"),("🟢","موافق",s["approved_count"],"#28a745"),
        ("🔵","مفقود",s["missing_count"],"#007bff"),("⚠️","مراجعة",s["review_count"],"#ff9800"),
    ]
    cols = st.columns(6)
    for i,(ic,lb,vl,cl) in enumerate(cards):
        with cols[i]: st.markdown(stat_card(ic,lb,vl,cl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(data=[go.Pie(
            labels=["رفع","خفض","موافق","مفقود","مراجعة"],
            values=[s["raise_count"],s["lower_count"],s["approved_count"],s["missing_count"],s["review_count"]],
            marker=dict(colors=["#dc3545","#ffc107","#28a745","#007bff","#ff9800"]),
            hole=.5, textinfo="label+percent")])
        fig.update_layout(title="توزيع القرارات",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#FAFAFA",family="Tajawal"),height=350,showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        df_all = r.get("all", pd.DataFrame())
        if not df_all.empty and "الخطورة" in df_all.columns:
            rc = df_all["الخطورة"].value_counts()
            fig2 = go.Figure(data=[go.Bar(x=rc.index,y=rc.values,marker_color=["#FF1744" if x=="حرج" else "#FFD600" if x=="متوسط" else "#00C853" for x in rc.index])])
            fig2.update_layout(title="توزيع الخطورة",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#FAFAFA",family="Tajawal"),height=350,xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,.1)"))
            st.plotly_chart(fig2, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("متوسط الفرق",f"{s.get('avg_diff',0):.1f} ر.س")
    with c2: st.metric("منتجات حرجة",s.get("critical",0))
    with c3: st.metric("عدد المنافسين",s.get("competitors",0))
    with c4: st.metric("حد المطابقة",f"{s.get('threshold',60)}%")


# =====================================================
# رفع الملفات
# =====================================================
def page_upload():
    st.markdown("## 📤 رفع الملفات")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📦 ملف المتجر")
        my_file = st.file_uploader("ملف المتجر", type=["xlsx","csv"], key="my_f", label_visibility="collapsed")
    with c2:
        st.markdown("#### 👥 ملفات المنافسين")
        comp_files = st.file_uploader("المنافسين", type=["xlsx","csv"], accept_multiple_files=True, key="comp_f", label_visibility="collapsed")

    c1,c2,c3 = st.columns(3)
    with c1: threshold = st.slider("حد المطابقة %",30,100,MATCH_THRESHOLD,5)
    with c2: save_db = st.checkbox("حفظ في قاعدة البيانات",True)
    with c3: bg_mode = st.checkbox("معالجة في الخلفية",False)

    if st.button("⚡ بدء التحليل", type="primary", use_container_width=True, disabled=not(my_file and comp_files)):
        my_data = {"name": my_file.name, "data": my_file.getvalue()}
        comp_data = [{"name": f.name, "data": f.getvalue()} for f in comp_files]

        if bg_mode:
            # معالجة في الخلفية
            def bg_analyze():
                return run_full_analysis(my_data, comp_data, threshold)

            task_id = bg_tasks.run("analysis", bg_analyze)
            st.session_state.bg_status["analysis"] = {"running": True, "label": "التحليل"}
            st.success("⏳ بدأ التحليل في الخلفية. سيتم التحديث تلقائياً.")
            st.info("يمكنك التنقل بين الصفحات أثناء المعالجة.")
        else:
            # معالجة مباشرة
            bar = st.progress(0)
            status = st.empty()
            t0 = time.time()

            def cb(pct, msg):
                bar.progress(min(pct,100))
                status.markdown(f'<div style="color:#6C63FF;font-weight:700">{msg}</div>', unsafe_allow_html=True)

            cb(5, "📂 جاري القراءة...")
            results = run_full_analysis(my_data, comp_data, threshold, cb)

            if "error" in results:
                st.error(f"❌ {results['error']}")
                return

            st.session_state.results = results

            if save_db:
                for k in ["raise","lower","approved","review"]:
                    df = results.get(k, pd.DataFrame())
                    if not df.empty: db.save_results(df)

            elapsed = time.time() - t0
            cb(100, f"✅ اكتمل في {elapsed:.1f}ث!")
            s = results["stats"]
            st.success(f"**تم!** {s['total']} منتج | رفع: {s['raise_count']} | خفض: {s['lower_count']} | موافق: {s['approved_count']} | مفقود: {s['missing_count']}")
            db.log_action('analysis', f"{s['total']} منتج في {elapsed:.1f}ث", 'رفع')

    # تحقق من المهام الخلفية
    if bg_tasks.is_running("analysis"):
        st.info("⏳ التحليل جارٍ في الخلفية...")
        if st.button("🔄 تحقق من الحالة"):
            st.rerun()
    else:
        result = bg_tasks.get_result("analysis")
        if result and st.session_state.results is None:
            if "error" not in result:
                st.session_state.results = result
                st.session_state.bg_status.pop("analysis", None)
                st.success("✅ اكتمل التحليل في الخلفية!")
                st.rerun()


# =====================================================
# صفحة منتجات عامة (رفع/خفض/موافق/مراجعة)
# =====================================================
def page_products(section_key, title, icon, color):
    st.markdown(f'## {icon} {title}')
    r = st.session_state.results
    if not r or section_key not in r:
        st.info("لا توجد بيانات. ارفع الملفات أولاً.")
        return

    df = r[section_key]
    if df.empty:
        st.success(f"✅ لا توجد منتجات في قسم {title}")
        return

    st.markdown(f'<div style="color:{color};font-size:1.1rem;font-weight:700">{len(df)} منتج</div>', unsafe_allow_html=True)

    filters = render_filters(df, section_key)
    fdf = apply_filters(df, filters)
    st.caption(f"عرض {len(fdf)} من {len(df)}")

    # أزرار
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        if st.button("✅ موافقة جماعية", key=f"{section_key}_approve"):
            db.log_action('approve', f'{len(fdf)} منتج', title)
            st.success(f"تمت الموافقة على {len(fdf)} منتج")
    with c2:
        st.download_button("📥 Excel", export_to_excel(fdf), f"{section_key}.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{section_key}_dl")
    with c3:
        wh = db.get_setting("make_webhook_url","")
        if st.button("⚡ Make", key=f"{section_key}_make"):
            if wh:
                res = send_price_updates(wh, fdf)
                st.success(res["message"]) if res["success"] else st.error(res["message"])
            else: st.warning("أضف رابط Webhook في الإعدادات")
    with c4:
        if st.button("🤖 تحقق AI", key=f"{section_key}_ai"):
            # تحقق في الخلفية
            with st.spinner("جاري التحقق..."):
                records = fdf.head(20).to_dict('records')
                result = bulk_verify(records)
                st.markdown(f'<div style="background:#1A1A2E;padding:12px;border-radius:8px;border:1px solid #333344">{result}</div>', unsafe_allow_html=True)

    # الجدول البصري
    paged = paginate_df(fdf, PAGES_PER_TABLE, f"{section_key}_pg")
    render_comparison_table(paged, section_key)

    # عرض جنب إلى جنب
    with st.expander("👁️ عرض جنب إلى جنب"):
        for _, row in paged.iterrows():
            st.markdown(vs_card(
                row.get("المنتج",""), row.get("السعر",0),
                row.get("اسم المنافس",""), row.get("أقل سعر منافس",0),
                row.get("الفرق",0)
            ), unsafe_allow_html=True)
            if row.get("التفسير"):
                st.caption(row["التفسير"])


# =====================================================
# منتجات مفقودة
# =====================================================
def page_missing():
    st.markdown("## 🔵 منتجات مفقودة")
    r = st.session_state.results
    if not r or "missing" not in r:
        st.info("لا توجد بيانات.")
        return
    df = r["missing"]
    if df.empty:
        st.success("✅ لا توجد منتجات مفقودة!")
        return

    st.markdown(f'<div style="color:#007bff;font-size:1.1rem;font-weight:700">{len(df)} منتج مفقود</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        search = st.text_input("🔍 بحث", key="miss_s", placeholder="ابحث...")
    with c2:
        if "المنافس" in df.columns:
            opts = ["الكل"] + sorted(df["المنافس"].dropna().unique().tolist())
            comp = st.selectbox("المنافس", opts, key="miss_c")
        else: comp = "الكل"

    fdf = df.copy()
    if search and "المنتج" in fdf.columns:
        fdf = fdf[fdf["المنتج"].str.lower().str.contains(search.lower(), na=False)]
    if comp != "الكل" and "المنافس" in fdf.columns:
        fdf = fdf[fdf["المنافس"] == comp]

    c1,c2 = st.columns(2)
    with c1:
        st.download_button("📥 Excel", export_to_excel(fdf), "missing.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        wh = db.get_setting("make_webhook_url","")
        if st.button("⚡ إرسال Make", key="miss_make") and wh:
            res = send_missing_products(wh, fdf)
            st.success(res["message"]) if res["success"] else st.error(res["message"])

    paged = paginate_df(fdf, PAGES_PER_TABLE, "miss_pg")
    st.dataframe(paged, use_container_width=True, hide_index=True)


# =====================================================
# تحقق AI
# =====================================================
def page_ai_verify():
    st.markdown("## 🤖 تحقق بالذكاء الصناعي")
    r = st.session_state.results
    if not r:
        st.info("لا توجد بيانات.")
        return

    tab1, tab2 = st.tabs(["🔍 تحقق فردي", "📊 تحقق جماعي"])

    with tab1:
        df_all = r.get("all", pd.DataFrame())
        if not df_all.empty:
            names = df_all["المنتج"].tolist()
            sel = st.selectbox("اختر منتج", names, key="ai_sel")
            if st.button("🤖 تحقق", key="ai_single"):
                row = df_all[df_all["المنتج"]==sel].iloc[0]
                with st.spinner("جاري التحقق..."):
                    res = verify_match(row.get("المنتج",""), row.get("اسم المنافس",""), row.get("نسبة التطابق",0))
                st.markdown(f'<div style="background:#1A1A2E;padding:14px;border-radius:10px;border:1px solid #333344">{res}</div>', unsafe_allow_html=True)

    with tab2:
        section = st.selectbox("القسم", ["رفع سعر","خفض سعر","مراجعة"], key="ai_sec")
        sec_map = {"رفع سعر":"raise","خفض سعر":"lower","مراجعة":"review"}
        if st.button("⚡ تحقق جماعي", key="ai_bulk"):
            df = r.get(sec_map[section], pd.DataFrame())
            if df.empty:
                st.warning("لا توجد بيانات")
            else:
                with st.spinner(f"جاري التحقق من {len(df)} منتج..."):
                    records = df.head(20).to_dict('records')
                    res = bulk_verify(records)
                st.markdown(f'<div style="background:#1A1A2E;padding:14px;border-radius:10px;border:1px solid #333344">{res}</div>', unsafe_allow_html=True)


# =====================================================
# دردشة AI
# =====================================================
def page_ai_chat():
    st.markdown("## 💬 دردشة AI")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div style="background:rgba(108,99,255,.12);padding:10px;border-radius:10px;margin:5px 0;border-right:3px solid #6C63FF"><b>👤 أنت:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:rgba(40,167,69,.08);padding:10px;border-radius:10px;margin:5px 0;border-right:3px solid #28a745"><b>🤖 مهووس AI:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("اكتب رسالتك...")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        context = ""
        r = st.session_state.results
        if r and "stats" in r:
            context = f"إحصائيات: {json.dumps(r['stats'], ensure_ascii=False)}"
        with st.spinner("🧠 جاري التفكير..."):
            response = chat_with_ai(user_input, context)
        st.session_state.chat_history.append({"role":"assistant","content":response})
        st.rerun()

    if st.button("🗑️ مسح", key="clr_chat"):
        st.session_state.chat_history = []
        st.rerun()


# =====================================================
# Make أتمتة
# =====================================================
def page_make():
    st.markdown("## ⚡ Make.com أتمتة")

    current_wh = db.get_setting("make_webhook_url","")
    wh = st.text_input("رابط Webhook", value=current_wh, type="password", key="make_wh")

    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("💾 حفظ", key="make_save"):
            db.save_setting("make_webhook_url", wh)
            st.success("✅ تم الحفظ")
    with c2:
        if st.button("🔌 اختبار", key="make_test"):
            if wh:
                st.success("✅ ناجح!") if test_webhook(wh) else st.error("❌ فشل")
            else: st.warning("أدخل الرابط أولاً")
    with c3:
        if st.button("⚡ إرسال الكل", key="make_all"):
            r = st.session_state.results
            if r and wh:
                for k in ["raise","lower"]:
                    df = r.get(k, pd.DataFrame())
                    if not df.empty: send_price_updates(wh, df)
                st.success("✅ تم الإرسال!")
            else: st.warning("لا توجد نتائج أو رابط")

    st.markdown("### 📋 سجل الإرسال")
    log = db.get_audit_log(20)
    if not log.empty:
        ml = log[log["action"].str.contains("make|send|webhook", case=False, na=False)]
        if not ml.empty: st.dataframe(ml[["timestamp","action","details"]], use_container_width=True, hide_index=True)
        else: st.info("لا يوجد سجل")
    else: st.info("لا يوجد سجل")


# =====================================================
# قاعدة البيانات
# =====================================================
def page_database():
    st.markdown("## 💾 قاعدة البيانات")
    tab1, tab2, tab3 = st.tabs(["📊 إحصائيات","📋 سجلات","📝 أحداث"])

    with tab1:
        s = db.get_statistics()
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("إجمالي السجلات", s['total'])
        with c2: st.metric("متوسط الفرق", f"{s['avg_price_diff']:.1f} ر.س")
        with c3: st.metric("متوسط التطابق", f"{s['avg_match_score']:.0f}%")

    with tab2:
        df = db.get_all_results(100)
        if not df.empty: st.dataframe(df, use_container_width=True, hide_index=True)
        else: st.info("لا توجد سجلات")

    with tab3:
        log = db.get_audit_log(50)
        if not log.empty: st.dataframe(log, use_container_width=True, hide_index=True)
        else: st.info("لا يوجد سجل")

    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🗑️ مسح النتائج", key="db_clr"):
            db.clear_results()
            st.success("✅ تم المسح")
    with c2:
        df = db.get_all_results()
        if not df.empty:
            st.download_button("📥 تصدير", export_to_excel(df), "db_export.xlsx",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =====================================================
# الإعدادات
# =====================================================
def page_settings():
    st.markdown("## ⚙️ الإعدادات")
    tab1, tab2, tab3 = st.tabs(["🔧 عام","🤖 AI","🔗 تكاملات"])

    with tab1:
        threshold = st.slider("حد المطابقة %",30,100,int(db.get_setting("match_threshold",str(MATCH_THRESHOLD))),5,key="s_th")
        price_tol = st.slider("تفاوت السعر (ر.س)",1,20,int(db.get_setting("price_tolerance",str(PRICE_TOLERANCE))),1,key="s_pt")
        pages = st.number_input("منتجات/صفحة",10,100,int(db.get_setting("pages_per_table",str(PAGES_PER_TABLE))),key="s_pp")
        if st.button("💾 حفظ",key="s_save"):
            db.save_setting("match_threshold",str(threshold))
            db.save_setting("price_tolerance",str(price_tol))
            db.save_setting("pages_per_table",str(pages))
            st.success("✅ تم الحفظ")

    with tab2:
        st.markdown("### مفاتيح AI")
        gemini_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password", key="s_gk")
        openrouter_key = st.text_input("OpenRouter API Key", value=OPENROUTER_API_KEY, type="password", key="s_ok")
        if st.button("💾 حفظ AI",key="s_ai_save"):
            db.save_setting("gemini_api_key", gemini_key)
            db.save_setting("openrouter_api_key", openrouter_key)
            st.success("✅ تم الحفظ")
        if st.button("🔌 اختبار AI",key="s_ai_test"):
            with st.spinner("جاري الاختبار..."):
                res = chat_with_ai("مرحباً، هل أنت متصل؟")
            st.markdown(f'<div style="background:#1A1A2E;padding:10px;border-radius:8px;border:1px solid #333344">{res}</div>', unsafe_allow_html=True)

    with tab3:
        wh = st.text_input("Make Webhook", value=db.get_setting("make_webhook_url",""), type="password", key="s_wh")
        if st.button("💾 حفظ",key="s_int_save"):
            db.save_setting("make_webhook_url", wh)
            st.success("✅ تم الحفظ")

    st.markdown("---")
    st.markdown(f'<div style="background:#1A1A2E;padding:12px;border-radius:8px;border:1px solid #333344"><b>النظام:</b> v{APP_VERSION} | <b>المحرك:</b> RapidFuzz+AI | <b>DB:</b> SQLite | <b>AI:</b> Gemini+OpenRouter</div>', unsafe_allow_html=True)


# =====================================================
# التوجيه
# =====================================================
PAGES = {
    "لوحة القيادة": page_dashboard,
    "رفع الملفات": page_upload,
    "رفع سعر": lambda: page_products("raise","رفع سعر","🔴","#dc3545"),
    "خفض سعر": lambda: page_products("lower","خفض سعر","🟡","#ffc107"),
    "موافق عليها": lambda: page_products("approved","موافق عليها","🟢","#28a745"),
    "منتجات مفقودة": page_missing,
    "يحتاج مراجعة": lambda: page_products("review","يحتاج مراجعة","⚠️","#ff9800"),
    "تحقق AI": page_ai_verify,
    "دردشة AI": page_ai_chat,
    "Make أتمتة": page_make,
    "قاعدة البيانات": page_database,
    "الإعدادات": page_settings,
}

fn = PAGES.get(page_name, page_dashboard)
fn()
