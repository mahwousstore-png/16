"""
نظام التسعير الذكي - مهووس v17.2
- Core: Hybrid Vectorization Engine (فائق السرعة)
- UI: Non-blocking Background Tasks (لا يتجمد المتصفح)
- AI: Smart Batch Verification (Gemini 2.0 Flash)
"""
import streamlit as st
import pandas as pd
import time
from config import *
from styles import get_styles, stat_card, vs_card
from engines.engine import (read_file, run_full_analysis, find_missing_products, 
                            export_excel, export_section_excel)
from engines.ai_engine import (chat_with_ai, verify_single_match, analyze_product, 
                               smart_bulk_verify, suggest_price, call_ai_json)
from utils.helpers import (TaskManager, apply_filters, get_filter_options, export_to_excel, 
                           export_multiple_sheets, parse_pasted_text, format_price, format_diff)
from utils.make_helper import (send_price_updates, send_new_products, send_missing_products, 
                               verify_webhook_connection, export_to_make_format)
from utils.db_manager import (init_db, log_event, log_decision, log_analysis, 
                              get_events, get_decisions, get_analysis_history)

# ===== إعداد الصفحة =====
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")
st.markdown(get_styles(), unsafe_allow_html=True)
init_db()

# ===== إدارة الحالة (Session State) =====
if 'results' not in st.session_state: st.session_state.results = None
if 'task_id' not in st.session_state: st.session_state.task_id = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# ===== الشريط الجانبي (Sidebar) =====
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.caption(f"Engine: v17.2 (Vectorized)")
    
    page = st.radio("القائمة الرئيسية", SECTIONS, label_visibility="collapsed")
    st.markdown("---")
    
    # ملخص الحالة
    if st.session_state.results:
        r = st.session_state.results
        st.markdown("**📊 إحصائيات سريعة:**")
        st.caption(f"🔴 رفع سعر: {len(r.get('price_raise', []))}")
        st.caption(f"🟢 خفض سعر: {len(r.get('price_lower', []))}")
        st.caption(f"🔍 مفقودات: {len(r.get('missing', []))}")

# ===== منطق المهام الخلفية (Background Task Polling) =====
if st.session_state.task_id:
    status = TaskManager.get_status(st.session_state.task_id)
    
    if status['status'] == 'running':
        st.info(f"⏳ {status['message']} ({status['progress']}%)")
        my_bar = st.progress(status['progress'])
        time.sleep(1) # تحديث كل ثانية
        st.rerun() # إعادة تحميل الصفحة لتحديث الشريط
        
    elif status['status'] == 'completed':
        st.success("✅ اكتمل التحليل بنجاح!")
        st.balloons()
        
        # معالجة النتائج وحفظها في Session
        full_df = status['result']
        
        # تقسيم النتائج
        results = {
            "price_raise": full_df[full_df["القرار"].str.contains("أعلى", na=False)],
            "price_lower": full_df[full_df["القرار"].str.contains("أقل", na=False)],
            "approved": full_df[full_df["القرار"].str.contains("موافق", na=False)],
            "review": full_df[full_df["القرار"].str.contains("مراجعة", na=False)],
            "all": full_df
        }
        
        # البحث عن المفقودات (مهمة فرعية سريعة)
        if 'comp_dfs_cache' in st.session_state:
             missing_df = find_missing_products(st.session_state.our_df_cache, st.session_state.comp_dfs_cache)
             results["missing"] = missing_df
        
        st.session_state.results = results
        st.session_state.task_id = None # إنهاء المهمة
        TaskManager.clear_task(st.session_state.task_id)
        st.rerun()
        
    elif status['status'] == 'failed':
        st.error(f"❌ حدث خطأ: {status['message']}")
        st.session_state.task_id = None

# ===== صفحة 1: رفع الملفات وتشغيل المحرك =====
if page == "📂 رفع الملفات":
    st.header("📂 مركز البيانات")
    
    col1, col2 = st.columns(2)
    with col1:
        our_file = st.file_uploader("📦 ملف منتجاتنا (الأساسي)", type=["csv", "xlsx"])
    with col2:
        comp_files = st.file_uploader("🏪 ملفات المنافسين", type=["csv", "xlsx"], accept_multiple_files=True)

    if st.button("🚀 تشغيل المحرك (Vector Engine)", type="primary", disabled=st.session_state.task_id is not None):
        if our_file and comp_files:
            # قراءة الملفات
            our_df, err = read_file(our_file)
            if err: st.error(err); st.stop()
            
            comp_dfs = {}
            for f in comp_files:
                cdf, cerr = read_file(f)
                if not cerr: comp_dfs[f.name] = cdf
            
            if not comp_dfs: st.error("لم يتم قراءة أي ملف منافس"); st.stop()

            # حفظ نسخة للكاش (للمفقودات لاحقاً)
            st.session_state.our_df_cache = our_df
            st.session_state.comp_dfs_cache = comp_dfs

            # بدء المهمة في الخلفية
            task_id = TaskManager.start_task(run_full_analysis, our_df, comp_dfs)
            st.session_state.task_id = task_id
            st.rerun()

# ===== صفحات النتائج (Dynamic Rendering) =====
elif page in ["🔴 سعر أعلى", "🟢 سعر أقل", "✅ موافق عليها", "⚠️ تحت المراجعة"]:
    
    # خريطة المفاتيح
    key_map = {
        "🔴 سعر أعلى": "price_raise",
        "🟢 سعر أقل": "price_lower",
        "✅ موافق عليها": "approved",
        "⚠️ تحت المراجعة": "review"
    }
    current_key = key_map[page]
    
    if st.session_state.results and current_key in st.session_state.results:
        df = st.session_state.results[current_key]
        st.header(f"{page} ({len(df)})")
        
        # 1. الفلاتر
        with st.expander("🔍 فلاتر البحث المتقدم", expanded=False):
            f_opts = get_filter_options(df)
            c1, c2, c3, c4 = st.columns(4)
            filters = {
                "search": c1.text_input("بحث نصي", key=f"s_{current_key}"),
                "brand": c2.selectbox("الماركة", f_opts["brands"], key=f"b_{current_key}"),
                "match_min": c3.slider("دقة التطابق %", 0, 100, 0, key=f"m_{current_key}"),
                "diff_min": c4.number_input("الحد الأدنى للفرق", 0, key=f"d_{current_key}")
            }
        
        filtered_df = apply_filters(df, filters)
        
        # 2. الأوامر الجماعية (Bulk Actions)
        col_act1, col_act2, col_act3 = st.columns([1,1,2])
        if col_act1.button("🤖 تحقق AI (أول 20)", key=f"ai_{current_key}"):
            with st.spinner("جاري استشارة Gemini..."):
                # تجهيز البيانات للذكاء الصناعي
                rows_to_check = []
                for idx, row in filtered_df.head(20).iterrows():
                    rows_to_check.append({
                        "id": idx, 
                        "our": row.get("المنتج"), 
                        "comp": row.get("منتج المنافس")
                    })
                
                ai_res = smart_bulk_verify(rows_to_check)
                # عرض النتائج
                for res in ai_res:
                    icon = "✅" if res.get('ai_match') else "❌"
                    st.write(f"{icon} {res['our']} -> {res['ai_reason']}")

        if col_act2.button("📥 تصدير Excel", key=f"ex_{current_key}"):
            data = export_to_excel(filtered_df, page)
            st.download_button("⬇️ تحميل الملف", data, f"{current_key}.xlsx")

        # 3. جدول العرض (Visual Table)
        for i, row in filtered_df.iterrows():
            # تحضير البيانات
            p_our = row.get("المنتج")
            p_comp = row.get("منتج المنافس")
            pr_our = row.get("السعر", 0)
            pr_comp = row.get("سعر المنافس", 0)
            diff = row.get("الفرق", 0)
            score = row.get("نسبة التطابق", 0)
            
            st.markdown(vs_card(
                p_our, pr_our, p_comp, pr_comp, diff, row.get("المنافس")
            ), unsafe_allow_html=True)
            
            # أزرار الإجراءات السريعة
            b1, b2, b3, b4 = st.columns([1,1,1,3])
            if b1.button("✅", key=f"ok_{i}_{current_key}", help="موافق"):
                log_decision(p_our, current_key, "approved")
                st.toast("تمت الموافقة")
            if b2.button("🗑️", key=f"del_{i}_{current_key}", help="إزالة"):
                log_decision(p_our, current_key, "removed")
                st.toast("تمت الإزالة")
            
            # عرض تفاصيل المنافسين الآخرين إن وجدت
            others = row.get("جميع المنافسين")
            if others and isinstance(others, list) and len(others) > 1:
                with st.expander(f"➕ {len(others)-1} منافسين آخرين"):
                    for o in others:
                        if o.get('name') != p_comp: # عدم تكرار العرض
                            st.caption(f"🏪 {o.get('competitor')}: {o.get('name')} | 💰 {o.get('price')} | 🔗 {o.get('score')}%")

            st.markdown("---")
            
    else:
        st.info("لا توجد بيانات. يرجى رفع الملفات أولاً.")

# ===== صفحة المفقودات =====
elif page == "🔍 منتجات مفقودة":
    if st.session_state.results and "missing" in st.session_state.results:
        df = st.session_state.results["missing"]
        st.header(f"🔍 منتجات مفقودة ({len(df)})")
        st.warning("هذه المنتجات موجودة عند المنافسين وليست في ملفك.")
        
        with st.expander("فلاتر"):
            f_opts = get_filter_options(df)
            brand_f = st.selectbox("الماركة", f_opts["brands"], key="miss_b")
            filters = {"brand": brand_f}
            
        f_df = apply_filters(df, filters)
        st.dataframe(f_df, use_container_width=True)
        
        if st.button("📤 إرسال إلى Make (إضافة منتجات)"):
             formatted = export_to_make_format(f_df, "missing")
             res = send_missing_products(formatted)
             if res['success']: st.success("تم الإرسال!")
             else: st.error(res['message'])
    else:
        st.info("لا توجد بيانات مفقودة.")

# ===== بقية الصفحات (Dashboard, Settings, etc.) يمكن إبقاؤها كما هي =====
elif page == "📊 لوحة التحكم":
    st.title("لوحة التحكم المركزية")
    # (نفس كود الإحصائيات السابق)

elif page == "🤖 الذكاء الصناعي":
    st.header("🤖 المحلل الذكي")
    q = st.text_input("اسأل عن بياناتك:")
    if q:
        res = chat_with_ai(q, st.session_state.chat_history)
        st.write(res['response'])
