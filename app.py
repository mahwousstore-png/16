"""
نظام التسعير الذكي - مهووس v17.6 (Manual Column Mapping)
- ميزة جديدة: تحديد الأعمدة يدوياً لضمان الدقة 100%.
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

# ===== إدارة الحالة =====
if 'results' not in st.session_state: st.session_state.results = None
if 'task_id' not in st.session_state: st.session_state.task_id = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'our_df_preview' not in st.session_state: st.session_state.our_df_preview = None
if 'comp_df_preview' not in st.session_state: st.session_state.comp_df_preview = None

# ===== الشريط الجانبي =====
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.caption(f"Engine: v17.6 (Manual Mapping)")
    page = st.radio("القائمة الرئيسية", SECTIONS, label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.results:
        r = st.session_state.results
        st.markdown("**📊 إحصائيات سريعة:**")
        st.caption(f"🔴 رفع سعر: {len(r.get('price_raise', []))}")
        st.caption(f"🟢 خفض سعر: {len(r.get('price_lower', []))}")
        st.caption(f"🔍 مفقودات: {len(r.get('missing', []))}")

# ===== المهام الخلفية =====
if st.session_state.task_id:
    status = TaskManager.get_status(st.session_state.task_id)
    if status['status'] == 'running':
        st.info(f"⏳ {status['message']} ({status['progress']}%)")
        my_bar = st.progress(status['progress'])
        time.sleep(1)
        st.rerun()
    elif status['status'] == 'completed':
        st.success("✅ اكتمل التحليل بنجاح!")
        st.balloons()
        full_df = status['result']
        results = {
            "price_raise": full_df[full_df["القرار"].str.contains("أعلى", na=False)],
            "price_lower": full_df[full_df["القرار"].str.contains("أقل", na=False)],
            "approved": full_df[full_df["القرار"].str.contains("موافق", na=False)],
            "review": full_df[full_df["القرار"].str.contains("مراجعة", na=False)],
            "all": full_df,
            "missing": find_missing_products(st.session_state.our_df_cache, st.session_state.comp_dfs_cache)
        }
        st.session_state.results = results
        st.session_state.task_id = None
        TaskManager.clear_task(st.session_state.task_id)
        st.rerun()
    elif status['status'] == 'failed':
        st.error(f"❌ حدث خطأ: {status['message']}")
        st.session_state.task_id = None

# ===== صفحة رفع الملفات (المعدلة) =====
if page == "📂 رفع الملفات":
    st.header("📂 مركز البيانات وتحديد الأعمدة")
    
    col1, col2 = st.columns(2)
    with col1:
        our_file = st.file_uploader("📦 ملف منتجاتنا", type=["csv", "xlsx"])
    with col2:
        comp_files = st.file_uploader("🏪 ملفات المنافسين", type=["csv", "xlsx"], accept_multiple_files=True)

    # قراءة أولية للملفات لعرض الأعمدة
    if our_file:
        try:
            if st.session_state.our_df_preview is None:
                df, _ = read_file(our_file)
                st.session_state.our_df_preview = df
        except: pass

    if comp_files and len(comp_files) > 0:
        try:
            if st.session_state.comp_df_preview is None:
                df, _ = read_file(comp_files[0]) # قراءة أول ملف منافس كعينة
                st.session_state.comp_df_preview = df
        except: pass

    # منطقة تحديد الأعمدة (Mapping)
    mapping = {}
    if st.session_state.our_df_preview is not None and st.session_state.comp_df_preview is not None:
        st.info("👇 يرجى تحديد الأعمدة الصحيحة لضمان دقة التحليل")
        
        with st.expander("⚙️ إعدادات الأعمدة (مهم جداً)", expanded=True):
            c1, c2 = st.columns(2)
            
            # أعمدة ملفنا
            with c1:
                st.markdown("**بياناتنا:**")
                our_cols = st.session_state.our_df_preview.columns.tolist()
                mapping['our_name'] = st.selectbox("عمود اسم المنتج (عندنا)", our_cols, index=0)
                mapping['our_price'] = st.selectbox("عمود السعر (عندنا)", our_cols, index=min(1, len(our_cols)-1))
            
            # أعمدة المنافس
            with c2:
                st.markdown("**بيانات المنافس (عينة):**")
                comp_cols = st.session_state.comp_df_preview.columns.tolist()
                mapping['comp_name'] = st.selectbox("عمود اسم المنتج (المنافس)", comp_cols, index=0)
                mapping['comp_price'] = st.selectbox("عمود السعر (المنافس)", comp_cols, index=min(1, len(comp_cols)-1))

        if st.button("🚀 تشغيل المحرك", type="primary", disabled=st.session_state.task_id is not None):
            # إعادة قراءة الملفات لتمريرها للمحرك
            our_df, _ = read_file(our_file)
            comp_dfs = {}
            for f in comp_files:
                cdf, _ = read_file(f)
                if cdf is not None: comp_dfs[f.name] = cdf
            
            st.session_state.our_df_cache = our_df
            st.session_state.comp_dfs_cache = comp_dfs
            
            # تمرير خريطة الأعمدة للمحرك
            task_id = TaskManager.start_task(run_full_analysis, our_df, comp_dfs, mapping=mapping)
            st.session_state.task_id = task_id
            st.rerun()
            
    elif our_file or comp_files:
        st.warning("يرجى رفع الملفات لظهور خيارات تحديد الأعمدة.")

# ===== بقية الصفحات كما هي =====
elif page in ["🔴 سعر أعلى", "🟢 سعر أقل", "✅ موافق عليها", "⚠️ تحت المراجعة"]:
    key_map = {"🔴 سعر أعلى": "price_raise", "🟢 سعر أقل": "price_lower", 
               "✅ موافق عليها": "approved", "⚠️ تحت المراجعة": "review"}
    current_key = key_map[page]
    
    if st.session_state.results and current_key in st.session_state.results:
        df = st.session_state.results[current_key]
        st.header(f"{page} ({len(df)})")
        
        with st.expander("🔍 فلاتر"):
            f_opts = get_filter_options(df)
            c1, c2 = st.columns(2)
            filters = {
                "search": c1.text_input("بحث", key=f"s_{current_key}"),
                "brand": c2.selectbox("الماركة", f_opts["brands"], key=f"b_{current_key}")
            }
        
        filtered_df = apply_filters(df, filters)
        
        # عرض البيانات
        for i, row in filtered_df.iterrows():
            st.markdown(vs_card(
                row.get("المنتج"), row.get("السعر"), 
                row.get("منتج المنافس"), row.get("سعر المنافس"), 
                row.get("الفرق"), row.get("المنافس")
            ), unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("لا توجد بيانات.")

elif page == "🔍 منتجات مفقودة":
    if st.session_state.results and "missing" in st.session_state.results:
        st.dataframe(st.session_state.results["missing"], use_container_width=True)
    else:
        st.info("لا توجد مفقودات.")

elif page == "📊 لوحة التحكم":
    st.title("لوحة التحكم")

elif page == "🤖 الذكاء الصناعي":
    st.header("مساعد AI")
    q = st.text_input("سؤالك:")
    if q: st.write(chat_with_ai(q)['response'])
