"""
ai_engine.py - محرك الذكاء الصناعي v17.4 (Perfume Expert Edition)
- تحسين البرومبت ليكون خبيراً في فك طلاسم أسماء العطور
- القدرة على التمييز بين التركيزات (EDP/EDT) والأحجام
- مخرجات JSON دقيقة لدمجها في النظام
"""
import requests
import json
import time
import re
from config import GEMINI_API_KEYS, OPENROUTER_API_KEY

# ===== إعدادات النماذج =====
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"

# ===== الشخصية (System Prompt) =====
PERFUME_EXPERT_PROMPT = """
أنت خبير بيانات ومتخصص في العطور العالمية. مهمتك هي المطابقة الدقيقة بين المنتجات.
قواعد المطابقة الصارمة:
1. الماركة (Brand): يجب أن تكون متطابقة تماماً.
2. العطر (Line): "Sauvage" يختلف عن "Sauvage Elixir".
3. التركيز (Concentration): الـ EDP يختلف عن EDT يختلف عن Parfum. (إلا إذا طُلب تجاهل ذلك).
4. الحجم (Size): 100ml يختلف عن 50ml. (تسامح بسيط 3.3oz = 100ml).
5. النوع (Tester): التستر هو نفس العطر ولكن بسعر أرخص (طابقهم ولكن نبهني).

أجب دائماً بصيغة JSON فقط.
"""

def _call_gemini(prompt, system_prompt=""):
    """الاتصال بـ Gemini مع التدوير بين المفاتيح"""
    full_prompt = f"{system_prompt}\n\n{prompt}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
    }
    
    for key in GEMINI_API_KEYS:
        if not key: continue
        try:
            url = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={key}"
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except:
            continue
    return None

def _call_openrouter(prompt, system_prompt=""):
    """الاتصال بـ OpenRouter كاحتياطي"""
    if not OPENROUTER_API_KEY: return None
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=20)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except:
        return None

def call_ai_json(prompt, system_prompt=PERFUME_EXPERT_PROMPT):
    """دالة موحدة تعيد JSON دائماً"""
    # محاولة 1: Gemini
    res = _call_gemini(prompt, system_prompt)
    if not res:
        # محاولة 2: OpenRouter
        res = _call_openrouter(prompt, system_prompt)
    
    if res:
        try:
            # تنظيف الرد لضمان أنه JSON نقي
            cleaned = res.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except:
            return None
    return None

# ===== دوال التطبيق =====

def verify_match_smart(our_name, comp_name, our_price, comp_price):
    """
    التحقق الذكي لمنتج واحد (للحالات المشكوك فيها)
    """
    prompt = f"""
    قارن بين هذين المنتجين:
    1. منتجنا: "{our_name}" (السعر: {our_price})
    2. المنافس: "{comp_name}" (السعر: {comp_price})

    هل هما نفس المنتج تماماً؟
    أجب بـ JSON:
    {{
        "match": true/false,
        "confidence": 0-100,
        "issue": "لا يوجد/اختلاف حجم/اختلاف تركيز/اختلاف عطر",
        "action": "موافق/مراجعة/رفض"
    }}
    """
    res = call_ai_json(prompt)
    if res:
        return {"success": True, **res}
    return {"success": False, "match": False, "confidence": 0, "issue": "فشل الاتصال"}

def bulk_resolve_reviews(items_list):
    """
    معالجة قائمة المراجعة دفعة واحدة (للسرعة)
    يستقبل قائمة: [{"id": 1, "our": "...", "comp": "..."}]
    """
    if not items_list: return []
    
    # تحويل القائمة لنص
    items_text = ""
    for item in items_list:
        items_text += f"- ID {item['id']}: Our='{item['our']}' VS Comp='{item['comp']}'\n"
        
    prompt = f"""
    لديك قائمة أزواج من المنتجات. حدد هل كل زوج متطابق أم لا.
    تذكر: تجاهل الفروقات البسيطة في الكتابة، ركز على الجوهر (الماركة، العطر، الحجم، التركيز).
    
    القائمة:
    {items_text}
    
    أعد JSON قائمة بالنتائج:
    [
        {{"id": 1, "match": true, "reason": "..."}},
        {{"id": 2, "match": false, "reason": "Different size"}}
    ]
    """
    
    res = call_ai_json(prompt)
    return res if isinstance(res, list) else []

def find_semantic_match(missing_product, candidates_list):
    """
    البحث عن منتج مفقود بين قائمة مرشحين (للمنتجات التي فشل البحث العادي في إيجادها)
    """
    candidates_text = "\n".join([f"- {c}" for c in candidates_list])
    prompt = f"""
    لدي منتج مفقود: "{missing_product}"
    وهذه قائمة منتجات عند المنافس:
    {candidates_text}
    
    هل يوجد أي منتج في القائمة هو نفسه المنتج المفقود (حتى لو اختلف الاسم قليلاً)؟
    
    أجب بـ JSON:
    {{
        "found": true/false,
        "matched_name": "اسم المنتج من القائمة أو فارغ",
        "confidence": 0-100
    }}
    """
    return call_ai_json(prompt)

# دوال التوافق مع الكود القديم
def chat_with_ai(msg, hist):
    return {"success": True, "response": "ميزة الدردشة قيد التحديث لتعمل مع المحرك الجديد."}

def verify_match(p1, p2, pr1=0, pr2=0):
    return verify_match_smart(p1, p2, pr1, pr2)

def analyze_product(p, pr):
    return {"success": False, "response": "غير مفعل"}

def suggest_price(p): return 0
def bulk_verify(items, page): return {"success": False, "response": "استخدم الدالة الجديدة"}
def process_paste(txt, page): return {"success": False}
def check_duplicate(n, l): return {"success": False}
