"""
ai_engine.py - محرك الذكاء الصناعي v17.2
- تحسين: فحص الدفعات (Batch Processing) لتقليل وقت الانتظار.
- ميزة: الحكم الذكي (Smart Judge) للمنتجات المشكوك فيها.
- تكامل: يدعم Gemini 2.0 Flash لسرعة استجابة فائقة.
"""
import requests
import json
import time
from config import GEMINI_API_KEYS, OPENROUTER_API_KEY

# ===== إعدادات النماذج =====
# نستخدم Gemini Flash لأنه الأسرع والأرخص للمقارنات الكبيرة
GEMINI_MODEL = "gemini-2.0-flash" 
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# إعدادات OpenRouter كبديل
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"

# ===== System Prompts (عقل النظام) =====
PROMPTS = {
    "verify_batch": """أنت خبير مطابقة منتجات عطور. سأعطيك قائمة أزواج (منتجنا vs منتج المنافس).
مهمتك: لكل زوج حدد هل هو "نفس المنتج" (Match) أم "مختلف" (Mismatch).
قواعد المطابقة:
1. تجاهل الفروق البسيطة في الأسماء (مثل Sauvage vs Savage).
2. الحجم يجب أن يكون متطابقاً (100ml لا تساوي 50ml).
3. النوع يجب أن يكون متطابقاً (EDP لا يساوي EDT).
4. التستر (Tester) لا يساوي العطر الأصلي إلا إذا ذكرت ذلك.

أجب بصيغة JSON فقط:
[
  {"id": 1, "match": true, "reason": "تطابق تام"},
  {"id": 2, "match": false, "reason": "اختلاف الحجم"}
]""",
    
    "suggest_price": """أنت خبير تسعير. لديك منتج وسعره الحالي وأسعار المنافسين.
اقترح سعراً جديداً يحقق التوازن بين الربح والمنافسة.
قواعد:
1. لا تنزل عن المنافس بأكثر من 10 ريال (حرق أسعار).
2. إذا كنا الأرخص، اقترح رفع السعر قليلاً لزيادة الربح مع البقاء الأرخص.
"""
}

def _call_gemini(prompt, system_prompt=""):
    """اتصال مباشر بـ Gemini مع تدوير المفاتيح"""
    full_text = f"{system_prompt}\n\nالمهمة:\n{prompt}"
    payload = {
        "contents": [{"parts": [{"text": full_text}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"} # إجبار على JSON
    }
    
    for key in GEMINI_API_KEYS:
        if not key: continue
        try:
            url = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={key}"
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code == 429:
                continue # تجربة المفتاح التالي عند انتهاء الحصة
        except:
            continue
    return None

def _call_openrouter(prompt, system_prompt=""):
    """اتصال بديل عبر OpenRouter"""
    if not OPENROUTER_API_KEY: return None
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
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
    return None

def call_ai_json(prompt, type="verify_batch"):
    """دالة موحدة لطلب JSON من AI"""
    sys_p = PROMPTS.get(type, "")
    
    # محاولة 1: Gemini
    res = _call_gemini(prompt, sys_p)
    if res: return parse_json(res)
    
    # محاولة 2: OpenRouter
    res = _call_openrouter(prompt, sys_p)
    if res: return parse_json(res)
    
    return None

def parse_json(text):
    """تنظيف وتحويل النص إلى JSON"""
    try:
        # إزالة أي نصوص قبل أو بعد الـ JSON
        start = text.find('[')
        if start == -1: start = text.find('{')
        end = text.rfind(']') + 1
        if end == 0: end = text.rfind('}') + 1
        
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return json.loads(text) # محاولة مباشرة
    except:
        return None

# ===== الوظائف الرئيسية =====

def smart_bulk_verify(rows):
    """
    التحقق من قائمة منتجات دفعة واحدة (أسرع بـ 10 مرات من التحقق الفردي)
    rows: list of dicts [{'id': 1, 'our': '...', 'comp': '...'}, ...]
    """
    if not rows: return []
    
    # تجهيز النص
    lines = []
    for r in rows:
        lines.append(f"ID: {r['id']}\nمنتجنا: {r['our']}\nالمنافس: {r['comp']}\n---")
    
    prompt = "\n".join(lines)
    result = call_ai_json(prompt, "verify_batch")
    
    if result:
        # دمج النتائج مع البيانات الأصلية
        lookup = {item['id']: item for item in result if 'id' in item}
        final = []
        for r in rows:
            decision = lookup.get(r['id'])
            if decision:
                r['ai_match'] = decision.get('match', False)
                r['ai_reason'] = decision.get('reason', 'تم التحقق')
            else:
                r['ai_match'] = None # فشل التحقق لهذا العنصر
            final.append(r)
        return final
    return rows # إعادة الأصلي في حال الفشل

def verify_single_match(our_name, comp_name, our_price, comp_price):
    """تحقق فردي دقيق"""
    prompt = f"""
    قارن بدقة:
    1. {our_name} ({our_price} ريال)
    2. {comp_name} ({comp_price} ريال)
    
    هل هما نفس المنتج؟ ولماذا؟
    أجب بـ JSON: {{"match": boolean, "confidence": int, "reason": string}}
    """
    # نستخدم verify_batch كقالب لأنه يطلب JSON
    res = call_ai_json(prompt, "verify_batch")
    if isinstance(res, list): res = res[0] # التعامل مع احتمال عودة قائمة
    return res

def suggest_optimization(product, current_price, comp_prices):
    """اقتراح تحسين السعر"""
    prices_str = ", ".join([str(p) for p in comp_prices])
    prompt = f"""
    المنتج: {product}
    سعرنا: {current_price}
    المنافسون: {prices_str}
    
    اقترح السعر المثالي وأعط سبباً.
    أجب بـ JSON: {{"suggested_price": float, "strategy": string}}
    """
    return call_ai_json(prompt, "suggest_price")
