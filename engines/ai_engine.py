"""
ai_engine.py - محرك الذكاء الصناعي v17.3 (إصلاح شامل)
- يدعم JSON (للمطابقة السريعة) و Text (للدردشة).
- يحتوي على كافة الدوال التي يطلبها app.py.
"""
import requests
import json
import time
from config import GEMINI_API_KEYS, OPENROUTER_API_KEY

# ===== إعدادات النماذج =====
GEMINI_MODEL = "gemini-2.0-flash" 
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"

# ===== System Prompts =====
PROMPTS = {
    "verify_batch": """أنت خبير مطابقة منتجات عطور.
    أجب بـ JSON فقط: [{"id": 1, "match": true, "reason": "..."}]""",
    
    "general": """أنت مساعد ذكي متخصص في تسعير العطور. أجب بالعربية باحترافية."""
}

# ===== دوال الاتصال الأساسية =====

def _call_gemini(prompt, system_prompt="", json_mode=True):
    """اتصال بـ Gemini مع دعم JSON أو Text"""
    full_text = f"{system_prompt}\n\n{prompt}"
    
    config = {"temperature": 0.3}
    if json_mode:
        config["responseMimeType"] = "application/json"
        
    payload = {
        "contents": [{"parts": [{"text": full_text}]}],
        "generationConfig": config
    }
    
    for key in GEMINI_API_KEYS:
        if not key: continue
        try:
            url = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={key}"
            resp = requests.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code == 429:
                continue 
        except:
            continue
    return None

def _call_openrouter(prompt, system_prompt="", json_mode=True):
    """اتصال بـ OpenRouter"""
    if not OPENROUTER_API_KEY: return None
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=25)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except:
        return None
    return None

def call_ai_base(prompt, system_type="general", json_mode=False):
    """دالة موحدة للطلب"""
    sys_p = PROMPTS.get(system_type, PROMPTS["general"])
    
    # Gemini First
    res = _call_gemini(prompt, sys_p, json_mode)
    if res: return res
    
    # OpenRouter Fallback
    res = _call_openrouter(prompt, sys_p, json_mode)
    if res: return res
    
    return None

def call_ai_json(prompt, type="verify_batch"):
    """طلب JSON حصراً"""
    res = call_ai_base(prompt, type, json_mode=True)
    if res: return parse_json(res)
    return None

def parse_json(text):
    try:
        start = text.find('[')
        if start == -1: start = text.find('{')
        end = text.rfind(']') + 1
        if end == 0: end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return json.loads(text)
    except:
        return None

# ===== الوظائف المطلوبة في app.py =====

def chat_with_ai(message, history=[]):
    """الدردشة العامة (مطلوبة في app.py)"""
    context = ""
    if history:
        # أخذ آخر 3 ردود فقط لتوفير التوكنز
        for h in history[-3:]:
            context += f"User: {h.get('user','')}\nAI: {h.get('ai','')}\n"
    
    full_prompt = f"{context}\nUser: {message}"
    response = call_ai_base(full_prompt, "general", json_mode=False)
    
    if response:
        return {"success": True, "response": response, "source": "AI"}
    return {"success": False, "response": "فشل الاتصال", "source": "Error"}

def verify_single_match(our_name, comp_name, our_price, comp_price):
    """تحقق فردي"""
    prompt = f"""هل المنتجين متطابقين؟
    1. {our_name} ({our_price})
    2. {comp_name} ({comp_price})
    أجب JSON: {{"match": bool, "confidence": int, "reason": "string"}}"""
    
    res = call_ai_json(prompt, "verify_batch")
    if isinstance(res, list): res = res[0]
    
    if res:
        return {"success": True, **res}
    return {"success": False, "match": False, "reason": "فشل الاتصال"}

def smart_bulk_verify(rows):
    """التحقق الجماعي"""
    if not rows: return []
    lines = [f"ID:{r['id']} | Our:{r['our']} | Comp:{r['comp']}" for r in rows]
    prompt = "\n".join(lines)
    
    result = call_ai_json(prompt, "verify_batch")
    if result:
        lookup = {item['id']: item for item in result if 'id' in item}
        final = []
        for r in rows:
            d = lookup.get(r['id'])
            if d:
                r['ai_match'] = d.get('match')
                r['ai_reason'] = d.get('reason')
            final.append(r)
        return final
    return rows

def analyze_product(product_name, price=0):
    """تحليل منتج (مطلوبة في app.py)"""
    prompt = f"حلل المنتج: {product_name} سعره {price}. أعطني الماركة والنوع وتقييم السعر."
    res = call_ai_base(prompt, "general", json_mode=False)
    if res:
        return {"success": True, "response": res}
    return {"success": False, "response": "فشل"}

def suggest_price(product, current_price, comp_prices):
    """اقتراح سعر (كانت تسمى suggest_optimization)"""
    prices = ",".join(map(str, comp_prices))
    prompt = f"اقترح سعر للمنتج {product} (سعرنا {current_price})، المنافسين: {prices}. أجب بـ JSON: {{'suggested_price': float, 'strategy': 'str'}}"
    return call_ai_json(prompt, "verify_batch")
