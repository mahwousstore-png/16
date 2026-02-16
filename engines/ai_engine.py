"""
ai_engine.py - الذكاء الصناعي المباشر v16.0
يدعم: Gemini API + OpenRouter (كـ fallback)
"""
import requests, json
from config import GEMINI_API_KEY, OPENROUTER_API_KEY

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """أنت مساعد ذكي متخصص في تسعير العطور لمتجر "مهووس".
مهامك:
- تحليل المنتجات والمقارنة بين الأسعار
- التحقق من صحة المطابقة بين المنتجات
- تقديم توصيات تسعير ذكية
- الإجابة على أسئلة المستخدم بخصوص السوق والمنافسين
أجب بالعربية دائماً. كن دقيقاً ومختصراً."""


def call_gemini(prompt, context=""):
    """استدعاء Gemini API مباشرة"""
    try:
        body = {
            "contents": [{
                "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{context}\n\nالسؤال: {prompt}"}]
            }],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
        }
        r = requests.post(GEMINI_URL, json=body, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return None
    except Exception:
        return None


def call_openrouter(prompt, context=""):
    """استدعاء OpenRouter كـ fallback"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "google/gemini-2.0-flash-exp:free",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\n{prompt}"}
            ],
            "temperature": 0.3, "max_tokens": 2048,
        }
        r = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return None
    except Exception:
        return None


def chat_with_ai(prompt, context=""):
    """دردشة مع AI - يحاول Gemini أولاً ثم OpenRouter"""
    result = call_gemini(prompt, context)
    if result: return result
    result = call_openrouter(prompt, context)
    if result: return result
    return "⚠️ تعذر الاتصال بالذكاء الصناعي. تحقق من المفاتيح."


def verify_match(our_product, comp_product, match_score):
    """تحقق AI من صحة المطابقة"""
    prompt = f"""تحقق من صحة هذه المطابقة بين منتجين:
منتجنا: {our_product}
المنافس: {comp_product}
نسبة التطابق: {match_score}%

هل هذان نفس المنتج؟ أجب بـ:
- ✅ صحيح (نفس المنتج)
- ❌ خطأ (منتجان مختلفان)
- ⚠️ غير متأكد

مع سبب مختصر."""
    return chat_with_ai(prompt)


def analyze_product(product_info):
    """تحليل منتج بالذكاء الصناعي"""
    prompt = f"""حلل هذا المنتج وقدم توصية تسعير:
المنتج: {product_info.get('name','')}
سعرنا: {product_info.get('our_price',0)} ر.س
سعر المنافس: {product_info.get('comp_price',0)} ر.س
نسبة التطابق: {product_info.get('match_score',0)}%

قدم:
1. تقييم المطابقة
2. توصية السعر المثالي
3. استراتيجية التسعير"""
    return chat_with_ai(prompt)


def bulk_verify(products_list):
    """تحقق جماعي من قائمة منتجات"""
    if not products_list: return []
    summary = "\n".join([
        f"- {p.get('المنتج','')} ↔ {p.get('اسم المنافس','')} ({p.get('نسبة التطابق',0)}%)"
        for p in products_list[:20]
    ])
    prompt = f"""تحقق من صحة هذه المطابقات (أجب بجدول مختصر):
{summary}

لكل مطابقة أجب: ✅ صحيح / ❌ خطأ / ⚠️ غير متأكد"""
    return chat_with_ai(prompt)


def suggest_price(product_name, our_price, comp_prices):
    """اقتراح سعر ذكي"""
    prices_str = ", ".join([f"{p:.0f}" for p in comp_prices])
    prompt = f"""اقترح السعر المثالي لهذا المنتج:
المنتج: {product_name}
سعرنا: {our_price} ر.س
أسعار المنافسين: {prices_str} ر.س

الاستراتيجية: أقل من أقل منافس بريال واحد (إن أمكن)
قدم السعر المقترح مع التبرير."""
    return chat_with_ai(prompt)
