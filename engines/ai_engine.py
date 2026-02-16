"""
ai_engine.py - محرك الذكاء الصناعي v17.1
- مفاتيح متعددة مع fallback تلقائي
- Gemini أولاً ثم OpenRouter
- تدريب مخصص لكل صفحة
"""
import requests, json, time
from config import GEMINI_API_KEYS, OPENROUTER_API_KEY

# ===== النماذج =====
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"

# ===== System Prompts لكل صفحة =====
PAGE_PROMPTS = {
    "price_raise": """أنت خبير تسعير عطور في السوق السعودي. مهمتك تحليل المنتجات التي سعرها أعلى من المنافسين.
لكل منتج: تحقق هل المطابقة صحيحة (نفس المنتج والحجم والماركة)، هل فرق السعر مبرر، وأعطِ توصية (خفض/إبقاء/مراجعة).
أجب بالعربية بشكل مختصر ومفيد.""",

    "price_lower": """أنت خبير تسعير عطور. مهمتك تحليل المنتجات التي سعرها أقل من المنافسين.
لكل منتج: تحقق هل يمكن رفع السعر، هل هناك فرصة ربح، وأعطِ توصية (رفع/إبقاء/مراجعة).
أجب بالعربية بشكل مختصر ومفيد.""",

    "approved": """أنت خبير تسعير عطور. مهمتك مراجعة المنتجات الموافق عليها.
تحقق أن التطابق صحيح وأن السعر مناسب للسوق.
أجب بالعربية بشكل مختصر.""",

    "missing": """أنت خبير عطور ومنتجات. مهمتك التحقق من المنتجات المفقودة (غير موجودة عندنا).
لكل منتج: ابحث هل هو منتج حقيقي، هل يستحق الإضافة للمتجر، السعر المقترح، وتأكد أنه ليس مكرراً بإسم مختلف.
أجب بالعربية بشكل مختصر ومفيد.""",

    "review": """أنت خبير تسعير عطور. مهمتك مراجعة المنتجات التي تحتاج مراجعة (تطابق غير مؤكد).
لكل منتج: تحقق هل المطابقة صحيحة أم خاطئة، وأعطِ توصية (نقل لموافق/نقل لمخفض/إزالة).
أجب بالعربية بشكل مختصر ومفيد.""",

    "general": """أنت مساعد ذكي متخصص في تسعير العطور والمنافسة في السوق السعودي.
تساعد في تحليل الأسعار والمنتجات والمنافسين واتخاذ القرارات.
أجب بالعربية بشكل احترافي ومفيد.""",

    "verify": """أنت خبير تحقق من منتجات العطور. مهمتك:
1. التحقق أن المنتجين متطابقين فعلاً (نفس الماركة، الاسم، الحجم، النوع)
2. التحقق من السعر المعقول في السوق
3. إعطاء نسبة ثقة (0-100%)
أجب بـ JSON فقط: {"match": true/false, "confidence": 0-100, "reason": "السبب", "suggestion": "التوصية"}""",

    "paste": """أنت مساعد ذكي. المستخدم سيلصق بيانات أو أوامر. حللها ونفذ المطلوب.
إذا كانت قائمة منتجات: صنفها وأعطِ توصيات.
إذا كانت أوامر: نفذها على البيانات المتاحة.
أجب بالعربية بشكل مختصر وعملي."""
}


def _call_gemini(prompt, system_prompt=""):
    """استدعاء Gemini API مع تجربة جميع المفاتيح"""
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048, "topP": 0.8}
    }
    for key in GEMINI_API_KEYS:
        if not key:
            continue
        try:
            url = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={key}"
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if "candidates" in data and data["candidates"]:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return text
            # إذا 429 (حصة منتهية) جرب المفتاح التالي
            elif resp.status_code == 429:
                continue
            # إذا خطأ آخر جرب التالي
            else:
                continue
        except:
            continue
    return None


def _call_openrouter(prompt, system_prompt=""):
    """استدعاء OpenRouter كـ fallback"""
    if not OPENROUTER_API_KEY:
        return None
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 2048}
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
        return None
    except:
        return None


def call_ai(prompt, page="general"):
    """استدعاء AI مع fallback تلقائي بين جميع المزودين"""
    system_prompt = PAGE_PROMPTS.get(page, PAGE_PROMPTS["general"])

    # محاولة 1: Gemini مباشر (يجرب جميع المفاتيح)
    result = _call_gemini(prompt, system_prompt)
    if result:
        return {"success": True, "response": result, "source": "Gemini"}

    # محاولة 2: OpenRouter
    result = _call_openrouter(prompt, system_prompt)
    if result:
        return {"success": True, "response": result, "source": "OpenRouter"}

    return {"success": False, "response": "فشل الاتصال بجميع مزودي AI. تحقق من المفاتيح أو الحصة.", "source": "none"}


def chat_with_ai(message, history=None, page="general"):
    """دردشة مع AI مع سياق المحادثة"""
    context = ""
    if history:
        for h in history[-5:]:
            context += f"المستخدم: {h['user']}\nAI: {h['ai']}\n"
    full_prompt = f"{context}\nالمستخدم: {message}" if context else message
    return call_ai(full_prompt, page)


def verify_match(our_product, comp_product, our_price=0, comp_price=0):
    """تحقق AI من صحة المطابقة"""
    prompt = f"""تحقق من تطابق هذين المنتجين:
منتجنا: {our_product} (السعر: {our_price} ر.س)
المنافس: {comp_product} (السعر: {comp_price} ر.س)

هل هما نفس المنتج؟ أجب بـ JSON فقط."""
    result = call_ai(prompt, "verify")
    if result["success"]:
        try:
            text = result["response"]
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                return {"success": True, **parsed, "source": result["source"]}
        except:
            pass
        return {"success": True, "match": None, "confidence": 0,
                "reason": result["response"][:300], "source": result["source"]}
    return {"success": False, "match": None, "confidence": 0, "reason": "فشل الاتصال"}


def analyze_product(product_name, price=0, context=""):
    """تحليل منتج واحد بالتفصيل"""
    prompt = f"""حلل هذا المنتج:
المنتج: {product_name}
السعر: {price} ر.س
السياق: {context}

أعطني: الماركة، الحجم، النوع، تقييم السعر، توصية."""
    return call_ai(prompt, "general")


def bulk_verify(products_list, page="review"):
    """تحقق جماعي من قائمة منتجات"""
    if not products_list:
        return {"success": False, "response": "لا توجد منتجات"}
    items = []
    for i, p in enumerate(products_list[:20]):
        items.append(f"{i+1}. منتجنا: {p.get('our', '')} ({p.get('our_price', 0)} ر.س) ↔ المنافس: {p.get('comp', '')} ({p.get('comp_price', 0)} ر.س)")
    prompt = f"""تحقق من هذه المطابقات وأعطِ توصية لكل واحدة:
{chr(10).join(items)}

لكل منتج أجب: ✅ صحيح / ❌ خطأ / ⚠️ غير متأكد + السبب"""
    return call_ai(prompt, page)


def suggest_price(product_name, current_price, comp_prices):
    """اقتراح سعر مناسب"""
    prices_text = ", ".join([f"{p} ر.س" for p in comp_prices if p > 0])
    prompt = f"""اقترح سعر مناسب لهذا المنتج:
المنتج: {product_name}
سعرنا الحالي: {current_price} ر.س
أسعار المنافسين: {prices_text}

أعطني السعر المقترح مع التبرير."""
    return call_ai(prompt, "general")


def process_paste(text, page="general"):
    """معالجة نص ملصوق من المستخدم"""
    prompt = f"""المستخدم لصق هذا النص في صفحة '{page}':
---
{text[:3000]}
---
حلل المحتوى وأعطِ النتائج والتوصيات المناسبة."""
    return call_ai(prompt, "paste")


def check_duplicate(product_name, existing_products):
    """التحقق من تكرار المنتج"""
    products_text = "\n".join(existing_products[:50])
    prompt = f"""هل هذا المنتج موجود بالفعل في القائمة التالية (ربما باسم مختلف)؟
المنتج: {product_name}

القائمة:
{products_text}

أجب: موجود/غير موجود + الاسم المطابق إن وجد."""
    return call_ai(prompt, "missing")
