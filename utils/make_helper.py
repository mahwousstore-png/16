"""
make_helper.py - أتمتة Make.com (نسخة أصلية مع التوافق)
الدوال الأصلية من v15 + دوال التوافق مع v16
"""
import requests, json, hashlib

# ===== Webhooks الأصلية =====
WEBHOOK_UPDATE_PRICES = "https://hook.eu2.make.com/99oljy0d6r3chwg6bdfsptcf6bk8htsd"
WEBHOOK_NEW_PRODUCTS = "https://hook.eu2.make.com/xvubj23dmpxu8qzilstd25cnumrwtdxm"


# ══════════════════════════════════════════════════════════════
# دوال أصلية من v15
# ══════════════════════════════════════════════════════════════

def _extract_brand(name):
    """استخراج الماركة من اسم المنتج"""
    from config import KNOWN_BRANDS
    name_lower = name.lower()
    for brand in KNOWN_BRANDS:
        if brand.lower() in name_lower:
            return brand
    return "عام"


def _extract_category(name, product_type):
    """تحديد التصنيف بناءً على النوع"""
    type_lower = str(product_type).lower()
    if "رجال" in type_lower or "men" in type_lower:
        return "عطور رجالية"
    elif "نسائ" in type_lower or "women" in type_lower:
        return "عطور نسائية"
    elif "unisex" in type_lower or "مشترك" in type_lower:
        return "عطور مشتركة"
    return "عطور"


def send_to_webhook(webhook_url, payload):
    """إرسال البيانات إلى Webhook"""
    try:
        response = requests.post(webhook_url, json=payload, timeout=30)
        if response.status_code == 200:
            return {"success": True, "message": "تم الإرسال بنجاح"}
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def verify_webhook_connection(webhook_url, test_type="update"):
    """التحقق من اتصال Webhook - أصلي"""
    try:
        # Make.com webhooks ترد 200 على GET مع رسالة "Accepted"
        response = requests.get(webhook_url, timeout=15)
        if response.status_code == 200:
            return {"connected": True, "message": "متصل ويعمل", "status_code": 200}
        return {"connected": False, "message": f"HTTP {response.status_code}", "status_code": response.status_code}
    except Exception as e:
        return {"connected": False, "message": str(e)}


def send_price_updates_original(products):
    """إرسال تحديثات الأسعار إلى Make.com - أصلي من v15"""
    payload = {"data": []}
    for p in products:
        sku = p.get("sku", p.get("رمز المنتج", ""))
        price_raw = p.get("السعر الجديد", p.get("new_price", p.get("السعر", 0)))
        try:
            price = int(float(str(price_raw).replace(',', '')))
        except:
            price = 0
        if price <= 0:
            price = 1

        item = {
            "رمز المنتج sku": sku,
            "سعر المنتج": price,
        }
        payload["data"].append(item)

    return send_to_webhook(WEBHOOK_UPDATE_PRICES, payload)


def send_new_products_original(products):
    """إرسال منتجات جديدة إلى Make.com - أصلي من v15"""
    # تنسيق يتوافق مع Make.com blueprint:
    # Iterator يستخدم {{1.data}} وSalla CreateProduct يستخدم أسماء عربية
    payload = {"data": []}
    for p in products:
        name = p.get("المنتج", p.get("name", ""))
        price_raw = p.get("السعر", p.get("price", p.get("أقل سعر منافس", 0)))
        try:
            price = int(float(str(price_raw).replace(',', '')))
        except:
            price = 0
        if price <= 0:
            price = 1  # سلة لا تقبل سعر 0

        # توليد SKU تلقائي إذا كان فارغاً
        sku = p.get("sku", p.get("رمز المنتج", ""))
        if not sku:
            sku = f"PERF-{hashlib.md5(name.encode()).hexdigest()[:8].upper()}"

        # استخراج الماركة تلقائياً إذا كانت فارغة
        brand = p.get("الماركة", p.get("brand", ""))
        if not brand:
            brand = _extract_brand(name)

        # تصنيف افتراضي إذا كان فارغاً - يستخدم 88 تصنيف من سلة
        category = p.get("التصنيف", p.get("category", ""))
        if not category:
            p_type = p.get("النوع", p.get("type", ""))
            category = _extract_category(name, str(p_type))

        # بناء الوصف
        desc = p.get("الوصف", p.get("description", ""))
        if not desc:
            desc = f"{name} - {p.get('النوع', p.get('type', ''))} - {p.get('الحجم', p.get('size', ''))}"

        # بناء البيانات بتنسيق يتوافق مع Salla API عبر Make.com
        # الحقول المدعومة في blueprint: أسم المنتج, سعر المنتج, رمز المنتج sku, الوزن, سعر التكلفة, السعر المخفض, الوصف
        # ملاحظة: categories و brand_id يحتاجان ID رقمي من سلة وليس اسم نصي
        item = {
            "أسم المنتج": name,
            "سعر المنتج": price,
            "رمز المنتج sku": sku,
            "الوزن": 1,
            "الوصف": desc,
        }

        # لا نرسل سعر التكلفة والسعر المخفض إذا كانا 0 لتجنب أخطاء سلة
        cost = p.get("سعر التكلفة", p.get("cost_price", 0))
        if cost and int(float(str(cost))) > 0:
            item["سعر التكلفة"] = int(float(str(cost)))

        sale = p.get("السعر المخفض", p.get("sale_price", 0))
        if sale and int(float(str(sale))) > 0:
            item["السعر المخفض"] = int(float(str(sale)))

        payload["data"].append(item)

    return send_to_webhook(WEBHOOK_NEW_PRODUCTS, payload)


# ══════════════════════════════════════════════════════════════
# دوال التوافق مع app.py v16 (تستدعي الدوال الأصلية)
# ══════════════════════════════════════════════════════════════

def send_to_make(webhook_url, data):
    """متوافق مع v16 - إرسال عام"""
    try:
        r = requests.post(webhook_url, json=data, timeout=15)
        return {"success": r.status_code == 200, "message": f"تم الإرسال ({r.status_code})"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def send_price_updates(webhook_url, df):
    """متوافق مع v16 - يقبل webhook_url و DataFrame"""
    if hasattr(df, 'empty') and df.empty:
        return {"success": False, "message": "لا توجد بيانات"}
    if hasattr(df, 'to_dict'):
        records = df.head(50).to_dict('records')
    else:
        records = df if isinstance(df, list) else []
    return send_to_make(webhook_url, {"type": "price_update", "products": records, "count": len(records)})


def send_missing_products(webhook_url, df):
    """متوافق مع v16 - إرسال منتجات مفقودة"""
    if hasattr(df, 'empty') and df.empty:
        return {"success": False, "message": "لا توجد بيانات"}
    if hasattr(df, 'to_dict'):
        records = df.head(50).to_dict('records')
    else:
        records = df if isinstance(df, list) else []
    return send_to_make(webhook_url, {"type": "missing_products", "products": records, "count": len(records)})


def test_webhook(webhook_url):
    """متوافق مع v16 - اختبار Webhook"""
    try:
        r = requests.post(webhook_url, json={"test": True, "source": "mahwous-erp-v16"}, timeout=10)
        return r.status_code == 200
    except:
        return False
