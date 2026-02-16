"""
config.py - الإعدادات المركزية v17.1
المفاتيح محمية عبر Streamlit Secrets (لا تظهر في الكود)
"""
import streamlit as st

# ===== معلومات التطبيق =====
APP_TITLE = "نظام التسعير الذكي - مهووس"
APP_NAME = APP_TITLE
APP_VERSION = "v17.1"
APP_ICON = "🧪"

# ===== مفاتيح AI (محمية عبر st.secrets) =====
def _get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except:
        return default

GEMINI_API_KEYS = [
    _get_secret("GEMINI_KEY_1"),
    _get_secret("GEMINI_KEY_2"),
    _get_secret("GEMINI_KEY_3"),
]
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""
OPENROUTER_API_KEY = _get_secret("OPENROUTER_KEY")
EXTRA_API_KEY = _get_secret("EXTRA_API_KEY")

# ===== Make.com Webhooks (محمية أيضاً) =====
WEBHOOK_UPDATE_PRICES = _get_secret("WEBHOOK_UPDATE_PRICES", "https://hook.eu2.make.com/99oljy0d6r3chwg6bdfsptcf6bk8htsd")
WEBHOOK_NEW_PRODUCTS = _get_secret("WEBHOOK_NEW_PRODUCTS", "https://hook.eu2.make.com/xvubj23dmpxu8qzilstd25cnumrwtdxm")

# ===== ألوان =====
COLORS = {
    "raise": "#dc3545", "lower": "#ffc107", "approved": "#28a745",
    "missing": "#007bff", "review": "#ff9800", "primary": "#6C63FF",
}

# ===== إعدادات المطابقة =====
MATCH_THRESHOLD = 60
HIGH_CONFIDENCE = 95
REVIEW_THRESHOLD = 85
PRICE_TOLERANCE = 5
MIN_MATCH_SCORE = MATCH_THRESHOLD
HIGH_MATCH_SCORE = HIGH_CONFIDENCE
PRICE_DIFF_THRESHOLD = PRICE_TOLERANCE

# ===== استثناء العينات فقط =====
REJECT_KEYWORDS = [
    "sample", "عينة", "عينه", "decant", "تقسيم", "تقسيمة",
    "split", "miniature", "0.5ml", "1ml", "2ml", "3ml",
]
TESTER_KEYWORDS = ["tester", "تستر", "تيستر"]
SET_KEYWORDS = ["set", "gift set", "طقم", "مجموعة", "coffret"]

# ===== العلامات التجارية =====
KNOWN_BRANDS = [
    "Dior","Chanel","Gucci","Tom Ford","Versace","Armani","YSL","Prada",
    "Burberry","Givenchy","Hermes","Creed","Montblanc","Calvin Klein",
    "Hugo Boss","Dolce & Gabbana","Valentino","Bvlgari","Cartier","Lancome",
    "Jo Malone","Amouage","Rasasi","Lattafa","Arabian Oud","Ajmal",
    "Al Haramain","Afnan","Armaf","Nishane","Xerjoff","Parfums de Marly",
    "Initio","Byredo","Le Labo","Mancera","Montale","Kilian","Roja",
    "Carolina Herrera","Jean Paul Gaultier","Narciso Rodriguez",
    "Paco Rabanne","Mugler","Chloe","Coach","Michael Kors","Ralph Lauren",
    "لطافة","العربية للعود","رصاصي","أجمل","الحرمين","أرماف",
    "أمواج","كريد","توم فورد","ديور","شانيل","غوتشي","برادا",
]

# ===== تطبيع =====
WORD_REPLACEMENTS = {
    'او دو بارفان':'edp','أو دو بارفان':'edp','او دي بارفان':'edp',
    'او دو تواليت':'edt','أو دو تواليت':'edt','او دي تواليت':'edt',
    'مل':'ml','ملي':'ml','سوفاج':'sauvage','ديور':'dior','شانيل':'chanel',
}

# ===== الأقسام =====
SECTIONS = [
    "📊 لوحة التحكم", "📂 رفع الملفات",
    "🔴 سعر أعلى", "🟢 سعر أقل",
    "✅ موافق عليها", "🔍 منتجات مفقودة",
    "⚠️ تحت المراجعة", "📊 مقارنة بصرية",
    "🤖 الذكاء الصناعي", "⚡ أتمتة Make",
    "⚙️ الإعدادات", "📜 السجل",
]
SIDEBAR_SECTIONS = SECTIONS

PAGES_PER_TABLE = 25
DB_PATH = "perfume_pricing.db"
