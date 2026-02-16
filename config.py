"""
config.py - الإعدادات المركزية v16.0
"""
import os, json

APP_VERSION = "16.0"
APP_NAME = "نظام التسعير الذكي - مهووس"
APP_ICON = "🧪"

# ===== مفاتيح AI =====
GEMINI_API_KEY = "AIzaSyCM_7dJ-0mq4H81CHBYAIA1MkDbj8lk7Ko"
OPENROUTER_API_KEY = "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"

# ===== ألوان النظام =====
COLORS = {
    "raise": "#dc3545", "lower": "#ffc107", "approved": "#28a745",
    "missing": "#007bff", "review": "#ff9800", "primary": "#6C63FF",
}

# ===== إعدادات المطابقة =====
MATCH_THRESHOLD = 60
HIGH_CONFIDENCE = 95
REVIEW_THRESHOLD = 85
PRICE_TOLERANCE = 5

# ===== المنتجات المستثناة: العينات فقط =====
REJECT_KEYWORDS = [
    "sample", "عينة", "عينه", "decant", "تقسيم", "تقسيمة",
    "split", "miniature", "0.5ml", "1ml", "2ml", "3ml",
]

# ===== تصنيف المنتجات =====
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

PAGES_PER_TABLE = 25

SIDEBAR_SECTIONS = [
    ("🏠","لوحة القيادة"), ("📤","رفع الملفات"),
    ("🔴","رفع سعر"), ("🟡","خفض سعر"),
    ("🟢","موافق عليها"), ("🔵","منتجات مفقودة"),
    ("⚠️","يحتاج مراجعة"), ("🤖","تحقق AI"),
    ("💬","دردشة AI"), ("⚡","Make أتمتة"),
    ("💾","قاعدة البيانات"), ("⚙️","الإعدادات"),
]

DB_PATH = "perfume_pricing.db"
