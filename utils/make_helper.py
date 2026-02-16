"""
make_helper.py - أتمتة Make.com v16.0
"""
import requests, json


def send_to_make(webhook_url, data):
    try:
        r = requests.post(webhook_url, json=data, timeout=15)
        return {"success": r.status_code == 200, "message": f"تم الإرسال ({r.status_code})"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def send_price_updates(webhook_url, df):
    if df.empty: return {"success": False, "message": "لا توجد بيانات"}
    records = df.head(50).to_dict('records')
    return send_to_make(webhook_url, {"type": "price_update", "products": records, "count": len(records)})


def send_missing_products(webhook_url, df):
    if df.empty: return {"success": False, "message": "لا توجد بيانات"}
    records = df.head(50).to_dict('records')
    return send_to_make(webhook_url, {"type": "missing_products", "products": records, "count": len(records)})


def test_webhook(webhook_url):
    try:
        r = requests.post(webhook_url, json={"test": True, "source": "mahwous-erp-v16"}, timeout=10)
        return r.status_code == 200
    except:
        return False
