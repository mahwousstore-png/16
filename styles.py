"""
styles.py - التصميم v16.0 (خفيف وسريع)
"""

def get_main_css():
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');
*{font-family:'Tajawal',sans-serif!important}
.main .block-container{max-width:1400px;padding:1rem 2rem}

.stat-card{background:#1A1A2E;border-radius:14px;padding:18px;text-align:center;border:1px solid #333344;transition:all .3s}
.stat-card:hover{transform:translateY(-3px);box-shadow:0 6px 24px rgba(108,99,255,.2);border-color:#6C63FF}
.stat-card .num{font-size:2.5rem;font-weight:900;margin:6px 0;line-height:1}
.stat-card .lbl{font-size:.9rem;color:#8B8B8B;font-weight:500}

.cmp-table{width:100%;border-collapse:separate;border-spacing:0;border-radius:10px;overflow:hidden;font-size:.92rem}
.cmp-table thead th{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:12px 10px;font-weight:700;text-align:center;border-bottom:2px solid #6C63FF;position:sticky;top:0;z-index:10}
.cmp-table tbody tr{transition:all .2s}
.cmp-table tbody tr:nth-child(even){background:rgba(26,26,46,.5)}
.cmp-table tbody tr:hover{background:rgba(108,99,255,.12)!important}
.cmp-table td{padding:10px;text-align:center;border-bottom:1px solid rgba(51,51,68,.5);vertical-align:middle}
.td-our{background:rgba(108,99,255,.08)!important;border-right:3px solid #6C63FF;text-align:right!important;font-weight:600;color:#B8B4FF}
.td-comp{background:rgba(255,152,0,.08)!important;border-left:3px solid #ff9800;text-align:right!important;font-weight:600;color:#FFD180}

.badge{display:inline-block;padding:3px 10px;border-radius:16px;font-size:.78rem;font-weight:700}
.b-high{background:rgba(255,23,68,.18);color:#FF1744;border:1px solid #FF1744}
.b-med{background:rgba(255,214,0,.18);color:#FFD600;border:1px solid #FFD600}
.b-low{background:rgba(0,200,83,.18);color:#00C853;border:1px solid #00C853}

.conf-bar{width:100%;height:7px;background:rgba(255,255,255,.1);border-radius:4px;overflow:hidden}
.conf-fill{height:100%;border-radius:4px;transition:width .4s}

.vs-row{display:grid;grid-template-columns:1fr 40px 1fr;gap:12px;align-items:center;padding:14px;background:#1A1A2E;border-radius:10px;margin:6px 0;border:1px solid #333344}
.vs-badge{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:.75rem}
.our-s{text-align:right;padding:10px;background:rgba(108,99,255,.06);border-radius:8px;border-right:3px solid #6C63FF}
.comp-s{text-align:left;padding:10px;background:rgba(255,152,0,.06);border-radius:8px;border-left:3px solid #ff9800}

section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0E1117,#1A1A2E)}
#MainMenu,footer,header{visibility:hidden}
</style>
"""

def stat_card(icon, label, value, color="#6C63FF"):
    return f'<div class="stat-card" style="border-top:3px solid {color}"><div style="font-size:1.4rem">{icon}</div><div class="num" style="color:{color}">{value}</div><div class="lbl">{label}</div></div>'

def vs_card(our_name, our_price, comp_name, comp_price, diff):
    dc = "#FF1744" if diff > 0 else "#00C853" if diff < 0 else "#FFD600"
    return f'''<div class="vs-row">
<div class="our-s"><div style="font-size:.75rem;color:#8B8B8B">منتجنا</div><div style="font-weight:700;color:#B8B4FF">{our_name}</div><div style="font-size:1.2rem;font-weight:900;color:#6C63FF;margin-top:3px">{our_price:.0f} ر.س</div></div>
<div class="vs-badge">VS</div>
<div class="comp-s"><div style="font-size:.75rem;color:#8B8B8B">المنافس</div><div style="font-weight:700;color:#FFD180">{comp_name}</div><div style="font-size:1.2rem;font-weight:900;color:#ff9800;margin-top:3px">{comp_price:.0f} ر.س</div></div>
</div><div style="text-align:center;margin:3px 0"><span style="color:{dc};font-weight:700">الفرق: {diff:+.0f} ر.س</span></div>'''
