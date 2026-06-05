# ╔══════════════════════════════════════════════════════════════════════╗
# ║      HỆ THỐNG CHẤM THI TRẮC NGHIỆM  –  app.py  v4 (final)        ║
# ╚══════════════════════════════════════════════════════════════════════╝
import sys, os, io, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    PDF_OK = True
except ImportError:
    PDF_OK = False

# ── OMR modules ──────────────────────────────────────────────────────────────
try:
    from OMR2020 import Omr_2020
    from OMR2025 import Omr_2025
    from V_ACT   import Omr_vact
    from OmrMDD  import readMDD_2020, readMDD_2025, readMDD_DGNL
    OMR_OK = True;  _omr_err = ""
except Exception as _e:
    OMR_OK = False; _omr_err = str(_e)

# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="OMR – Chấm Thi", page_icon="📝",
                   layout="wide", initial_sidebar_state="expanded")

# ═════════════════════════════════════════════════════════════════════════════
# CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* Streamlit 1.58: chỉ target các class nội dung, không dùng wildcard element selector
   vì nó khiến Streamlit render lại label widget → chữ bị chồng */
.stMarkdown, .stText, .element-container p,
.stButton > button, .stSelectbox label,
.stRadio label, .stCheckbox label,
.stFileUploader label, .stExpander summary {
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* ── app bg ── */
[data-testid="stAppViewContainer"] { background: #0c1118 !important; }
.main .block-container {
    background: transparent !important;
    padding-top: .8rem; padding-bottom: 2rem; max-width: 1440px;
}

/* ── sidebar: chỉ đổi bg, KHÔNG override màu chữ wildcard ── */
[data-testid="stSidebar"] > div:first-child {
    background: #111827 !important;
    border-right: 1px solid #1e2d3f;
}
/* ẩn tooltip keyboard_... khi hover vào widget sidebar */
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] { display: none !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827; border-radius: 8px 8px 0 0; gap: 2px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 6px;
    font-size: .85rem; font-weight: 500; padding: .4rem 1.1rem;
}
.stTabs [aria-selected="true"] { background: #1e3a5f !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: #111827 !important; border: 1px solid #1e2d3f;
    border-top: none; border-radius: 0 0 8px 8px; padding: 1.2rem;
}

/* ── card ── */
.card {
    background: #161e2e; border: 1px solid #1e2d3f;
    border-radius: 8px; padding: .9rem 1.1rem; margin-bottom: .7rem;
}
.card-title {
    font-size: .68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #3b82f6; margin-bottom: .5rem;
}

/* ── page header ── */
.page-title {
    font-size: 1.5rem; font-weight: 700; color: #f1f5f9;
    border-left: 4px solid #3b82f6; padding-left: .75rem; margin-bottom: .1rem;
}
.page-sub { font-size: .8rem; color: #475569; padding-left: .2rem; margin-bottom: .8rem; }

/* ── stat card ── */
.stat-card {
    background: #161e2e; border: 1px solid #1e2d3f;
    border-radius: 8px; padding: .8rem; text-align: center;
}
.stat-val { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.stat-lbl { font-size: .69rem; color: #475569; margin-top: 2px;
            text-transform: uppercase; letter-spacing: .6px; }

/* ── result table ── */
.rtable { width: 100%; border-collapse: collapse; font-size: .82rem; }
.rtable th {
    background: #1a2535; color: #4b6a8a; padding: 7px 10px; text-align: left;
    border-bottom: 1px solid #1e2d3f; font-size: .7rem;
    text-transform: uppercase; letter-spacing: .7px;
}
.rtable td { padding: 6px 10px; border-bottom: 1px solid #161e2e;
             color: #94a3b8; vertical-align: middle; }
.rtable tr:hover td { background: #1a2535; }

/* ── chips ── */
.chip { display: inline-block; font-family: 'IBM Plex Mono', monospace;
        font-weight: 600; font-size: .88rem; padding: 2px 9px; border-radius: 4px; }
.chip-hi  { background: rgba(34,197,94,.15); color: #4ade80; }
.chip-mid { background: rgba(251,191,36,.12); color: #fbbf24; }
.chip-lo  { background: rgba(239,68,68,.13); color: #f87171; }

/* ── answer items ── */
.ans-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(62px, 1fr));
    gap: 4px; margin-top: .4rem;
}
.ai {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .71rem; font-weight: 600;
    padding: 4px 5px; border-radius: 4px;
    text-align: center; white-space: nowrap;
    line-height: 1.35;
}
/* đúng  */
.ai-ok   { background: rgba(34,197,94,.13);  color: #4ade80;
           border: 1px solid rgba(34,197,94,.25); }
/* sai   */
.ai-err  { background: rgba(239,68,68,.15);  color: #f87171;
           border: 1px solid rgba(239,68,68,.28); }
/* bỏ trống */
.ai-skip { background: rgba(251,191,36,.11); color: #fbbf24;
           border: 1px solid rgba(251,191,36,.22); }
/* tô nhiều */
.ai-mul  { background: rgba(99,102,241,.14); color: #a5b4fc;
           border: 1px solid rgba(99,102,241,.28); }
/* số câu nhỏ hơn, mờ hơn */
.ai .qn  { font-size: .62rem; opacity: .65; display: block; margin-bottom: 1px; }

/* ── banners ── */
.bk { background: rgba(34,197,94,.08);  border-left: 3px solid #22c55e;
      border-radius: 4px; padding: .45rem .8rem; color: #86efac;
      font-size: .82rem; margin: .35rem 0; }
.be { background: rgba(239,68,68,.08);  border-left: 3px solid #ef4444;
      border-radius: 4px; padding: .45rem .8rem; color: #fca5a5;
      font-size: .82rem; margin: .35rem 0; }
.bi { background: rgba(59,130,246,.08); border-left: 3px solid #3b82f6;
      border-radius: 4px; padding: .45rem .8rem; color: #93c5fd;
      font-size: .82rem; margin: .35rem 0; }
.bw { background: rgba(251,191,36,.08); border-left: 3px solid #fbbf24;
      border-radius: 4px; padding: .45rem .8rem; color: #fde68a;
      font-size: .82rem; margin: .35rem 0; }

/* ── bar chart ── */
.bar-wrap { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
.bar-lbl  { width: 52px; font-size: .73rem; color: #64748b; text-align: right; }
.bar-body { height: 14px; border-radius: 3px; min-width: 4px; }
.bar-cnt  { font-size: .73rem; color: #94a3b8; }

/* ── info grid ── */
.igrid { display: grid; grid-template-columns: auto 1fr;
         gap: 5px 14px; font-size: .84rem; }
.ik { color: #475569; font-weight: 500; white-space: nowrap; }
.iv { color: #cbd5e1; }

/* ── ans legend ── */
.ans-legend { font-size: .72rem; color: #475569; margin-bottom: .4rem; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
LOAI_PHIEU = {
    "2020 – 120 câu":         "2020",
    "2025 – 3 phần (40+8+6)": "2025",
    "ĐGNL – 120 câu":         "dgnl",
}
MP_CHR = {0:"A", 1:"B", 2:"C", 3:"D"}
MP_INT = {"A":0, "B":1, "C":2, "D":3}

# ═════════════════════════════════════════════════════════════════════════════
# PARSER – ĐÁP ÁN EXCEL
# ═════════════════════════════════════════════════════════════════════════════
def _blank(v) -> bool:
    return str(v).strip().lower() in ("", "nan", "none", "n/a", "na")

def _is_header_row(raw_ans: list) -> bool:
    """Trả về True nếu đây là hàng header (toàn chữ không phải A/B/C/D)."""
    valid = [v for v in raw_ans if v.strip()]
    if not valid:
        return True
    return all(v not in MP_INT for v in valid)

def parse_excel_2020(df: pd.DataFrame) -> dict:
    """
    Cột 0 = mã đề, cột 1..N = đáp án A/B/C/D.
    Tự bỏ qua hàng tiêu đề lọt vào data.
    """
    result = {}
    cols   = list(df.columns)
    id_col, ans_cols = cols[0], cols[1:]
    for _, row in df.iterrows():
        ma = str(row[id_col]).strip()
        if _blank(ma):
            continue
        raw = [str(row[c]).strip().upper() for c in ans_cols]
        if _is_header_row(raw):          # bỏ hàng header lọt vào
            continue
        ans = [MP_INT.get(v, 0) for v in raw]
        if ans:
            result[ma] = ans
    return result

def parse_excel_2025(sheets: dict) -> dict:
    def get_sheet(*names):
        for n in names:
            if n in sheets:
                df = sheets[n].copy()
                df.columns = [str(c).strip() for c in df.columns]
                return df
        return None

    df1 = get_sheet("PHAN1", "phan1", "Phần 1", "PART1")
    df2 = get_sheet("PHAN2", "phan2", "Phần 2", "PART2")
    df3 = get_sheet("PHAN3", "phan3", "Phần 3", "PART3")
    if df1 is None or df2 is None or df3 is None:
        return {}

    TF = {"ĐÚNG":0,"ĐUNG":0,"DUNG":0,"TRUE":0,"T":0,"1":0,
          "SAI":1,"FALSE":1,"F":1,"0":1}
    result = {}
    for _, row1 in df1.iterrows():
        ma = str(row1[df1.columns[0]]).strip()
        if _blank(ma): continue
        raw1 = [str(row1[c]).strip().upper() for c in df1.columns[1:41]]
        if _is_header_row(raw1): continue
        p1 = [MP_INT.get(v, 0) for v in raw1]

        r2 = df2[df2[df2.columns[0]].astype(str).str.strip() == ma]
        if not len(r2): continue
        r2 = r2.iloc[0]
        raw2 = [str(r2[c]).strip().upper() for c in df2.columns[1:33]]
        p2 = [[TF.get(raw2[x*4+y], 0) for y in range(4)] for x in range(8)]

        r3 = df3[df3[df3.columns[0]].astype(str).str.strip() == ma]
        if not len(r3): continue
        r3 = r3.iloc[0]
        p3 = [str(r3[c]).strip() for c in df3.columns[1:7]]

        result[ma] = [p1, p2, p3]
    return result

def load_ans_excel(file, loai: str) -> dict:
    try:
        sheets = pd.read_excel(file, sheet_name=None, dtype=str)
    except Exception:
        return {}
    if loai == "2025":
        return parse_excel_2025(sheets)
    result = {}
    for _, df in sheets.items():
        df.columns = [str(c).strip() for c in df.columns]
        result.update(parse_excel_2020(df))
    return result

# ═════════════════════════════════════════════════════════════════════════════
# PARSER – DANH SÁCH SV
# ═════════════════════════════════════════════════════════════════════════════
def load_sv_excel(file) -> pd.DataFrame:
    try:
        df = pd.read_excel(file, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        for col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()

def guess_id_col(df: pd.DataFrame) -> str:
    if df is None or not len(df): return ""
    for col in df.columns:
        cl = col.lower().replace(" ", "").replace("_","")
        if any(k in cl for k in ["masv","mssv","msv","mãsv","id","sbd","sobao","mãsinh"]):
            return col
    return df.columns[0]

# ═════════════════════════════════════════════════════════════════════════════
# OMR WRAPPERS
# ═════════════════════════════════════════════════════════════════════════════
def img_bytes_to_np(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def np_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def run_mdd(img_np, loai):
    """→ (img_with_mdd_overlay, sbd_str, made_str)"""
    if not OMR_OK:
        return img_np.copy(), "xxxxxxxx", "000"
    if loai == "2020": return readMDD_2020(img_np)
    if loai == "2025": return readMDD_2025(img_np)
    return readMDD_DGNL(img_np)

def run_omr(img_np, loai, ans):

    if not OMR_OK:
        raise RuntimeError(
            "OMR modules failed to load. Check project files."
        )

    loai = loai.lower()

    if loai == "2020":
        img_graded, score, my_index = Omr_2020(
            img_np.copy(),
            ans
        )

    elif loai == "2025":
        img_graded, score, my_index = Omr_2025(
            img_np.copy(),
            ans
        )

    elif loai in ["dgnl", "vact"]:
        img_graded, score, my_index = Omr_vact(
            img_np.copy(),
            ans
        )

    else:
        raise ValueError(
            f"Loại phiếu không hỗ trợ: {loai}"
        )

    return img_graded, score, my_index

# ═════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═════════════════════════════════════════════════════════════════════════════
def _border():
    s = Side(style="thin", color="BDD7EE")
    return Border(left=s, right=s, top=s, bottom=s)

def build_excel_result(sv_df: pd.DataFrame, results: list) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Kết Quả"
    thin = _border()
    ctr  = Alignment(horizontal="center", vertical="center")
    lft  = Alignment(horizontal="left",   vertical="center")

    sv_cols  = list(sv_df.columns) if (sv_df is not None and len(sv_df)) else ["Số báo danh"]
    all_cols = sv_cols + ["Mã đề", "Điểm", "Ghi chú"]
    lc       = get_column_letter(len(all_cols))
    scores   = [float(r.get("diem", 0)) for r in results]
    avg      = sum(scores)/len(scores) if scores else 0

    # Row 1 – tiêu đề
    ws.merge_cells(f"A1:{lc}1")
    c = ws.cell(1, 1, "KẾT QUẢ CHẤM THI TRẮC NGHIỆM")
    c.font  = Font(name="Arial", bold=True, size=14, color="1F4E79")
    c.alignment = ctr
    c.fill  = PatternFill("solid", fgColor="EBF3FB")
    ws.row_dimensions[1].height = 28

    # Row 2 – tóm tắt
    ws.merge_cells(f"A2:{lc}2")
    c2 = ws.cell(2, 1,
        f"Tổng: {len(results)}  |  TB: {avg:.2f}  |  "
        f"Cao: {max(scores):.2f}  |  Thấp: {min(scores):.2f}")
    c2.font = Font(name="Arial", bold=True, size=10, color="1F4E79")
    c2.alignment = ctr
    c2.fill = PatternFill("solid", fgColor="DBEAFE")
    ws.row_dimensions[2].height = 17

    # Row 3 – header
    for ci, col in enumerate(all_cols, 1):
        c = ws.cell(3, ci, col)
        c.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill = PatternFill("solid", fgColor="1F4E79")
        c.alignment = ctr; c.border = thin
    ws.row_dimensions[3].height = 20
    ws.freeze_panes = "A4"

    # Data rows
    for ri, rec in enumerate(results, 4):
        rf = PatternFill("solid", fgColor="EBF3FB" if ri%2==0 else "FFFFFF")
        bf = Font(name="Arial", size=10)
        for ci, col in enumerate(sv_cols, 1):
            val = str(rec.get(col, rec.get("sbd", "")))
            c = ws.cell(ri, ci, val)
            c.font=bf; c.fill=rf; c.border=thin
            c.alignment = ctr if ci == 1 else lft

        c = ws.cell(ri, len(sv_cols)+1, rec.get("ma_de",""))
        c.font=bf; c.fill=rf; c.border=thin; c.alignment=ctr

        diem = float(rec.get("diem", 0))
        c = ws.cell(ri, len(sv_cols)+2, round(diem, 2))
        c.border=thin; c.alignment=ctr; c.number_format="0.00"
        if diem >= 8:
            c.fill=PatternFill("solid", fgColor="DCFCE7")
            c.font=Font(name="Arial", bold=True, color="166534", size=10)
        elif diem >= 5:
            c.fill=PatternFill("solid", fgColor="FEF9C3")
            c.font=Font(name="Arial", bold=True, color="713F12", size=10)
        else:
            c.fill=PatternFill("solid", fgColor="FEE2E2")
            c.font=Font(name="Arial", bold=True, color="991B1B", size=10)

        # Cột Ghi chú: ghi SBD nếu không tìm thấy trong DSSV
        note_val = (f"SBD {rec.get('sbd','')} không có trong DS"
                    if rec.get("_not_in_dssv") else "")
        c_note = ws.cell(ri, len(sv_cols)+3, note_val)
        c_note.font = Font(name="Arial", size=10,
                           color="B45309" if note_val else "64748B")
        c_note.fill = rf
        c_note.border = thin
        c_note.alignment = lft

    # Column widths
    for ci, col in enumerate(all_cols, 1):
        if col == "Ghi chú":
            ws.column_dimensions[get_column_letter(ci)].width = 36
        else:
            vals = [str(col)] + [str(rec.get(col, rec.get("sbd",""))) for rec in results]
            ws.column_dimensions[get_column_letter(ci)].width = min(max(len(v) for v in vals)+4, 40)

    buf = io.BytesIO(); wb.save(buf); return buf.getvalue()

def build_pdf_result(sv_df, results, loai_label) -> bytes:
    if not PDF_OK: return b""
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              leftMargin=28, rightMargin=28,
                              topMargin=36, bottomMargin=28)
    stys = getSampleStyleSheet()
    T    = ParagraphStyle("T", parent=stys["Heading1"], fontSize=14,
                           textColor=colors.HexColor("#1e3a5f"), alignment=1, spaceAfter=4)
    S    = ParagraphStyle("S", parent=stys["Normal"], fontSize=9,
                           textColor=colors.HexColor("#64748b"), alignment=1, spaceAfter=10)
    N    = ParagraphStyle("N", parent=stys["Normal"], fontSize=8.5, leading=11)

    sv_cols  = list(sv_df.columns) if (sv_df is not None and len(sv_df)) else ["Số báo danh"]
    all_cols = sv_cols + ["Mã đề", "Điểm", "Ghi chú"]
    scores   = [float(r.get("diem", 0)) for r in results]
    avg      = sum(scores)/len(scores) if scores else 0

    elems = [
        Paragraph("KẾT QUẢ CHẤM THI TRẮC NGHIỆM", T),
        Paragraph(
            f"Phiếu: {loai_label}  |  Tổng: {len(results)}  |  "
            f"TB: {avg:.2f}  |  Cao: {max(scores):.2f}  |  Thấp: {min(scores):.2f}", S),
    ]
    hdr  = [Paragraph(f"<b>{c}</b>", N) for c in all_cols]
    data = [hdr]
    for rec in results:
        row = [Paragraph(str(rec.get(c, rec.get("sbd",""))), N) for c in sv_cols]
        row.append(Paragraph(str(rec.get("ma_de","")), N))
        row.append(Paragraph(f"{float(rec.get('diem',0)):.2f}", N))
        note = (f"SBD {rec.get('sbd','')} không có trong DS"
                if rec.get("_not_in_dssv") else "")
        row.append(Paragraph(note, N))
        data.append(row)

    pw  = A4[0] - 56
    note_w = 80
    cw  = [(pw - 50 - 42 - note_w) / max(len(sv_cols), 1)] * len(sv_cols) + [50, 42, note_w]
    tbl = Table(data, colWidths=cw, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.HexColor("#EBF3FB"), colors.white]),
        ("ALIGN",         (-1,1),(-1,-1),"CENTER"),
        ("GRID",          (0,0),(-1,-1),.4, colors.HexColor("#BDD7EE")),
        ("TOPPADDING",    (0,0),(-1,-1),3),
        ("BOTTOMPADDING", (0,0),(-1,-1),3),
    ]))
    for ri, rec in enumerate(results, 1):
        d  = float(rec.get("diem", 0))
        bg = (colors.HexColor("#DCFCE7") if d>=8
              else colors.HexColor("#FEF9C3") if d>=5
              else colors.HexColor("#FEE2E2"))
        tbl.setStyle(TableStyle([("BACKGROUND",(-1,ri),(-1,ri), bg)]))

    elems.append(tbl)
    doc.build(elems)
    return buf.getvalue()

# ═════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def score_chip(s) -> str:
    s = float(s)
    cls = "chip-hi" if s>=8 else ("chip-mid" if s>=5 else "chip-lo")
    return f'<span class="chip {cls}">{s:.2f}</span>'

def render_ans_2020(ans_key: list, my_index):
    """
    Hiển thị từng câu dạng:
        <số câu>
        <HS>/<ĐA>

    Màu:
        xanh lá  — đúng        (HS == ĐA)
        đỏ       — sai         (HS ≠ ĐA, HS không phải -1/-2)
        vàng     — bỏ trống    (HS == -1)
        tím      — tô nhiều    (HS == -2)

    Nếu my_index=None (chưa tính được): chỉ hiển thị đáp án chuẩn, không phân loại.
    """
    n = len(ans_key)
    items = ""
    for i in range(n):
        correct   = ans_key[i]
        c_chr     = MP_CHR.get(correct, str(correct))   # đáp án chuẩn
        q_num     = str(i + 1)

        if my_index is None:
            # Chưa có đáp án học sinh → chỉ hiện đáp án chuẩn
            items += (f'<span class="ai ai-ok">'
                      f'<span class="qn">{q_num}</span>'
                      f'—/{c_chr}'
                      f'</span>')
        else:
            s     = my_index[i] if i < len(my_index) else -1
            s_chr = MP_CHR.get(s, "?") if s >= 0 else ("–" if s == -1 else "?")

            if s == -1:                          # bỏ trống
                css  = "ai-skip"
                body = f"–/{c_chr}"
            elif s == -2:                        # tô nhiều
                css  = "ai-mul"
                body = f"?/{c_chr}"
            elif s == correct:                   # đúng
                css  = "ai-ok"
                body = f"{c_chr}/{c_chr}"
            else:                                # sai
                css  = "ai-err"
                body = f"{s_chr}/{c_chr}"

            items += (f'<span class="ai {css}">'
                      f'<span class="qn">{q_num}</span>'
                      f'{body}'
                      f'</span>')

    st.markdown(f'<div class="ans-row">{items}</div>', unsafe_allow_html=True)

def render_ans_2025(ans_key: list, my_index):
    """
    Hiển thị đáp án 3 phần theo format HS/ĐA — giống render_ans_2020.

    ans_key = [p1, p2, p3]
      p1: list 40 int (0-3)
      p2: list 8 x list 4 int (0=Đúng, 1=Sai)
      p3: list 6 str (chuỗi ký tự '-,0-9')

    my_index = (myIndex_1, myIndex_2, myIndex_3) hoặc None
      myIndex_1: list 40 int (0-3, -1, -2)
      myIndex_2: list 8 x list 4 int (0-3, -1, -2)  — mỗi ý a/b/c/d
      myIndex_3: list 6 x list int — index ký tự theo char_to_index
    """
    # Mapping index → ký tự hiển thị cho Part 3
    # char_to_index: '-'→0, ','→1, '0'→2, '1'→3, ..., '9'→11
    IDX_TO_CHAR = {0:'-', 1:',', 2:'0', 3:'1', 4:'2', 5:'3',
                   6:'4', 7:'5', 8:'6', 9:'7', 10:'8', 11:'9'}

    p1_key, p2_key, p3_key = ans_key
    mi1 = my_index[0] if my_index else None  # list 40
    mi2 = my_index[1] if my_index else None  # list 8 x list 4
    mi3 = my_index[2] if my_index else None  # list 6 x list int

    # ── PHẦN 1 ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="card-title" style="margin-top:.6rem">' +
        '📌 PHẦN 1 — Trắc nghiệm (40 câu · ×0.25đ/câu)</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="ans-legend">' +
        '<span style="color:#4ade80">■</span> Đúng &nbsp;' +
        '<span style="color:#f87171">■</span> Sai &nbsp;' +
        '<span style="color:#fbbf24">■</span> Bỏ trống &nbsp;' +
        '<span style="color:#a5b4fc">■</span> Tô nhiều &nbsp;' +
        '&nbsp;—&nbsp; format: <code>HS/ĐA</code></div>',
        unsafe_allow_html=True
    )
    items = ""
    for i, correct in enumerate(p1_key):
        c_chr = MP_CHR.get(correct, str(correct))
        q_num = str(i + 1)
        if mi1 is None:
            items += (f'<span class="ai ai-ok">' +
                      f'<span class="qn">{q_num}</span>—/{c_chr}</span>')
        else:
            s     = mi1[i] if i < len(mi1) else -1
            s_chr = MP_CHR.get(s, "?") if s >= 0 else ("–" if s == -1 else "?")
            if s == -1:
                css, body = "ai-skip", f"–/{c_chr}"
            elif s == -2:
                css, body = "ai-mul",  f"?/{c_chr}"
            elif s == correct:
                css, body = "ai-ok",   f"{c_chr}/{c_chr}"
            else:
                css, body = "ai-err",  f"{s_chr}/{c_chr}"
            items += (f'<span class="ai {css}">' +
                      f'<span class="qn">{q_num}</span>{body}</span>')
    st.markdown(f'<div class="ans-row">{items}</div>', unsafe_allow_html=True)

    # ── PHẦN 2 ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="card-title" style="margin-top:1rem">' +
        '📌 PHẦN 2 — Đúng / Sai (8 câu × 4 ý)</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="ans-legend">' +
        '<span style="color:#4ade80">■</span> Đúng &nbsp;' +
        '<span style="color:#f87171">■</span> Sai &nbsp;' +
        '<span style="color:#fbbf24">■</span> Bỏ trống &nbsp;' +
        '<span style="color:#a5b4fc">■</span> Tô nhiều &nbsp;' +
        '&nbsp;—&nbsp; format: <code>HS/ĐA</code> (ý a·b·c·d)</div>',
        unsafe_allow_html=True
    )
    DS_LABEL = ["Đúng", "Sai"]      # đáp án 0=Đúng, 1=Sai hiển thị
    YS_LABEL = ["a","b","c","d"]
    for ci, row_key in enumerate(p2_key):
        mi_row = mi2[ci] if (mi2 and ci < len(mi2)) else None
        cells  = ""
        for yi, correct_v in enumerate(row_key):
            c_lbl = DS_LABEL[correct_v] if correct_v in (0,1) else str(correct_v)
            if mi_row is None:
                cells += (f'<span class="ai ai-ok">' +
                          f'<span class="qn">{YS_LABEL[yi]}</span>—/{c_lbl}</span>')
            else:
                sv = mi_row[yi] if yi < len(mi_row) else -1
                s_lbl = DS_LABEL[sv] if sv in (0,1) else ("–" if sv==-1 else "?")
                if sv == -1:
                    css, body = "ai-skip", f"–/{c_lbl}"
                elif sv == -2:
                    css, body = "ai-mul",  f"?/{c_lbl}"
                elif sv == correct_v:
                    css, body = "ai-ok",   f"{c_lbl}/{c_lbl}"
                else:
                    css, body = "ai-err",  f"{s_lbl}/{c_lbl}"
                cells += (f'<span class="ai {css}">' +
                          f'<span class="qn">{YS_LABEL[yi]}</span>{body}</span>')
        st.markdown(
            f'<div style="margin-bottom:.35rem">' +
            f'<span style="font-size:.78rem;color:#7fa8c9;font-weight:600;margin-right:.5rem">' +
            f'Câu {ci+1}</span>' +
            f'<span style="display:inline-flex;gap:3px">{cells}</span></div>',
            unsafe_allow_html=True
        )

    # ── PHẦN 3 ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="card-title" style="margin-top:1rem">' +
        '📌 PHẦN 3 — Điền số (6 câu · ×0.5đ/câu)</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="ans-legend">' +
        '<span style="color:#4ade80">■</span> Đúng &nbsp;' +
        '<span style="color:#f87171">■</span> Sai &nbsp;' +
        '<span style="color:#fbbf24">■</span> Bỏ trống &nbsp;' +
        '<span style="color:#a5b4fc">■</span> Tô nhiều &nbsp;' +
        '&nbsp;—&nbsp; format: <code>HS/ĐA</code> (từng ký tự)</div>',
        unsafe_allow_html=True
    )
    for ci, ans_str in enumerate(p3_key):
        # Chuyển chuỗi đáp án chuẩn thành list index để so sánh
        from support2025 import char_to_index as c2i
        ans_idx = [c2i(ch) for ch in ans_str]
        mi_row  = mi3[ci] if (mi3 and ci < len(mi3)) else None
        cells   = ""
        n_chars = max(len(ans_idx), len(mi_row) if mi_row else 0)
        for ki in range(n_chars):
            c_idx  = ans_idx[ki] if ki < len(ans_idx) else -1
            c_char = IDX_TO_CHAR.get(c_idx, "?")
            if mi_row is None:
                cells += (f'<span class="ai ai-ok">' +
                          f'<span class="qn">{ki+1}</span>—/{c_char}</span>')
            else:
                sv      = mi_row[ki] if ki < len(mi_row) else -1
                sv_char = IDX_TO_CHAR.get(sv,"?") if sv>=0 else ("–" if sv==-1 else "?")
                if sv == -1:
                    css, body = "ai-skip", f"–/{c_char}"
                elif sv == -2:
                    css, body = "ai-mul",  f"?/{c_char}"
                elif sv == c_idx:
                    css, body = "ai-ok",   f"{c_char}/{c_char}"
                else:
                    css, body = "ai-err",  f"{sv_char}/{c_char}"
                cells += (f'<span class="ai {css}">' +
                          f'<span class="qn">{ki+1}</span>{body}</span>')
        st.markdown(
            f'<div style="margin-bottom:.35rem">' +
            f'<span style="font-size:.78rem;color:#7fa8c9;font-weight:600;margin-right:.5rem">' +
            f'Câu {ci+1} <code style="color:#cbd5e1">{ans_str}</code></span>' +
            f'<span style="display:inline-flex;gap:3px">{cells}</span></div>',
            unsafe_allow_html=True
        )

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
_defaults = {
    "loai":        "2020",
    "dap_an":      {},          # {ma_de: ans}
    "sv_df":       None,
    "results":     [],          # list[dict] – mỗi bài đã chấm
    "graded_imgs": {},          # {filename: np.ndarray}
    "detail_idx":  0,
    "_id_col":     "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Dùng HTML thay vì st.markdown("### ...") để tránh Streamlit 1.58
    # tự thêm keyboard shortcut tooltip vào heading widget
    # st.markdown(
    #     '<p style="font-size:1rem;font-weight:700;margin:0 0 .5rem;'
    #     'color:#e2e8f0;letter-spacing:-.2px">📂 Cấu hình</p>',
    #     unsafe_allow_html=True
    # )
    # st.divider()

    # ── Loại phiếu ───────────────────────────────────────────────────────────
    st.markdown('<p style="font-size:.82rem;font-weight:600;color:#cbd5e1;margin:.2rem 0 .3rem">Loại phiếu</p>', unsafe_allow_html=True)
    loai_label = st.radio(
        "loai_phieu_radio", list(LOAI_PHIEU.keys()),
        index=list(LOAI_PHIEU.values()).index(st.session_state.loai),
        label_visibility="collapsed"
    )
    loai = LOAI_PHIEU[loai_label]
    if loai != st.session_state.loai:
        st.session_state.loai      = loai
        st.session_state.dap_an    = {}
        st.session_state.results   = []
        st.session_state.graded_imgs = {}
        st.session_state.detail_idx = 0
        st.rerun()

    st.divider()

    # ── File đáp án ───────────────────────────────────────────────────────────
    st.markdown('<p style="font-size:.82rem;font-weight:600;color:#cbd5e1;margin:.2rem 0 .3rem">File đáp án (.xlsx)</p>', unsafe_allow_html=True)
    with st.expander("📋 Hướng dẫn format"):
        if loai == "2025":
            st.markdown("""
**3 sheet: PHAN1 / PHAN2 / PHAN3**

`PHAN1` – Cột A=Mã đề, B→AO=40 đáp án A/B/C/D

`PHAN2` – Cột A=Mã đề, B→AE=32 ý (8 câu × 4 ý)
Thứ tự: a1,b1,c1,d1,a2,... Giá trị T/F hoặc Đúng/Sai

`PHAN3` – Cột A=Mã đề, B→G=6 câu điền số
""")
        else:
            st.markdown("""
**1 sheet** – mỗi hàng = 1 mã đề

Cột A = Mã đề
Cột B → = 120 đáp án A/B/C/D
""")

    ans_file = st.file_uploader("Chọn file đáp án", type=["xlsx","xls"], key="ans_up")
    if ans_file:
        parsed = load_ans_excel(ans_file, loai)
        if parsed:
            st.session_state.dap_an = parsed
            st.success(f"✓ {len(parsed)} mã đề: {', '.join(list(parsed.keys()))}")
        else:
            st.error("❌ Không đọc được. Kiểm tra format.")
    elif st.session_state.dap_an:
        st.success(f"✓ {len(st.session_state.dap_an)} mã đề sẵn sàng")

    st.divider()

    # ── Danh sách SV ─────────────────────────────────────────────────────────
    st.markdown('<p style="font-size:.82rem;font-weight:600;color:#cbd5e1;margin:.2rem 0 .3rem">Danh sách sinh viên (.xlsx)</p>', unsafe_allow_html=True)
    sv_file = st.file_uploader("Chọn file danh sách", type=["xlsx","xls"], key="sv_up")
    if sv_file:
        df = load_sv_excel(sv_file)
        if len(df):
            st.session_state.sv_df = df
            st.session_state["_id_col"] = guess_id_col(df)
            st.success(f"✓ {len(df)} sv · {len(df.columns)} cột")
        else:
            st.error("❌ Không đọc được.")
    elif st.session_state.sv_df is not None:
        st.success(f"✓ {len(st.session_state.sv_df)} sinh viên")

    # Chọn cột mã SV
    if st.session_state.sv_df is not None:
        cols = list(st.session_state.sv_df.columns)
        cur  = st.session_state.get("_id_col","")
        idx  = cols.index(cur) if cur in cols else 0
        st.markdown('<p style="font-size:.82rem;font-weight:600;color:#cbd5e1;margin:.2rem 0 .3rem">Cột mã sinh viên / SBD</p>', unsafe_allow_html=True)
        chosen = st.selectbox("id_col_select", cols, index=idx, label_visibility="collapsed")
        st.session_state["_id_col"] = chosen

    st.divider()

    # ── Status ────────────────────────────────────────────────────────────────
    # st.markdown(f"**OMR:** {'✅ Sẵn sàng' if OMR_OK else '❌ ' + _omr_err[:60]}")
    # st.markdown(f"**PDF:** {'✅' if PDF_OK else '⚠ pip install reportlab'}")
    if st.session_state.results:
        st.markdown(f"**Đã chấm:** {len(st.session_state.results)} bài")
        if st.button("🗑 Xóa kết quả", use_container_width=True):
            st.session_state.results    = []
            st.session_state.graded_imgs = {}
            st.session_state.detail_idx = 0
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="page-title">Hệ Thống Chấm Thi Trắc Nghiệm</div>',
            unsafe_allow_html=True)
sv_count = len(st.session_state.sv_df) if st.session_state.sv_df is not None else 0
st.markdown(
    f'<div class="page-sub">Phiếu: <b>{loai_label}</b> &nbsp;·&nbsp; '
    f'Đáp án: <b>{len(st.session_state.dap_an)}</b> mã đề &nbsp;·&nbsp; '
    f'SV: <b>{sv_count}</b> &nbsp;·&nbsp; '
    f'Đã chấm: <b>{len(st.session_state.results)}</b> bài</div>',
    unsafe_allow_html=True
)

tab_up, tab_res, tab_det, tab_exp = st.tabs(
    ["📤 Tải & Chấm", "📊 Kết quả", "🔍 Chi tiết bài", "📥 Xuất file"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – TẢI & CHẤM
# ─────────────────────────────────────────────────────────────────────────────
with tab_up:
    if not st.session_state.dap_an:
        st.markdown('<div class="bi">💡 Tải file đáp án Excel ở thanh bên trái trước.</div>',
                    unsafe_allow_html=True)

    img_files = st.file_uploader(
        "Tải ảnh phiếu trả lời (chọn nhiều file cùng lúc)",
        type=["jpg","jpeg","png"], accept_multiple_files=True, key="img_up"
    )

    if img_files:
        done_set  = {r["_filename"] for r in st.session_state.results}
        new_files = [f for f in img_files if f.name not in done_set]

        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f'<div class="bi">📁 <b>{len(img_files)}</b> ảnh &nbsp;·&nbsp; '
                f'✅ <b>{len(done_set & {f.name for f in img_files})}</b> đã chấm '
                f'&nbsp;·&nbsp; ⏳ <b>{len(new_files)}</b> chưa chấm</div>',
                unsafe_allow_html=True
            )
        with c2:
            btn = st.button(
                f"▶ Chấm {len(new_files)} ảnh",
                disabled=not (new_files and st.session_state.dap_an),
                use_container_width=True
            )

        if btn and new_files and st.session_state.dap_an:
            prog   = st.progress(0)
            status = st.empty()
            errs   = []
            id_col = st.session_state.get("_id_col", "")
            sv_df  = st.session_state.sv_df

            for i, f in enumerate(new_files):
                status.markdown(
                    f'<div class="bi">⏳ Đang xử lý <b>{f.name}</b>'
                    f' ({i+1}/{len(new_files)})…</div>',
                    unsafe_allow_html=True
                )
                try:
                    img_np = img_bytes_to_np(f.read())

                    # 1) Đọc SBD + mã đề – trả về ảnh đã vẽ overlay SBD/mã đề
                    img_mdd, sbd, ma_de = run_mdd(img_np, loai)
                    sbd   = str(sbd).strip()
                    ma_de = str(ma_de).strip()

                    # 2) Lấy đáp án theo mã đề
                    ans = st.session_state.dap_an.get(ma_de)
                    ma_de_display = ma_de
                    if ans is None:
                        # fallback mã đề đầu tiên
                        first_key = list(st.session_state.dap_an.keys())[0]
                        ans = st.session_state.dap_an[first_key]
                        ma_de_display = first_key + " (?)"

                    # 3) Chấm bài trên ảnh đã có overlay SBD/mã đề
                    img_graded, score, my_index = run_omr(img_mdd, loai, ans)
                    st.session_state.graded_imgs[f.name] = img_graded

                    # 4) Khớp danh sách SV
                    sv_row        = {}
                    not_in_dssv   = False   # True nếu có DSSV nhưng không tìm thấy SBD
                    if sv_df is not None and len(sv_df) and id_col in sv_df.columns:
                        match = sv_df[sv_df[id_col].str.strip() == sbd]
                        if len(match):
                            sv_row = match.iloc[0].to_dict()
                        else:
                            not_in_dssv = True   # có DSSV, không khớp

                    st.session_state.results.append({
                        **sv_row,
                        "_filename":     f.name,
                        "sbd":           sbd,
                        "ma_de":         ma_de_display,
                        "diem":          score,
                        "_ans_key":      ans,
                        "_my_index":     my_index,
                        "_not_in_dssv":  not_in_dssv,
                    })

                except Exception as ex:
                    errs.append(f"{f.name}: {ex}\n{traceback.format_exc()[-400:]}")

                prog.progress((i+1)/len(new_files))

            status.empty()
            if errs:
                with st.expander(f"⚠ Lỗi {len(errs)} file – bấm để xem chi tiết"):
                    for e in errs:
                        st.code(e)
            if len(new_files) > len(errs):
                st.markdown(
                    f'<div class="bk">✓ Chấm xong <b>{len(new_files)-len(errs)}</b> bài!'
                    f' Xem kết quả ở tab <b>Kết quả</b>.</div>',
                    unsafe_allow_html=True
                )
            st.rerun()

        # Thumbnails
        st.markdown("---")
        st.markdown("**Xem trước ảnh đã chọn:**")
        tcols = st.columns(min(len(img_files), 5))
        for ci, f in enumerate(img_files[:10]):
            f.seek(0)
            done = f.name in {r["_filename"] for r in st.session_state.results}
            with tcols[ci % 5]:
                st.image(Image.open(f),
                         caption=("✅ " if done else "🔲 ") + f.name[:20],
                         use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – KẾT QUẢ
# ─────────────────────────────────────────────────────────────────────────────
with tab_res:
    results = st.session_state.results
    if not results:
        st.markdown('<div class="bi">Chưa có kết quả. Hãy chấm ảnh ở tab '
                    '<b>Tải & Chấm</b>.</div>', unsafe_allow_html=True)
    else:
        scores = [float(r.get("diem", 0)) for r in results]
        avg    = sum(scores)/len(scores)

        # Stat cards
        sc = st.columns(5)
        for col, lbl, val, clr in [
            (sc[0], "Tổng bài",  len(results),             "#f1f5f9"),
            (sc[1], "Điểm TB",   f"{avg:.2f}",              "#fbbf24"),
            (sc[2], "Cao nhất",  f"{max(scores):.2f}",      "#4ade80"),
            (sc[3], "Thấp nhất", f"{min(scores):.2f}",      "#f87171"),
            (sc[4], "≥ 5 điểm",
             f"{sum(1 for s in scores if s>=5)}/{len(results)}", "#60a5fa"),
        ]:
            with col:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-val" style="color:{clr}">{val}</div>'
                    f'<div class="stat-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

        # Bar chart phân phối
        st.markdown('<div class="card"><div class="card-title">Phân phối điểm</div>',
                    unsafe_allow_html=True)
        bins = [(0,2,"<2","#ef4444"),(2,4,"2-4","#f87171"),(4,5,"4-5","#fbbf24"),
                (5,6.5,"5-6.5","#60a5fa"),(6.5,8,"6.5-8","#34d399"),(8,10.1,"≥8","#4ade80")]
        cnts = [sum(1 for s in scores if lo<=s<hi) for lo,hi,_,_ in bins]
        mx   = max(cnts) or 1
        bhtml = ""
        for (lo, hi, lbl, clr), cnt in zip(bins, cnts):
            w = max(4, int(cnt/mx*220))
            bhtml += (f'<div class="bar-wrap"><div class="bar-lbl">{lbl}</div>'
                      f'<div class="bar-body" style="width:{w}px;background:{clr}"></div>'
                      f'<div class="bar-cnt">{cnt}</div></div>')
        st.markdown(bhtml, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Bảng kết quả
        sv_df   = st.session_state.sv_df
        sv_cols = list(sv_df.columns) if (sv_df is not None and len(sv_df)) else []
        dcols   = sv_cols if sv_cols else ["sbd"]

        st.markdown('<div class="card"><div class="card-title">Danh sách kết quả</div>',
                    unsafe_allow_html=True)
        hdr  = "".join(f"<th>{c}</th>" for c in dcols)
        hdr += "<th>Mã đề</th><th>Điểm</th><th>Ghi chú</th><th>File</th>"
        rows = ""
        for rec in results:
            cells  = "".join(f"<td>{rec.get(c,'—')}</td>" for c in dcols)
            cells += f"<td style='color:#93c5fd'>{rec.get('ma_de','')}</td>"
            cells += f"<td>{score_chip(rec.get('diem',0))}</td>"
            # Ghi chú: SBD nếu không tìm thấy trong DSSV
            if rec.get("_not_in_dssv"):
                note = (f"<td style='color:#fbbf24;font-size:.74rem'>"
                        f"⚠ SBD {rec.get('sbd','')} không có trong DS</td>")
            else:
                note = "<td style='color:#475569;font-size:.74rem'>—</td>"
            cells += note
            cells += (f"<td style='color:#475569;font-size:.74rem'>"
                      f"{rec['_filename']}</td>")
            rows  += f"<tr>{cells}</tr>"
        st.markdown(
            f'<div style="overflow-x:auto"><table class="rtable">'
            f'<thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – CHI TIẾT BÀI
# ─────────────────────────────────────────────────────────────────────────────
with tab_det:
    results = st.session_state.results
    if not results:
        st.markdown('<div class="bi">Chưa có kết quả. Hãy chấm ảnh trước.</div>',
                    unsafe_allow_html=True)
    else:
        sv_df   = st.session_state.sv_df
        sv_cols = list(sv_df.columns) if (sv_df is not None and len(sv_df)) else []
        id_col  = st.session_state.get("_id_col","")

        def rec_label(i, r):
            sbd_v = r.get(id_col, r.get("sbd","?")) if id_col else r.get("sbd","?")
            name  = ""
            for c in sv_cols:
                if any(k in c.lower() for k in ["tên","ten","name","họ"]):
                    name = r.get(c,""); break
            if not name and len(sv_cols) >= 3:
                name = r.get(sv_cols[2],"")
            label = f"{i+1}.  {name}  ({sbd_v})  –  {float(r.get('diem',0)):.2f}đ"
            return label.replace("  (", " (").strip()

        cur = min(st.session_state.detail_idx or 0, len(results)-1)
        sel = st.selectbox(
            "Chọn bài:",
            range(len(results)),
            index=cur,
            format_func=lambda i: rec_label(i, results[i]),
            key="det_sel"
        )
        st.session_state.detail_idx = sel

        # Chỉ có nút "Trước", không có nút "Sau"
        # nv, _ = st.columns([1, 8])
        # with nv:
        #     if sel > 0 and st.button("◀ Trước"):
        #         st.session_state.detail_idx = sel - 1
        #         st.rerun()

        st.markdown("---")
        rec     = results[sel]
        fname   = rec["_filename"]
        img_out = st.session_state.graded_imgs.get(fname)
        ma_de   = rec.get("ma_de","?")
        diem    = float(rec.get("diem",0))
        sbd     = rec.get("sbd","?")
        ans_key = rec.get("_ans_key")
        my_idx  = rec.get("_my_index")   # None nếu OMR không expose

        col_img, col_info = st.columns([11, 10])

        # ── Ảnh đã chấm ───────────────────────────────────────────────────
        with col_img:
            st.markdown('<div class="card"><div class="card-title">Ảnh bài đã chấm'
                        ' (SBD & mã đề được ghi trên ảnh)</div>',
                        unsafe_allow_html=True)
            if img_out is not None:
                pil = np_to_pil(img_out)
                st.image(pil, use_container_width=True)
                buf_img = io.BytesIO(); pil.save(buf_img, "PNG")
                st.download_button(
                    "💾 Tải ảnh (.png)",
                    data=buf_img.getvalue(),
                    file_name=f"cham_{fname}.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.markdown('<div class="be">Không tìm thấy ảnh kết quả.</div>',
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Thông tin + đáp án ────────────────────────────────────────────
        with col_info:
            # Thông tin SV
            st.markdown('<div class="card"><div class="card-title">Thông tin</div>',
                        unsafe_allow_html=True)
            rows_info = [
                ("SBD (đọc từ ảnh)",     f"<code>{sbd}</code>"),
                ("Mã đề (đọc từ ảnh)",   f"<code>{ma_de}</code>"),
                ("Điểm",                 score_chip(diem)),
            ]
            for c in sv_cols:
                v = rec.get(c,"")
                if v and v not in ("nan","None",""):
                    rows_info.append((c, str(v)))
            grid_html = '<div class="igrid">' + "".join(
                f'<span class="ik">{k}</span><span class="iv">{v}</span>'
                for k, v in rows_info
            ) + "</div>"
            st.markdown(grid_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Đáp án mẫu + đúng/sai từng câu
            if ans_key:
                if loai in ("2020","dgnl"):
                    # Legend + card cho 2020/dgnl
                    legend = (
                        '<div class="ans-legend">'
                        '<span style="color:#4ade80">■</span> Đúng &nbsp;'
                        '<span style="color:#f87171">■</span> Sai &nbsp;'
                        '<span style="color:#fbbf24">■</span> Bỏ trống &nbsp;'
                        '<span style="color:#a5b4fc">■</span> Tô nhiều &nbsp;'
                        '&nbsp;—&nbsp; format: <code>HS/ĐA</code>'
                        '</div>'
                    )
                    st.markdown(
                        '<div class="card">'
                        '<div class="card-title">Đáp án theo từng câu</div>'
                        + legend,
                        unsafe_allow_html=True
                    )
                    render_ans_2020(ans_key, my_idx)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # 2025: render_ans_2025 tự vẽ 3 phần + legend riêng
                    st.markdown('<div class="card">'
                                '<div class="card-title">Đáp án theo từng câu</div>',
                                unsafe_allow_html=True)
                    render_ans_2025(ans_key, my_idx)
                    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – XUẤT FILE
# ─────────────────────────────────────────────────────────────────────────────
with tab_exp:
    results = st.session_state.results
    sv_df   = (st.session_state.sv_df
               if st.session_state.sv_df is not None
               else pd.DataFrame())

    if not results:
        st.markdown('<div class="bi">Chưa có kết quả để xuất.</div>',
                    unsafe_allow_html=True)
    else:
        scores = [float(r.get("diem",0)) for r in results]
        st.markdown(
            f'<div class="bk">✓ <b>{len(results)}</b> bài sẵn sàng xuất &nbsp;·&nbsp; '
            f'TB: <b>{sum(scores)/len(scores):.2f}</b> &nbsp;·&nbsp; '
            f'Cao: <b>{max(scores):.2f}</b></div>',
            unsafe_allow_html=True
        )

        e1, e2 = st.columns(2)

        with e1:
            st.markdown('<div class="card"><div class="card-title">📊 Xuất Excel (.xlsx)</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "Đầy đủ cột từ danh sách SV + **Mã đề** + **Điểm**.  \n"
                "Màu ô điểm: 🟢 ≥ 8 · 🟡 ≥ 5 · 🔴 < 5.  \n"
                "Dòng tiêu đề cố định khi cuộn, dòng trung bình ở cuối."
            )
            excel_bytes = build_excel_result(sv_df, results)
            st.download_button(
                "📥 Tải Excel",
                data=excel_bytes,
                file_name="ket_qua_cham_thi.xlsx",
                mime=("application/vnd.openxmlformats-officedocument"
                      ".spreadsheetml.sheet"),
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with e2:
            st.markdown('<div class="card"><div class="card-title">📄 Xuất PDF (.pdf)</div>',
                        unsafe_allow_html=True)
            if PDF_OK:
                st.markdown(
                    "Bảng kết quả dạng PDF, có tiêu đề, tóm tắt thống kê.  \n"
                    "Phù hợp in ấn và lưu trữ."
                )
                pdf_bytes = build_pdf_result(sv_df, results, loai_label)
                st.download_button(
                    "📄 Tải PDF",
                    data=pdf_bytes,
                    file_name="ket_qua_cham_thi.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.markdown(
                    '<div class="bw">⚠ Chưa cài reportlab.<br>'
                    '<code>pip install reportlab</code></div>',
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Preview
        st.markdown('<div class="card"><div class="card-title">Xem trước dữ liệu xuất</div>',
                    unsafe_allow_html=True)
        sv_cols = list(sv_df.columns) if len(sv_df) else []
        preview = []
        for rec in results:
            row = ({c: rec.get(c,"") for c in sv_cols}
                   if sv_cols else {"Số báo danh": rec.get("sbd","")})
            row["Mã đề"]  = rec.get("ma_de","")
            row["Điểm"]   = round(float(rec.get("diem",0)), 2)
            row["Ghi chú"] = (f"SBD {rec.get('sbd','')} không có trong DS"
                              if rec.get("_not_in_dssv") else "")
            preview.append(row)
        st.dataframe(pd.DataFrame(preview), use_container_width=True, height=300)
        st.markdown("</div>", unsafe_allow_html=True)