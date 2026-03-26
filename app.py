import streamlit as st
import torch
import torch.nn as nn
import copy
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermAI · CDSS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN SYSTEM ──────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">

<style>
/* ── ROOT ── */
:root {
    --bg-base:       #080D14;
    --bg-surface:    #0F1823;
    --bg-card:       #131E2B;
    --bg-card-hover: #172133;
    --border:        #1E2F42;
    --border-bright: #243B52;

    --accent:        #00C9A7;
    --accent-dim:    rgba(0,201,167,.15);
    --accent-glow:   rgba(0,201,167,.35);

    --warn:          #F59E0B;
    --danger:        #EF4444;
    --danger-dim:    rgba(239,68,68,.15);
    --safe:          #22C55E;
    --safe-dim:      rgba(34,197,94,.15);

    --txt-primary:   #E8EDF2;
    --txt-secondary: #7A92A8;
    --txt-muted:     #3E5469;

    --radius-sm:     8px;
    --radius-md:     14px;
    --radius-lg:     20px;
}

/* ── GLOBAL ── */
html, body, .stApp {
    background: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--txt-primary) !important;
}

/* remove streamlit chrome — κρύβουμε μόνο ό,τι δεν χρειαζόμαστε */
#MainMenu, footer { visibility: hidden; }

/* Κάνουμε το header διάφανο αντί για αόρατο, ώστε να λειτουργούν τα κλικ στο κουμπί */
header { background: transparent !important; }

/* το κουμπί επαναφοράς sidebar πρέπει να παραμένει ορατό */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
}
[data-testid="collapsedControl"] svg { color: var(--txt-secondary) !important; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--txt-primary) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--txt-primary) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="track"] div:first-child {
    background: var(--accent) !important;
}

/* ── INPUTS (main area) ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--txt-primary) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-bright) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.5rem !important;
    transition: border-color .25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] * { color: var(--txt-secondary) !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── PROGRESS BAR ── */
.stProgress > div > div > div { background: var(--accent) !important; }

/* ── ALERTS ── */
.stSuccess { background: var(--safe-dim) !important; border-left: 3px solid var(--safe) !important; border-radius: var(--radius-sm) !important; }
.stError   { background: var(--danger-dim) !important; border-left: 3px solid var(--danger) !important; border-radius: var(--radius-sm) !important; }
.stWarning { background: rgba(245,158,11,.12) !important; border-left: 3px solid var(--warn) !important; border-radius: var(--radius-sm) !important; }
.stSuccess *, .stError *, .stWarning * { color: var(--txt-primary) !important; }

/* ── IMAGES ── */
img { border-radius: var(--radius-md) !important; }

/* ── DOWNLOAD BUTTON ── */
[data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, var(--accent), #00A896) !important;
    color: #080D14 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: .02em !important;
    padding: .6rem 1.4rem !important;
    transition: all .2s ease !important;
    box-shadow: 0 4px 16px var(--accent-glow) !important;
}
[data-testid="baseButton-secondary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px var(--accent-glow) !important;
}

/* ── METRIC ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"]  { color: var(--txt-secondary) !important; font-size: .8rem !important; letter-spacing:.08em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"]  { color: var(--txt-primary) !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; }
[data-testid="stMetricDelta"]  { font-size: .85rem !important; }

/* ── CAPTION ── */
.stCaption { color: var(--txt-secondary) !important; }

/* ── HORIZONTAL RULE ── */
hr { border-color: var(--border) !important; }

/* ── CUSTOM COMPONENTS ── */

.derm-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1.8rem 2rem;
    background: linear-gradient(135deg, #0F1F30 0%, #0A1520 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.derm-header::before {
    content:'';
    position: absolute;
    top:-40px; right:-40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
}
.derm-header .icon-wrap {
    width: 56px; height: 56px;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    flex-shrink: 0;
}
.derm-header h1 {
    margin: 0;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: var(--txt-primary) !important;
    letter-spacing: -.01em;
}
.derm-header p {
    margin: .2rem 0 0;
    color: var(--txt-secondary);
    font-size: .85rem;
}
.derm-header .badge {
    margin-left: auto;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent) !important;
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    padding: .3rem .8rem;
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    flex-shrink: 0;
}

.section-label {
    display: flex;
    align-items: center;
    gap: .6rem;
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--txt-secondary);
    margin-bottom: 1rem;
    padding-bottom: .6rem;
    border-bottom: 1px solid var(--border);
}
.section-label .num {
    width: 20px; height: 20px;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: var(--accent) !important;
    font-size: .65rem;
    font-family: 'Space Mono', monospace;
    flex-shrink: 0;
}

.diagnosis-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    margin-bottom: .8rem;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.diagnosis-card.primary {
    border-color: var(--accent);
    background: linear-gradient(135deg, rgba(0,201,167,.06) 0%, var(--bg-card) 60%);
}
.diagnosis-card.danger-card { border-color: rgba(239,68,68,.4); }
.diagnosis-card:hover { border-color: var(--border-bright); }

.diag-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: .7rem;
}
.diag-name {
    font-size: 1rem;
    font-weight: 600;
    color: var(--txt-primary);
}
.diag-code {
    font-family: 'Space Mono', monospace;
    font-size: .75rem;
    background: var(--bg-surface);
    color: var(--txt-secondary);
    padding: .15rem .5rem;
    border-radius: 4px;
    border: 1px solid var(--border);
}
.diag-pct {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent);
}
.diag-bar-bg {
    height: 4px;
    background: var(--border);
    border-radius: 100px;
    overflow: hidden;
}
.diag-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent), #00E5C9);
    transition: width .8s ease;
}
.diag-bar-fill.danger { background: linear-gradient(90deg, var(--danger), #F87171); }

.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    padding: .25rem .7rem;
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
}
.risk-HIGH  { background: var(--danger-dim); color: var(--danger) !important; border: 1px solid rgba(239,68,68,.3); }
.risk-MOD   { background: rgba(245,158,11,.12); color: var(--warn) !important; border: 1px solid rgba(245,158,11,.3); }
.risk-LOW   { background: var(--safe-dim); color: var(--safe) !important; border: 1px solid rgba(34,197,94,.3); }

.cbir-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    transition: border-color .2s, transform .2s;
}
.cbir-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.cbir-footer {
    padding: .7rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.cbir-label {
    font-family: 'Space Mono', monospace;
    font-size: .7rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: .06em;
}
.cbir-sim {
    font-size: .72rem;
    color: var(--txt-secondary);
}

.sidebar-patient {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.sidebar-patient h3 {
    margin: 0 0 .5rem;
    font-size: .7rem !important;
    font-weight: 600 !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    color: var(--txt-secondary) !important;
}

.info-box {
    background: var(--accent-dim);
    border: 1px solid rgba(0,201,167,.2);
    border-radius: var(--radius-sm);
    padding: .8rem 1rem;
    font-size: .8rem;
    color: var(--txt-secondary) !important;
    line-height: 1.5;
}
.info-box strong { color: var(--accent) !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/isic2019_resnet18_multimodal.pth"
META_CSV     = "data/ISIC_2019_Training_Metadata.csv"
CBIR_DB_PATH = "data/cbir_database.pkl"

CLASS_NAMES  = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
FULL_NAMES   = {
    'MEL':  'Μελάνωμα',
    'NV':   'Σπίλος / Ελιά',
    'BCC':  'Βασικοκυτταρικό Καρκίνωμα',
    'AK':   'Ακτινική Κεράτωση',
    'BKL':  'Καλοήθης Κεράτωση',
    'DF':   'Δερματοΐνωμα',
    'VASC': 'Αγγειακή Βλάβη',
    'SCC':  'Ακανθοκυτταρικό Καρκίνωμα',
    'UNK':  'Άγνωστο',
}
# Risk levels: HIGH = κακοήθη, MOD = borderline, LOW = καλοήθη
RISK_LEVEL = {
    'MEL': 'HIGH', 'BCC': 'HIGH', 'SCC': 'HIGH',
    'AK':  'MOD',
    'NV':  'LOW',  'BKL': 'LOW',  'DF': 'LOW', 'VASC': 'LOW', 'UNK': 'MOD',
}
RISK_LABEL = {'HIGH': '⬤ Υψηλός Κίνδυνος', 'MOD': '⬤ Μέτριος Κίνδυνος', 'LOW': '⬤ Χαμηλός Κίνδυνος'}

GENDER_MAP = {"Άνδρας": "male", "Γυναίκα": "female", "Άλλο / Άγνωστο": "unknown"}
SITE_MAP = {
    "Κορμός (Εμπρός)": "anterior torso", "Κορμός (Πίσω)": "posterior torso",
    "Κορμός (Πλάγια)": "lateral torso",  "Κάτω Άκρα": "lower extremity",
    "Άνω Άκρα": "upper extremity",       "Κεφάλι / Λαιμός": "head/neck",
    "Παλάμες / Πέλματα": "palms/soles",  "Στοματική / Γεννητική": "oral/genital",
    "Άγνωστο": "unknown"
}

# ── MODEL ─────────────────────────────────────────────────────────────────────
class MultimodalNet(nn.Module):
    def __init__(self, clinical_dim, num_classes):
        super().__init__()
        self.image_branch = models.resnet18(weights=None)
        self.image_branch.fc = nn.Identity()
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, image, clinical_data):
        return self.classifier(
            torch.cat((self.image_branch(image), self.clinical_branch(clinical_data)), dim=1)
        )

class MultimodalWrapperForCAM(nn.Module):
    def __init__(self, model, clinical_data):
        super().__init__()
        self.model = model
        self.clinical_data = clinical_data
    def forward(self, x):
        return self.model(x, self.clinical_data)

@st.cache_resource
def load_system():
    meta_df = pd.read_csv(META_CSV)
    meta_df['age_approx']          = meta_df['age_approx'].fillna(meta_df['age_approx'].mean())
    meta_df['sex']                  = meta_df['sex'].fillna('unknown')
    meta_df['anatom_site_general']  = meta_df['anatom_site_general'].fillna('unknown')

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(meta_df[['sex', 'anatom_site_general']])
    scaler = StandardScaler()
    scaler.fit(meta_df[['age_approx']])

    dummy = pd.DataFrame([{'sex': 'male', 'anatom_site_general': 'anterior torso', 'age_approx': 50}])
    clinical_dim = np.concatenate([
        scaler.transform(dummy[['age_approx']]),
        encoder.transform(dummy[['sex', 'anatom_site_general']])
    ], axis=1).shape[1]

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    m = MultimodalNet(clinical_dim, len(CLASS_NAMES))
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    m = m.to(device).eval()

    with open(CBIR_DB_PATH, "rb") as f:
        cbir_db = pickle.load(f)

    return device, m, encoder, scaler, cbir_db

device, model, encoder, scaler, cbir_db = load_system()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def check_quality(img_arr, blur_thr=20.0, dark_thr=40.0):
    bgr  = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_thr:
        return False, "Η εικόνα είναι πολύ θολή για ανάλυση."
    if np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]) < dark_thr:
        return False, "Η εικόνα είναι πολύ σκοτεινή για ανάλυση."
    return True, "Ποιότητα εικόνας: ΟΚ"

def predict_multimodal(image_pil, age, gender, site):
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = tf(image_pil).unsqueeze(0).to(device)
    inp   = pd.DataFrame([{'sex': GENDER_MAP[gender], 'anatom_site_general': SITE_MAP[site], 'age_approx': age}])
    clin  = torch.tensor(
        np.concatenate([scaler.transform(inp[['age_approx']]), encoder.transform(inp[['sex', 'anatom_site_general']])], axis=1),
        dtype=torch.float
    ).to(device)

    with torch.no_grad():
        img_feat = model.image_branch(img_t)
        combined = torch.cat((img_feat, model.clinical_branch(clin)), dim=1)
        probs    = torch.nn.functional.softmax(model.classifier(combined)[0], dim=0) * 100

    img_feat_np = img_feat.cpu().numpy().flatten()
    top3_prob, top3_idx = torch.topk(probs, 3)

    # Grad-CAM on CPU — χρησιμοποιούμε deepcopy ώστε το cached model να μην μετακινηθεί
    m_cpu    = copy.deepcopy(model).to('cpu')
    wrapper  = MultimodalWrapperForCAM(m_cpu, clin.to('cpu'))
    cam      = GradCAM(model=wrapper, target_layers=[m_cpu.image_branch.layer4[-1]])
    gs_cam   = cam(input_tensor=img_t.to('cpu'), targets=[ClassifierOutputTarget(top3_idx[0].item())])[0, :]
    img_np   = np.float32(np.array(image_pil.resize((224, 224)))) / 255.0
    heatmap  = show_cam_on_image(img_np, gs_cam, use_rgb=True)

    return top3_idx, top3_prob, heatmap, img_feat_np

def generate_report(patient_name, age, gender, site, image_pil, heatmap_np, top3_idx, top3_prob):
    import tempfile, matplotlib

    try:
        # ── Fonts (bundled με matplotlib, δεν χρειάζεται δίκτυο) ──────────────
        mpl_font_dir = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
        font_regular = os.path.join(mpl_font_dir, 'DejaVuSans.ttf')
        font_bold    = os.path.join(mpl_font_dir, 'DejaVuSans-Bold.ttf')
        pdfmetrics.registerFont(TTFont('DejaVu',     font_regular))
        pdfmetrics.registerFont(TTFont('DejaVu-Bold', font_bold))

        # ── Styles ─────────────────────────────────────────────────────────────
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=40, rightMargin=40,
                                topMargin=40, bottomMargin=40,
                                title=f'DermAI Report · {patient_name}',
                                author='DermAI CDSS',
                                subject='Ιατρική Αναφορά Δερματοσκόπησης')
        W = A4[0] - 80  # usable width

        def style(name, font='DejaVu', size=11, bold=False, color=colors.HexColor('#1a1a2e'),
                  align='LEFT', space_after=4):
            return ParagraphStyle(name,
                fontName='DejaVu-Bold' if bold else font,
                fontSize=size,
                textColor=color,
                alignment={'LEFT':0,'CENTER':1,'RIGHT':2}.get(align,0),
                spaceAfter=space_after)

        title_s    = style('title',   size=18, bold=True,  color=colors.HexColor('#0d3b66'), align='CENTER', space_after=10)
        subtitle_s = style('sub',     size=10, color=colors.HexColor('#555555'), align='CENTER', space_after=20)
        section_s  = style('section', size=13, bold=True,  color=colors.HexColor('#0d3b66'), space_after=6)
        body_s     = style('body',    size=11, space_after=3)
        label_s    = style('label',   size=11, bold=True,  space_after=3)
        warn_s     = style('warn',    size=8,  color=colors.HexColor('#b91c1c'), space_after=0)

        story = []

        # ── Τίτλος ─────────────────────────────────────────────────────────────
        story.append(Paragraph('Ιατρική Αναφορά · DermAI CDSS', title_s))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f'Ημερομηνία: {datetime.now().strftime("%d/%m/%Y %H:%M")}', subtitle_s))

        # ── Στοιχεία ασθενούς ──────────────────────────────────────────────────
        story.append(Paragraph('Στοιχεία Ασθενούς', section_s))
        patient_data = [
            ['Όνομα',   patient_name],
            ['Ηλικία',  str(age)],
            ['Φύλο',    gender],
            ['Θέση',    site],
        ]
        pt = Table([[Paragraph(l, label_s), Paragraph(v, body_s)] for l, v in patient_data],
                   colWidths=[W*0.3, W*0.7])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#eef2f7')),
            ('BOX',        (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
            ('INNERGRID',  (0,0), (-1,-1), 0.3, colors.HexColor('#dddddd')),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('LEFTPADDING',(0,0),(-1,-1),8),
        ]))
        story.append(pt)
        story.append(Spacer(1, 14))

        # ── Εικόνες ────────────────────────────────────────────────────────────
        story.append(Paragraph('Ανάλυση Εικόνας', section_s))
        tmp_orig    = os.path.join(tempfile.gettempdir(), 'dermai_orig.jpg')
        tmp_heatmap = os.path.join(tempfile.gettempdir(), 'dermai_heatmap.jpg')
        image_pil.resize((224, 224)).save(tmp_orig,   format='JPEG', quality=90)
        Image.fromarray(heatmap_np).save(tmp_heatmap, format='JPEG', quality=90)

        img_w = W * 0.44
        cap_s = style('cap', size=9, color=colors.HexColor('#555555'), align='CENTER', space_after=0)
        img_table = Table([
            [RLImage(tmp_orig, width=img_w, height=img_w),
             RLImage(tmp_heatmap, width=img_w, height=img_w)],
            [Paragraph('Αρχική εικόνα', cap_s),
             Paragraph(f'Grad-CAM · {CLASS_NAMES[top3_idx[0]]}', cap_s)],
        ], colWidths=[W*0.5, W*0.5])
        img_table.setStyle(TableStyle([
            ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING',   (0,0),(-1,-1),4),
            ('BOTTOMPADDING',(0,0),(-1,-1),4),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 14))

        # ── Αποτελέσματα ───────────────────────────────────────────────────────
        story.append(Paragraph('Αποτελέσματα (Top-3)', section_s))
        header_s = style('hdr', size=11, bold=True, color=colors.white, align='CENTER', space_after=0)
        cell_s   = style('cell', size=11, align='CENTER', space_after=0)
        res_data = [[Paragraph('Διάγνωση', header_s), Paragraph('Πιθανότητα (%)', header_s)]]
        for i in range(3):
            code = CLASS_NAMES[top3_idx[i]]
            name = FULL_NAMES.get(code, code)
            res_data.append([
                Paragraph(f'{code} — {name}', cell_s),
                Paragraph(f'{top3_prob[i].item():.2f}%', cell_s),
            ])
        rt = Table(res_data, colWidths=[W*0.65, W*0.35])
        rt.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (-1,0),  colors.HexColor('#0d3b66')),
            ('BACKGROUND',   (0,1), (-1,1),  colors.HexColor('#eef6ff')),
            ('BACKGROUND',   (0,2), (-1,2),  colors.white),
            ('BACKGROUND',   (0,3), (-1,3),  colors.HexColor('#eef6ff')),
            ('BOX',          (0,0), (-1,-1), 0.5, colors.HexColor('#aaaaaa')),
            ('INNERGRID',    (0,0), (-1,-1), 0.3, colors.HexColor('#cccccc')),
            ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING',   (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ]))
        story.append(rt)
        story.append(Spacer(1, 20))

        # ── Αποποίηση ──────────────────────────────────────────────────────────
        disclaimer = (
            "ΑΠΟΠΟΙΗΣΗ ΕΥΘΥΝΗΣ: Το παρόν αποτελεί ερευνητική εξαγωγή από σύστημα ΤΝ (CDSS) "
            "και ΟΧΙ οριστική ιατρική διάγνωση. Τα αποτελέσματα πρέπει ΠΑΝΤΟΤΕ να επαληθεύονται "
            "από εξειδικευμένο Δερματολόγο."
        )
        disc_table = Table([[Paragraph(disclaimer, warn_s)]], colWidths=[W])
        disc_table.setStyle(TableStyle([
            ('BOX',         (0,0),(-1,-1), 0.5, colors.HexColor('#f87171')),
            ('BACKGROUND',  (0,0),(-1,-1), colors.HexColor('#fff5f5')),
            ('TOPPADDING',  (0,0),(-1,-1), 8),
            ('BOTTOMPADDING',(0,0),(-1,-1),8),
            ('LEFTPADDING', (0,0),(-1,-1), 10),
        ]))
        story.append(disc_table)

        doc.build(story)
        return buf.getvalue()

    except Exception:
        import traceback
        traceback.print_exc()   # εμφανίζεται στο terminal του streamlit για debugging
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:.5rem 0 1.2rem;">
        <div style="font-size:.65rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;
                    color:var(--txt-secondary);margin-bottom:.5rem;">Φάκελος Ασθενούς</div>
        <div style="font-size:1.1rem;font-weight:600;color:var(--txt-primary);">Νέα Εξέταση</div>
    </div>
    """, unsafe_allow_html=True)

    patient_name = st.text_input("Όνομα Ασθενούς", "Ιωάννης Παπαδόπουλος")
    age          = st.slider("Ηλικία (έτη)", 1, 100, 45)
    gender       = st.selectbox("Φύλο", list(GENDER_MAP.keys()))
    location     = st.selectbox("Ανατομική Θέση", list(SITE_MAP.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Πολυτροπική Ανάλυση</strong><br>
        Το μοντέλο συνδυάζει τη δερματοσκοπική εικόνα με τα κλινικά δεδομένα
        για πιο ακριβή αξιολόγηση.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:.68rem;color:var(--txt-muted);line-height:1.8;
                border-top:1px solid var(--border);padding-top:.8rem;">
        <div>🔬 ResNet-18 · Multimodal</div>
        <div>📊 ISIC 2019 Dataset</div>
        <div>🛡️ v2.0 · CDSS Research</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="derm-header">
    <div class="icon-wrap">🔬</div>
    <div>
        <h1>DermAI · Σύστημα Υποβοήθησης Κλινικών Αποφάσεων</h1>
        <p>Ογκολογικό Δερματολογικό Κέντρο &nbsp;·&nbsp; Μονάδα Τεχνητής Νοημοσύνης</p>
    </div>
    <div class="badge">CDSS v2.0</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-label">
    <div class="num">1</div> Εισαγωγή Δερματοσκοπικής Εικόνας
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Σύρετε ή επιλέξτε αρχείο · JPG / JPEG / PNG",
    type=["jpg", "jpeg", "png"]
)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    is_valid, qmsg = check_quality(np.array(image_pil))

    st.markdown("<br>", unsafe_allow_html=True)

    if not is_valid:
        st.error(f"⚠️  {qmsg}")
        st.stop()

    st.success(f"✅  {qmsg} — Εκκίνηση Ανάλυσης…")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── RUN MODEL ──────────────────────────────────────────────────────────────
    with st.spinner("Ανάλυση δεδομένων — παρακαλώ περιμένετε…"):
        top3_idx, top3_prob, heatmap, img_feat_np = predict_multimodal(
            image_pil, age, gender, location
        )

    top1_code = CLASS_NAMES[top3_idx[0]]
    top1_risk = RISK_LEVEL.get(top1_code, 'MOD')

    # ── STEP 2 ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-label">
        <div class="num">2</div> Αποτελέσματα Πολυτροπικής Ανάλυσης
    </div>
    """, unsafe_allow_html=True)

    col_img1, col_img2, col_res = st.columns([3, 3, 4], gap="large")

    with col_img1:
        st.image(image_pil, caption="Αρχική Δερματοσκοπική Εικόνα", use_container_width=True)

    with col_img2:
        st.image(heatmap, caption=f"Grad-CAM · Εστίαση AI ({top1_code})", use_container_width=True)

    with col_res:
        # Primary diagnosis card
        bar_danger = top1_risk == 'HIGH'
        st.markdown(f"""
        <div class="diagnosis-card primary {'danger-card' if bar_danger else ''}">
            <div class="diag-top">
                <div>
                    <div style="font-size:.65rem;font-weight:600;letter-spacing:.12em;
                                text-transform:uppercase;color:var(--txt-secondary);
                                margin-bottom:.3rem;">Επικρατέστερη Διάγνωση</div>
                    <div class="diag-name">{FULL_NAMES.get(top1_code, top1_code)}</div>
                </div>
                <div style="text-align:right;">
                    <div class="diag-pct">{top3_prob[0].item():.1f}%</div>
                    <div class="diag-code">{top1_code}</div>
                </div>
            </div>
            <div class="diag-bar-bg">
                <div class="diag-bar-fill {'danger' if bar_danger else ''}"
                     style="width:{top3_prob[0].item():.1f}%"></div>
            </div>
            <div style="margin-top:.8rem;">
                <span class="risk-badge risk-{top1_risk}">{RISK_LABEL[top1_risk]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        # Alternative diagnoses
        st.markdown("""
        <div style="font-size:.65rem;font-weight:600;letter-spacing:.12em;
                    text-transform:uppercase;color:var(--txt-secondary);margin-bottom:.5rem;">
            Εναλλακτικές Εκτιμήσεις
        </div>
        """, unsafe_allow_html=True)

        for i in range(1, 3):
            c       = CLASS_NAMES[top3_idx[i]]
            p       = top3_prob[i].item()
            r       = RISK_LEVEL.get(c, 'MOD')
            st.markdown(f"""
            <div class="diagnosis-card" style="padding:.9rem 1.2rem; margin-bottom:.5rem;">
                <div class="diag-top" style="margin-bottom:.5rem;">
                    <div>
                        <div class="diag-name" style="font-size:.9rem;">{FULL_NAMES.get(c,c)}</div>
                        <span class="risk-badge risk-{r}" style="margin-top:.3rem;display:inline-flex;">
                            {RISK_LABEL[r]}
                        </span>
                    </div>
                    <div class="diag-pct" style="font-size:1rem;">{p:.1f}%</div>
                </div>
                <div class="diag-bar-bg">
                    <div class="diag-bar-fill" style="width:{p:.1f}%;opacity:.7;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        # PDF export
        report_bytes = generate_report(
            patient_name, age, gender, location,
            image_pil, heatmap, top3_idx, top3_prob
        )
        if report_bytes:
            ts  = datetime.now().strftime("%Y%m%d_%H%M")
            fn  = f"DermAI_Report_{patient_name.replace(' ','_')}_{ts}.pdf"
            st.download_button(
                label="📄  Εξαγωγή Ιατρικής Αναφοράς (PDF)",
                data=report_bytes,
                file_name=fn,
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("⚠️  Αδυναμία δημιουργίας PDF — ελέγξτε τη σύνδεση δικτύου.")

    # ── STEP 3 — CBIR ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">
        <div class="num">3</div> Κλινικό Αρχείο · Παρόμοια Ιστορικά Περιστατικά (CBIR)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:.8rem;color:var(--txt-secondary);margin-bottom:1.2rem;">
        Αναζήτηση οπτικής ομοιότητας σε βάση
        <strong style="color:var(--txt-primary);">2.000 επιβεβαιωμένων περιστατικών</strong>
        βάσει του διανύσματος χαρακτηριστικών (512-d cosine similarity).
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Αναζήτηση στα ιατρικά αρχεία…"):
        db_feats = np.array([item['features'] for item in cbir_db])
        sims     = cosine_similarity([img_feat_np], db_feats)[0]
        top3_sim = sims.argsort()[-3:][::-1]

    sim_cols = st.columns(3, gap="medium")
    for i, idx in enumerate(top3_sim):
        match     = cbir_db[idx]
        img_path  = os.path.join("data/images", f"{match['image_name']}.jpg")
        risk_m    = RISK_LEVEL.get(match['label'], 'MOD')
        try:
            m_img = Image.open(img_path)
            with sim_cols[i]:
                st.markdown('<div class="cbir-card">', unsafe_allow_html=True)
                st.image(m_img, use_container_width=True)
                st.markdown(f"""
                <div class="cbir-footer">
                    <div>
                        <div class="cbir-label">{match['label']} — {FULL_NAMES.get(match['label'],'')}</div>
                        <span class="risk-badge risk-{risk_m}" style="margin-top:.3rem;display:inline-flex;font-size:.62rem;">
                            {RISK_LABEL[risk_m]}
                        </span>
                    </div>
                    <div class="cbir-sim">Ομοιότητα<br><strong style="color:var(--accent);font-family:'Space Mono',monospace;">
                        {sims[idx]*100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception:
            pass

    # ── DISCLAIMER ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.2);
                border-radius:var(--radius-sm);padding:.9rem 1.2rem;
                font-size:.78rem;color:var(--txt-secondary);line-height:1.6;">
        <strong style="color:var(--danger);">⚠️ Αποποίηση Ευθύνης</strong>&nbsp;
        Το παρόν αποτέλεσμα προέρχεται από σύστημα Τεχνητής Νοημοσύνης (CDSS) για
        <em>ερευνητικούς σκοπούς</em> και <strong>δεν αποτελεί οριστική ιατρική διάγνωση</strong>.
        Τα αποτελέσματα πρέπει πάντοτε να επαληθεύονται από εξειδικευμένο Δερματολόγο.
    </div>
    """, unsafe_allow_html=True)
