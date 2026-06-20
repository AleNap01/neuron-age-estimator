"""
app.py — Brain Age Estimation: interfaccia Streamlit (design "NeuroAge")

Carica una scansione MRI (.nii / .nii.gz) e stima l'età del soggetto
usando, a scelta, il modello ML classico (Ensemble SVR+RF) o la CNN 3D.

Design ispirato al mockup "NeuroAge·MRI" creato con Claude Design:
palette clinica chiara (blu #1d72c2 / verde #15976a), font IBM Plex.

Esecuzione:
    streamlit run app.py
"""

import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

from brain_age.config import CNN_MODEL_PATH, MODEL_V2_PATH
from brain_age.data.preprocessing import downsample_volume
from brain_age.features.extraction import extract_features_v2
from brain_age.models.cnn3d import BrainAgeCNN3D
from brain_age.models.gradcam import GradCAM3D, overlay_gradcam_on_slice

# ──────────────────────────────────────────────
# CONFIGURAZIONE PAGINA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroAge · MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

TRAIN_AGE_MEAN = 33.21
TRAIN_AGE_STD = 21.19

# MAE di riferimento dei due modelli (da cross-validation / validation set)
MODEL_INFO = {
    "Ensemble ML classico": {
        "subtitle": "Feature morfometriche + SVR/Random Forest",
        "mae": 4.76,
        "n": "2.3k",
    },
    "CNN 3D": {
        "subtitle": "Rete convoluzionale volumetrica end-to-end",
        "mae": 4.06,
        "n": "2.3k",
    },
}

# ──────────────────────────────────────────────
# CSS — palette e font del mockup NeuroAge
# ──────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --na-blue: #1d72c2;
    --na-blue-dark: #1565ad;
    --na-green: #15976a;
    --na-bg: #eef3f7;
    --na-text: #16222e;
    --na-text-soft: #5b6b79;
    --na-text-faint: #7c8b9a;
    --na-border: #e2e9f0;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    font-size: 17px;
}

.stApp {
    background: var(--na-bg);
}

/* Header personalizzato */
.na-header {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 14px 0 18px 0;
    border-bottom: 1px solid var(--na-border);
    margin-bottom: 18px;
}
.na-logo {
    width: 38px; height: 38px;
    border-radius: 9px;
    background: linear-gradient(135deg, var(--na-blue), var(--na-green));
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 8px rgba(29,114,194,.25);
    font-size: 19px;
}
.na-title { font-weight: 700; font-size: 20px; color: var(--na-text); letter-spacing: -0.2px; }
.na-title span { color: var(--na-blue); }
.na-subtitle { font-size: 14px; color: var(--na-text-faint); font-weight: 500; }

.na-pill {
    font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--na-green);
    background: #e4f5ee;
    padding: 5px 10px;
    border-radius: 99px;
    font-weight: 500;
}

/* Card modello (sidebar) */
.na-model-card {
    border: 1.5px solid var(--na-border);
    border-radius: 12px;
    padding: 13px 14px;
    margin-bottom: 10px;
    background: #fff;
}
.na-model-card.selected {
    border-color: var(--na-blue);
    background: #f1f7fd;
    box-shadow: 0 0 0 3px rgba(29,114,194,.10);
}
.na-model-name { font-size: 15.5px; font-weight: 600; color: var(--na-text); }
.na-model-desc { font-size: 13px; color: var(--na-text-faint); margin-top: 2px; line-height: 1.4; }
.na-model-mae { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--na-blue); margin-top: 6px; }

/* Risultato */
.na-result-box {
    background: linear-gradient(160deg, #f4faff 0%, #eef7f2 100%);
    border: 1px solid var(--na-border);
    border-radius: 16px;
    padding: 28px 30px;
}
.na-result-label {
    font-size: 13px; font-weight: 600; letter-spacing: .6px;
    text-transform: uppercase; color: var(--na-text-faint); margin-bottom: 8px;
}
.na-result-age {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 64px; font-weight: 600; letter-spacing: -2px;
    color: var(--na-blue); line-height: 1;
}
.na-result-unit { font-size: 21px; font-weight: 500; color: var(--na-text-soft); margin-left: 6px; }

.na-badge {
    display: inline-block;
    font-size: 14.5px; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    padding: 8px 15px; border-radius: 99px;
    margin-top: 14px;
}
.na-badge.ok { color: #15976a; background: #e4f5ee; }
.na-badge.mid { color: #c98a16; background: #faf2e0; }
.na-badge.high { color: #c4453d; background: #fae8e7; }

/* Slice card */
.na-slice-label {
    font-size: 15px; font-weight: 600; letter-spacing: .4px;
    text-transform: uppercase; color: var(--na-text-faint);
}

.na-disclaimer {
    font-size: 10.5px; color: #a6b1bc; line-height: 1.5;
    border-top: 1px solid var(--na-border); padding-top: 14px; margin-top: 18px;
}

.na-info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px 22px;
}
.na-info-label { font-size: 13px; color: var(--na-text-faint); margin-bottom: 3px; }
.na-info-value { font-family: 'IBM Plex Mono', monospace; font-size: 18px; font-weight: 600; color: var(--na-text); }

/* Bottoni Streamlit */
.stButton > button {
    background: linear-gradient(135deg, var(--na-blue), var(--na-blue-dark));
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    padding: 10px 22px;
    box-shadow: 0 2px 10px rgba(29,114,194,.28);
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--na-blue-dark), var(--na-blue-dark));
    color: white;
}

/* Radio nella sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--na-border);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# STATO DI SESSIONE — storico delle analisi
# ──────────────────────────────────────────────
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


# ──────────────────────────────────────────────
# CARICAMENTO MODELLI (cache)
# ──────────────────────────────────────────────
@st.cache_resource
def load_classical_model():
    if not MODEL_V2_PATH.exists():
        return None
    return joblib.load(MODEL_V2_PATH)


@st.cache_resource
def load_cnn_model():
    if not CNN_MODEL_PATH.exists():
        return None
    model = BrainAgeCNN3D()
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# ──────────────────────────────────────────────
# FUNZIONI DI PREDIZIONE
# ──────────────────────────────────────────────
def predict_classical(volume: np.ndarray, model) -> float:
    features = extract_features_v2(volume).reshape(1, -1)
    return float(model.predict(features)[0])


def predict_cnn(volume: np.ndarray, model) -> tuple[float, np.ndarray]:
    """Predice l'età con la CNN; ritorna anche il volume downsampled (serve per Grad-CAM)."""
    volume_small = downsample_volume(volume, target_size=64)
    tensor = torch.from_numpy(
        volume_small[np.newaxis, np.newaxis, :, :, :]).float()
    with torch.no_grad():
        prediction_normalized = model(tensor).item()
    age = prediction_normalized * TRAIN_AGE_STD + TRAIN_AGE_MEAN
    return age, volume_small


@st.cache_resource
def get_gradcam(_model):
    """Il modello viene passato con underscore per evitare che Streamlit tenti di hashare i pesi."""
    return GradCAM3D(_model)


def compute_gradcam(volume_small: np.ndarray, model) -> np.ndarray:
    """Calcola la mappa Grad-CAM 3D per un volume già downsampled a 64^3."""
    gradcam = get_gradcam(model)
    tensor = torch.from_numpy(
        volume_small[np.newaxis, np.newaxis, :, :, :]).float()
    cam, _ = gradcam.generate(tensor)
    return cam


def plot_slices(volume: np.ndarray, cam: np.ndarray | None = None, indices: tuple[int, int, int] | None = None):
    """
    Visualizza 3 slice del volume (assiale, coronale, sagittale).

    Se `indices` non è fornito, usa le slice centrali. Se viene fornita una
    mappa cam (stessa shape del volume), sovrappone una heatmap rosso/giallo
    che evidenzia le zone con maggiore influenza sulla predizione della CNN
    (Grad-CAM).
    """
    if indices is None:
        cx, cy, cz = (s // 2 for s in volume.shape)
    else:
        cx, cy, cz = indices

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.patch.set_facecolor("#0a0e14")

    titles = ["ASSIALE", "CORONALE", "SAGITTALE"]
    slices = [volume[cx, :, :], volume[:, cy, :], volume[:, :, cz]]
    slice_numbers = [cx, cy, cz]
    slice_maxes = [volume.shape[0], volume.shape[1], volume.shape[2]]

    if cam is not None:
        cam_slices = [cam[cx, :, :], cam[:, cy, :], cam[:, :, cz]]
    else:
        cam_slices = [None, None, None]

    for ax, sl, cam_sl, title, idx, idx_max in zip(
        axes, slices, cam_slices, titles, slice_numbers, slice_maxes
    ):
        if cam_sl is not None:
            image = overlay_gradcam_on_slice(sl, cam_sl, alpha=0.5)
            ax.imshow(image, origin="lower")
        else:
            ax.imshow(sl, cmap="gray", origin="lower")
        ax.set_facecolor("#0a0e14")
        ax.text(0.04, 0.94, title, transform=ax.transAxes,
                color="#7fd0ff", fontsize=10, fontfamily="monospace",
                verticalalignment="top")
        ax.text(0.96, 0.04, f"{idx} / {idx_max}", transform=ax.transAxes,
                color="#9aa7b4", fontsize=9, fontfamily="monospace",
                horizontalalignment="right", verticalalignment="bottom")
        ax.axis("off")

    plt.tight_layout()
    return fig


def render_header():
    st.markdown(
        """
        <div class="na-header">
            <div class="na-logo">🧠</div>
            <div style="flex:1;">
                <div class="na-title">NeuroAge<span>·</span>MRI</div>
                <div class="na-subtitle">Stima dell'età cerebrale da risonanza magnetica 3D</div>
            </div>
            <div class="na-pill">● Pipeline attiva</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(name: str, selected: bool):
    info = MODEL_INFO[name]
    css_class = "na-model-card selected" if selected else "na-model-card"
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="na-model-name">{name}</div>
            <div class="na-model-desc">{info['subtitle']}</div>
            <div class="na-model-mae">MAE {info['mae']:.1f} anni · n={info['n']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gap_badge(gap: float) -> str:
    abs_gap = abs(gap)
    sign = "+" if gap >= 0 else "−"
    if abs_gap >= 6:
        css_class, label = "high", "elevato"
    elif abs_gap >= 3:
        css_class, label = "mid", "lievemente elevato"
    else:
        css_class, label = "ok", "nella norma"
    return (
        f'<span class="na-badge {css_class}">'
        f"Δ {sign}{abs_gap:.1f} anni · {label}</span>"
    )


def generate_pdf_report(
    filename: str,
    model_name: str,
    chrono_age: float,
    brain_age: float,
    mae: float,
    slices_fig,
) -> bytes:
    """Genera un referto PDF sintetico con i risultati dell'analisi corrente."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"], fontSize=20,
        textColor=colors.HexColor("#1d72c2"), spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=11,
        textColor=colors.HexColor("#7c8b9a"), spaceAfter=16,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#16222e"), leading=15,
    )

    story = [
        Paragraph("NeuroAge · MRI", title_style),
        Paragraph("Referto di stima dell'età cerebrale", subtitle_style),
    ]

    info_table = Table(
        [
            ["File analizzato", filename],
            ["Data analisi", datetime.now().strftime("%d/%m/%Y %H:%M")],
            ["Modello utilizzato", model_name],
            ["Età anagrafica dichiarata", f"{chrono_age} anni"],
            ["Età cerebrale stimata", f"{brain_age:.1f} anni"],
            ["Scostamento (gap)", f"{brain_age - chrono_age:+.1f} anni"],
            ["Errore atteso del modello (MAE)", f"± {mae:.1f} anni"],
        ],
        colWidths=[7*cm, 9*cm],
    )
    info_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#7c8b9a")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#16222e")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#e2e9f0")),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 18))

    # Anteprima slice come immagine
    img_buffer = BytesIO()
    slices_fig.savefig(img_buffer, format="png", dpi=120, facecolor="#0a0e14")
    img_buffer.seek(0)
    story.append(Paragraph("Anteprima slice", body_style))
    story.append(Spacer(1, 6))
    story.append(RLImage(img_buffer, width=16*cm, height=5.6*cm))
    story.append(Spacer(1, 20))

    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8,
        textColor=colors.HexColor("#a6b1bc"), leading=12,
    )
    story.append(Paragraph(
        "Strumento di supporto alla ricerca, sviluppato come progetto di "
        "portfolio. Non destinato all'uso diagnostico autonomo. Il risultato "
        "va sempre interpretato nel contesto clinico complessivo da personale "
        "qualificato.",
        disclaimer_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;letter-spacing:.6px;'
        'text-transform:uppercase;color:#9aa7b4;margin-bottom:12px;">'
        "Modello AI</div>",
        unsafe_allow_html=True,
    )

    model_choice = st.radio(
        "Modello",
        options=["Ensemble ML classico", "CNN 3D"],
        label_visibility="collapsed",
    )

    for name in MODEL_INFO:
        render_model_card(name, selected=(name == model_choice))

    show_gradcam = False
    if model_choice == "CNN 3D":
        show_gradcam = st.checkbox(
            "🔥 Mostra mappa di attivazione (Grad-CAM)",
            value=False,
            help=(
                "Evidenzia le zone del volume che hanno influenzato di più "
                "la predizione della rete. Richiede qualche secondo in più."
            ),
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;font-weight:600;letter-spacing:.6px;'
        'text-transform:uppercase;color:#9aa7b4;margin-bottom:12px;">'
        "Dati paziente</div>",
        unsafe_allow_html=True,
    )

    chrono_age = st.number_input(
        "Età anagrafica (anni)", min_value=1, max_value=110, value=40,
        help="Età reale del soggetto, usata solo per calcolare lo scostamento (brain age gap).",
    )

    st.markdown(
        '<div class="na-disclaimer">Strumento di supporto alla ricerca. '
        "Non destinato all'uso diagnostico autonomo.</div>",
        unsafe_allow_html=True,
    )

    # ── Storico delle analisi della sessione corrente ──
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown(
            '<div style="font-size:11px;font-weight:600;letter-spacing:.6px;'
            'text-transform:uppercase;color:#9aa7b4;margin-bottom:10px;">'
            f"Storico sessione ({len(st.session_state.analysis_history)})</div>",
            unsafe_allow_html=True,
        )
        for entry in reversed(st.session_state.analysis_history[-5:]):
            st.markdown(
                f"""
                <div style="font-size:11.5px;padding:8px 10px;background:#f7f9fb;
                            border-radius:8px;margin-bottom:6px;line-height:1.5;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-weight:600;
                                color:#16222e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {entry['file']}
                    </div>
                    <div style="color:#7c8b9a;">
                        {entry['modello']} · stimata {entry['eta_stimata']} anni
                        (Δ{entry['gap']:+.1f})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("🗑️ Svuota storico", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()

# ──────────────────────────────────────────────
# HEADER + AREA PRINCIPALE
# ──────────────────────────────────────────────
render_header()

st.markdown(
    f"<p style='color:#5b6b79;font-size:14px;margin-top:-8px;'>"
    f"Carica un volume MRI strutturale per stimare l'età biologica del "
    f"cervello con il modello <strong style='color:#1d72c2;'>{model_choice}</strong>.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Carica un file MRI",
    type=["nii", "gz"],
    help="Formati supportati: .nii, .nii.gz",
)

if uploaded_file is not None:
    # ── Validazione preliminare (punto 7: robustezza) ──
    MAX_FILE_SIZE_MB = 200
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"❌ Il file è troppo grande ({file_size_mb:.1f} MB). "
            f"Dimensione massima supportata: {MAX_FILE_SIZE_MB} MB. "
            "Verifica di aver caricato un volume MRI a risoluzione standard "
            "e non, ad esempio, una serie DICOM completa non compressa."
        )
        st.stop()

    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)

    try:
        with st.spinner("Caricamento del volume MRI..."):
            try:
                img = nib.load(str(tmp_path))
                volume = np.asarray(img.dataobj, dtype=np.float32)
            except Exception as load_error:
                st.error(
                    "❌ Impossibile leggere il file come volume NIfTI valido. "
                    "Verifica che sia un file .nii o .nii.gz non corrotto "
                    f"(dettaglio tecnico: {load_error})."
                )
                st.stop()

            img.uncache()
            if hasattr(img, "file_map"):
                for file_holder in img.file_map.values():
                    if hasattr(file_holder, "fileobj") and file_holder.fileobj is not None:
                        file_holder.fileobj.close()
            del img

        # Validazione struttura del volume: deve essere 3D
        if volume.ndim != 3:
            st.error(
                f"❌ Il volume caricato ha {volume.ndim} dimensioni invece di 3. "
                "Questo strumento si aspetta una singola scansione MRI "
                "strutturale 3D (non una serie 4D/temporale)."
            )
            st.stop()

        # Validazione dimensioni: volumi troppo piccoli o eccessivamente
        # squilibrati sono probabilmente file non adeguati
        if min(volume.shape) < 32:
            st.error(
                f"❌ Il volume ha dimensioni insolitamente piccole {volume.shape}. "
                "Verifica di aver caricato una scansione cerebrale completa."
            )
            st.stop()

        if max(volume.shape) / min(volume.shape) > 3:
            st.warning(
                f"⚠️ Il volume ha proporzioni inusuali {volume.shape}. "
                "Il modello è stato addestrato su volumi pressoché cubici "
                "(128×128×128): risultati su geometrie molto diverse "
                "potrebbero non essere affidabili."
            )

        # Validazione range di intensità: i volumi del dataset di training
        # sono normalizzati in [0, 1]; un range molto diverso indica che la
        # MRI non è stata pre-processata nello stesso modo
        vol_min, vol_max = float(volume.min()), float(volume.max())
        if vol_max > 10 or vol_min < -1:
            st.warning(
                f"⚠️ Il range di intensità del volume ([{vol_min:.2f}, {vol_max:.2f}]) "
                "è molto diverso da quello atteso ([0, 1], dati normalizzati). "
                "Le predizioni potrebbero essere poco accurate: il modello è "
                "stato addestrato su MRI già normalizzate e pre-processate."
            )

        st.markdown(
            f'<div class="na-pill" style="margin-bottom:14px;">'
            f"✓ Volume validato · {volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]}"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander("🎚️ Naviga manualmente tra le slice", expanded=False):
            col_ax, col_co, col_sa = st.columns(3)
            with col_ax:
                idx_axial = st.slider(
                    "Assiale", 0, volume.shape[0] - 1, volume.shape[0] // 2)
            with col_co:
                idx_coronal = st.slider(
                    "Coronale", 0, volume.shape[1] - 1, volume.shape[1] // 2)
            with col_sa:
                idx_sagittal = st.slider(
                    "Sagittale", 0, volume.shape[2] - 1, volume.shape[2] // 2)
            custom_indices = (idx_axial, idx_coronal, idx_sagittal)

        # Per la CNN calcoliamo subito la predizione (serve il volume
        # downsampled anche per il Grad-CAM, evitiamo di downsamplare due volte)
        brain_age = None
        cam_map = None

        if model_choice == "Ensemble ML classico":
            model = load_classical_model()
            if model is None:
                st.error(f"Modello non trovato in {MODEL_V2_PATH}.")
            else:
                with st.spinner("Estrazione feature e predizione..."):
                    brain_age = predict_classical(volume, model)

            st.markdown(
                '<div class="na-slice-label">Anteprima slice</div>', unsafe_allow_html=True)
            fig = plot_slices(volume, indices=custom_indices)
            st.pyplot(fig)

        else:  # CNN 3D
            model = load_cnn_model()
            if model is None:
                st.error(f"Modello non trovato in {CNN_MODEL_PATH}.")
            else:
                with st.spinner("Downsampling e predizione..."):
                    brain_age, volume_small = predict_cnn(volume, model)

                # Gli slider sono calibrati sulla risoluzione originale (es. 128^3);
                # li riscaliamo alla risoluzione 64^3 vista internamente dalla CNN.
                scale_factors = [s_small / s_orig for s_small,
                                 s_orig in zip(volume_small.shape, volume.shape)]
                small_indices = tuple(
                    min(int(idx * factor), s_small - 1)
                    for idx, factor, s_small in zip(custom_indices, scale_factors, volume_small.shape)
                )

                if show_gradcam:
                    with st.spinner("Calcolo della mappa Grad-CAM..."):
                        cam_map = compute_gradcam(volume_small, model)
                    st.markdown(
                        '<div class="na-slice-label">Anteprima slice — '
                        'mappa di attivazione Grad-CAM</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "🔴 Zone in rosso/giallo: voxel che hanno influenzato di più "
                        "la predizione della rete. Volume mostrato a risoluzione 64³ "
                        "(quella usata internamente dalla CNN)."
                    )
                    fig = plot_slices(
                        volume_small, cam=cam_map, indices=small_indices)
                else:
                    st.markdown(
                        '<div class="na-slice-label">Anteprima slice</div>', unsafe_allow_html=True)
                    fig = plot_slices(volume_small, indices=small_indices)

                st.pyplot(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        if brain_age is not None:
            gap = brain_age - chrono_age
            info = MODEL_INFO[model_choice]

            col_result, col_details = st.columns([1, 1.3])

            with col_result:
                st.markdown(
                    f"""
                    <div class="na-result-box">
                        <div class="na-result-label">Età cerebrale stimata</div>
                        <div><span class="na-result-age">{brain_age:.1f}</span>
                        <span class="na-result-unit">anni</span></div>
                        {gap_badge(gap)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_details:
                st.markdown(
                    f"""
                    <div class="na-info-grid">
                        <div>
                            <div class="na-info-label">Età anagrafica</div>
                            <div class="na-info-value">{chrono_age} anni</div>
                        </div>
                        <div>
                            <div class="na-info-label">Modello</div>
                            <div class="na-info-value" style="font-size:14px;">{model_choice}</div>
                        </div>
                        <div>
                            <div class="na-info-label">Errore atteso (MAE)</div>
                            <div class="na-info-value">± {info['mae']:.1f} anni</div>
                        </div>
                        <div>
                            <div class="na-info-label">Scostamento (gap)</div>
                            <div class="na-info-value">{gap:+.1f} anni</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="na-disclaimer" style="margin-top:18px;">'
                "ⓘ Risultato a supporto della ricerca — da interpretare nel "
                "contesto clinico complessivo.</div>",
                unsafe_allow_html=True,
            )

            # ── Download referto PDF (punto 6) ──
            pdf_bytes = generate_pdf_report(
                filename=uploaded_file.name,
                model_name=model_choice,
                chrono_age=chrono_age,
                brain_age=brain_age,
                mae=info["mae"],
                slices_fig=fig,
            )
            st.download_button(
                "📄 Scarica referto PDF",
                data=pdf_bytes,
                file_name=f"neuroage_referto_{uploaded_file.name.split('.')[0]}.pdf",
                mime="application/pdf",
            )

            # Registra l'analisi nello storico di sessione (evita duplicati
            # consecutivi identici, es. al solo cambio di uno slider)
            entry = {
                "file": uploaded_file.name,
                "modello": model_choice,
                "eta_anagrafica": chrono_age,
                "eta_stimata": round(brain_age, 1),
                "gap": round(gap, 1),
            }
            if not st.session_state.analysis_history or st.session_state.analysis_history[-1] != entry:
                st.session_state.analysis_history.append(entry)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Confronto diretto tra i due modelli sullo stesso file ──
            with st.expander("⚖️ Confronta entrambi i modelli su questo file"):
                if st.button("Esegui confronto"):
                    other_model_name = "CNN 3D" if model_choice == "Ensemble ML classico" else "Ensemble ML classico"

                    with st.spinner(f"Calcolo predizione con {other_model_name}..."):
                        if other_model_name == "Ensemble ML classico":
                            other_model = load_classical_model()
                            other_age = predict_classical(
                                volume, other_model) if other_model else None
                        else:
                            other_model = load_cnn_model()
                            other_age, _ = predict_cnn(
                                volume, other_model) if other_model else (None, None)

                    if other_age is not None:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(model_choice, f"{brain_age:.1f} anni",
                                      help=f"MAE atteso: ±{MODEL_INFO[model_choice]['mae']:.1f} anni")
                        with col_b:
                            st.metric(other_model_name, f"{other_age:.1f} anni",
                                      help=f"MAE atteso: ±{MODEL_INFO[other_model_name]['mae']:.1f} anni")

                        diff = abs(brain_age - other_age)
                        st.caption(
                            f"Differenza tra i due modelli: **{diff:.1f} anni**. "
                            f"Uno scostamento ampio può indicare un caso limite "
                            f"per entrambi gli approcci."
                        )
                    else:
                        st.error(
                            f"Modello {other_model_name} non disponibile.")

    except Exception as e:
        st.error(
            f"❌ Si è verificato un errore durante l'elaborazione del file.\n\n"
            f"**Dettaglio tecnico:** {type(e).__name__}: {e}\n\n"
            "Possibili cause: file non valido, modello mancante in "
            "`outputs/`, o memoria insufficiente per volumi molto grandi. "
            "Se il problema persiste, verifica i log del terminale."
        )

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except PermissionError:
            pass

else:
    st.markdown(
        """
        <div style="background:#fff;border:2px dashed #cdd9e4;border-radius:12px;
                    padding:48px 20px;text-align:center;margin-top:10px;">
            <div style="font-size:15px;font-weight:600;color:#16222e;">
                ⬆️ Carica un file MRI per iniziare
            </div>
            <div style="font-size:12.5px;color:#8c99a6;margin-top:6px;font-family:'IBM Plex Mono',monospace;">
                .nii · .nii.gz
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Informazioni sul progetto"):
        st.markdown(
            """
            Questo strumento è stato sviluppato come progetto di portfolio,
            a partire da un'esercitazione del corso di Machine Learning
            (UniNa – Federico II).

            **Modelli disponibili:**
            - **Ensemble ML classico**: combina SVR e Random Forest su 155
              feature statistiche estratte da ciascun volume MRI
              (MAE ≈ 4.76 anni in cross-validation)
            - **CNN 3D**: rete neurale convoluzionale addestrata su volumi
              ridotti a 64³ voxel (MAE ≈ 4.06 anni in validazione)

            Entrambi i modelli sono stati addestrati interamente su CPU,
            senza GPU NVIDIA/CUDA disponibile in fase di sviluppo.

            Design dell'interfaccia ispirato a un mockup creato con Claude Design.
            """
        )
