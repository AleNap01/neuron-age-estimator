"""
main.py — Backend NeuroAge (V2: predizione, mesh 3D, Grad-CAM, referto PDF)

Carica i modelli (Ensemble ML classico e CNN 3D) UNA SOLA VOLTA all'avvio
del server, poi li riusa per ogni richiesta. Caricarli ad ogni richiesta
sarebbe corretto ma lentissimo (il modello CNN va ricostruito e i pesi
ricaricati da disco ogni volta) — un classico errore da evitare in un
backend reale.

Esecuzione:
    uvicorn main:app --reload --port 8000
"""

import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # nessun display: il backend gira come server headless
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import trimesh
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.ndimage import zoom as ndzoom
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes

from brain_age.config import CNN_MODEL_PATH, MODEL_V2_PATH
from brain_age.data.preprocessing import downsample_volume
from brain_age.features.extraction import extract_features_v2
from brain_age.models.cnn3d import BrainAgeCNN3D
from brain_age.models.gradcam import GradCAM3D, overlay_gradcam_on_slice

app = FastAPI(title="NeuroAge Backend")

# Permette al frontend Vite (in dev su una porta diversa) di chiamare l'API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MESH_VOLUME_SIZE = 96  # compromesso dettaglio/peso per la mesh 3D del cervello
CNN_VOLUME_SIZE = 64

# MAE di riferimento dei due modelli (da cross-validation / validation set),
# stessi numeri mostrati nella V1 Streamlit — usati nel referto PDF.
MODEL_INFO = {
    "ensemble": {"label": "Ensemble ML classico", "mae": 4.76},
    "cnn": {"label": "CNN 3D", "mae": 4.06},
}

# ──────────────────────────────────────────────
# CARICAMENTO MODELLI ALL'AVVIO DEL SERVER
# ──────────────────────────────────────────────
# Queste righe vengono eseguite UNA volta, quando il server parte
# (non a ogni richiesta). I modelli restano in memoria, pronti all'uso.

TRAIN_AGE_MEAN = 33.21
TRAIN_AGE_STD = 21.19

print("Caricamento modello Ensemble...")
classical_model = joblib.load(MODEL_V2_PATH) if MODEL_V2_PATH.exists() else None

print("Caricamento modello CNN 3D...")
cnn_model = None
if CNN_MODEL_PATH.exists():
    cnn_model = BrainAgeCNN3D()
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location="cpu"))
    cnn_model.eval()

print(f"Modelli pronti — Ensemble: {'OK' if classical_model else 'MANCANTE'}, "
      f"CNN: {'OK' if cnn_model else 'MANCANTE'}")


# ──────────────────────────────────────────────
# FUNZIONI DI SUPPORTO
# ──────────────────────────────────────────────
def load_volume_from_upload(contents: bytes, filename: str) -> np.ndarray:
    """Salva temporaneamente i byte ricevuti e li legge come volume NIfTI."""
    suffix = ".nii.gz" if filename.endswith(".gz") else ".nii"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        img = nib.load(str(tmp_path))
        volume = np.asarray(img.dataobj, dtype=np.float32)

        img.uncache()
        if hasattr(img, "file_map"):
            for file_holder in img.file_map.values():
                if hasattr(file_holder, "fileobj") and file_holder.fileobj is not None:
                    file_holder.fileobj.close()
        del img

        return volume
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except PermissionError:
            pass


def predict_with_classical(volume: np.ndarray) -> float:
    features = extract_features_v2(volume).reshape(1, -1)
    return float(classical_model.predict(features)[0])


def predict_with_cnn(volume: np.ndarray) -> float:
    volume_small = downsample_volume(volume, target_size=CNN_VOLUME_SIZE)
    tensor = torch.from_numpy(volume_small[np.newaxis, np.newaxis, :, :, :]).float()
    with torch.no_grad():
        prediction_normalized = cnn_model(tensor).item()
    return prediction_normalized * TRAIN_AGE_STD + TRAIN_AGE_MEAN


def compute_gradcam_volume(volume: np.ndarray) -> np.ndarray:
    """Mappa di salienza Grad-CAM 3D (valori in [0,1]) alla risoluzione CNN_VOLUME_SIZE^3."""
    volume_small = downsample_volume(volume, target_size=CNN_VOLUME_SIZE)
    tensor = torch.from_numpy(volume_small[np.newaxis, np.newaxis, :, :, :]).float()
    cam, _ = GradCAM3D(cnn_model).generate(tensor)
    return cam, volume_small


REGION_PALETTE = {
    "frontale": (66, 153, 225),
    "parietale": (56, 178, 172),
    "temporale": (237, 137, 54),
    "occipitale": (159, 122, 234),
    "cervelletto_tronco": (229, 62, 62),
    "centrale": (207, 227, 242),
}


def region_vertex_colors(verts: np.ndarray) -> np.ndarray:
    """
    Colora i vertici della mesh in zone per posizione geometrica (assi della
    mesh: 0=verticale, 1=antero-posteriore, 2=laterale), ispirate ai lobi
    cerebrali. È una suddivisione ILLUSTRATIVA basata su geometria, non un
    atlante anatomico validato — serve a rendere visibili strutture diverse
    quando si seziona il modello, non a localizzare aree cliniche precise.
    """
    v = verts.astype(np.float64)
    spread = v.max(axis=0) - v.min(axis=0)
    spread[spread == 0] = 1.0
    norm = (v - v.mean(axis=0)) / spread  # ~ -0.5..0.5 per asse

    vertical, antpost, lateral = norm[:, 0], norm[:, 1], norm[:, 2]

    colors = np.tile(np.array(REGION_PALETTE["centrale"], dtype=np.uint8), (len(v), 1))

    def paint(mask, key):
        colors[mask] = REGION_PALETTE[key]

    paint(vertical < -0.28, "cervelletto_tronco")
    remaining = vertical >= -0.28
    paint(remaining & (np.abs(lateral) > 0.27) & (vertical < 0.05), "temporale")
    paint(remaining & (antpost > 0.22), "frontale")
    paint(remaining & (antpost < -0.22), "occipitale")
    paint(remaining & (vertical > 0.18) & (np.abs(antpost) <= 0.22), "parietale")

    alpha = np.full((len(v), 1), 255, dtype=np.uint8)
    return np.hstack([colors, alpha])


def gradcam_vertex_colors(cam_at_vertex: np.ndarray) -> np.ndarray:
    """
    Colora i vertici della mesh in base alla salienza Grad-CAM: dal blu-grigio
    neutro del cervello (bassa salienza) al rosso acceso (alta salienza),
    stessa logica "calore" di overlay_gradcam_on_slice ma per vertici 3D.
    """
    t = np.clip(cam_at_vertex, 0, 1)[:, None]
    low = np.array([207, 227, 242])
    high = np.array([230, 45, 40])
    rgb = (low * (1 - t) + high * t).astype(np.uint8)
    alpha = np.full((len(t), 1), 255, dtype=np.uint8)
    return np.hstack([rgb, alpha])


def build_brain_mesh_glb(volume: np.ndarray, cam: np.ndarray | None = None) -> bytes:
    """
    Ricostruisce la superficie 3D del cervello dal volume MRI e la esporta
    come GLB, pronto per essere caricato da three.js nel browser.

    Il volume del dataset è già skull-stripped (background ~0), quindi una
    soglia di Otsu sui voxel non-nulli separa bene cervello da sfondo senza
    bisogno di segmentazione aggiuntiva.

    Se `cam` (mappa Grad-CAM, shape CNN_VOLUME_SIZE^3) è fornita, i vertici
    vengono colorati in base alla salienza. Altrimenti, di default, vengono
    colorati per zona geometrica (vedi `region_vertex_colors`): senza questo
    accorgimento la mesh è di un blu uniforme e, sezionandola, le strutture
    interne sono indistinguibili tra loro.
    """
    volume_small = downsample_volume(volume, target_size=MESH_VOLUME_SIZE)

    nonzero = volume_small[volume_small > 0]
    level = float(threshold_otsu(nonzero)) if nonzero.size else 0.01

    verts, faces, _, _ = marching_cubes(volume_small, level=level, step_size=1)

    if cam is not None:
        zoom_factor = MESH_VOLUME_SIZE / cam.shape[0]
        cam_resized = np.clip(ndzoom(cam, zoom=zoom_factor, order=1), 0, 1)
        idx = np.clip(verts.round().astype(int), 0, MESH_VOLUME_SIZE - 1)
        cam_at_vertex = cam_resized[idx[:, 0], idx[:, 1], idx[:, 2]]
        vertex_colors = gradcam_vertex_colors(cam_at_vertex)
    else:
        vertex_colors = region_vertex_colors(verts)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors, process=True)

    # Decimazione: la superficie marching-cubes ha più triangoli di quanti
    # ne servano per un rendering fluido nel browser — dimezzarli riduce il
    # peso del GLB scaricato senza una perdita visibile di dettaglio.
    try:
        mesh = mesh.simplify_quadric_decimation(percent=0.5)
    except Exception:
        pass  # se fast-simplification non è disponibile, si esporta la mesh intera

    mesh.vertices -= mesh.bounding_box.centroid
    scale = 2.0 / max(mesh.extents.max(), 1e-6)
    mesh.vertices *= scale

    return mesh.export(file_type="glb")


def build_slice_figure(volume: np.ndarray, cam: np.ndarray | None = None):
    """3 slice ortogonali (assiale/coronale/sagittale) centrali, con overlay Grad-CAM opzionale."""
    cx, cy, cz = (s // 2 for s in volume.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.patch.set_facecolor("#0a0e14")

    titles = ["ASSIALE", "CORONALE", "SAGITTALE"]
    slices = [volume[cx, :, :], volume[:, cy, :], volume[:, :, cz]]
    cam_slices = [cam[cx, :, :], cam[:, cy, :], cam[:, :, cz]] if cam is not None else [None, None, None]

    vmax = volume.max() or 1.0
    for ax, sl, cam_sl, title in zip(axes, slices, cam_slices, titles):
        sl_norm = np.clip(sl / vmax, 0, 1)
        if cam_sl is not None:
            ax.imshow(overlay_gradcam_on_slice(sl_norm, cam_sl, alpha=0.5), origin="lower")
        else:
            ax.imshow(sl_norm, cmap="gray", origin="lower")
        ax.set_title(title, color="#9db4c4", fontsize=10, fontfamily="monospace")
        ax.axis("off")

    fig.tight_layout()
    return fig


def _interpretation_text(gap: float, mae: float) -> str:
    """Testo di interpretazione del brain-age gap, con linguaggio clinico ma sempre cautelativo."""
    abs_gap = abs(gap)
    if abs_gap >= 6:
        verdict = (
            f"Lo scostamento osservato tra età cerebrale stimata ed età anagrafica è "
            f"<b>elevato</b> rispetto all'errore atteso del modello (±{mae:.1f} anni). Si "
            f"raccomanda la correlazione con il quadro clinico complessivo e, qualora il "
            f"medico curante lo ritenga utile, un approfondimento specialistico."
        )
    elif abs_gap >= 3:
        verdict = (
            f"Lo scostamento osservato è <b>moderato</b>, parzialmente compatibile con "
            f"l'errore atteso del modello (±{mae:.1f} anni). In assenza di altri elementi "
            f"clinici di rilievo, può essere ragionevole un'osservazione nel tempo."
        )
    else:
        verdict = (
            f"Il valore stimato è <b>coerente</b> con l'età anagrafica dichiarata, entro "
            f"l'errore atteso del modello (±{mae:.1f} anni)."
        )
    return verdict + (
        " La stima è generata da un modello statistico/di apprendimento automatico a "
        "partire da immagini di risonanza magnetica e non costituisce, da sola, una diagnosi."
    )


def _methodology_text(model_label: str, mae: float) -> str:
    if "CNN" in model_label:
        return (
            "Rete neurale convoluzionale 3D (CNN), addestrata end-to-end su volumi MRI "
            "strutturali T1-pesati (n≈2.300 volumi di addestramento). Errore medio assoluto "
            f"(MAE) misurato in validazione: ±{mae:.2f} anni."
        )
    return (
        "Ensemble di modelli di machine learning classico (Support Vector Regression e "
        "Random Forest) addestrato su feature morfometriche estratte dal volume MRI "
        f"strutturale (n≈2.300 volumi di addestramento). Errore medio assoluto (MAE) "
        f"misurato in validazione: ±{mae:.2f} anni."
    )


def _report_id(filename: str) -> str:
    suffix = abs(hash(filename)) % 10000
    return f"NA-{datetime.now():%Y%m%d}-{suffix:04d}"


def _draw_letterhead(canvas, doc, report_id: str, generated_at: str) -> None:
    canvas.saveState()
    width, height = A4

    canvas.setFillColor(rl_colors.HexColor("#1d72c2"))
    canvas.roundRect(2 * cm, height - 2.35 * cm, 0.85 * cm, 0.85 * cm, 3, fill=1, stroke=0)
    canvas.setFillColor(rl_colors.white)
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawCentredString(2 * cm + 0.425 * cm, height - 2.35 * cm + 0.28 * cm, "N")

    canvas.setFillColor(rl_colors.HexColor("#16222e"))
    canvas.setFont("Helvetica-Bold", 15)
    canvas.drawString(3.1 * cm, height - 1.85 * cm, "NeuroAge")
    name_width = canvas.stringWidth("NeuroAge", "Helvetica-Bold", 15)
    canvas.setFillColor(rl_colors.HexColor("#1d72c2"))
    canvas.drawString(3.1 * cm + name_width, height - 1.85 * cm, "·MRI")

    canvas.setFillColor(rl_colors.HexColor("#7c8b9a"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(3.1 * cm, height - 2.28 * cm, "REFERTO DI ANALISI NEURO-RADIOLOGICA COMPUTAZIONALE")

    canvas.setFont("Helvetica-Bold", 9.5)
    canvas.setFillColor(rl_colors.HexColor("#16222e"))
    canvas.drawRightString(width - 2 * cm, height - 1.8 * cm, f"Referto N. {report_id}")
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(rl_colors.HexColor("#7c8b9a"))
    canvas.drawRightString(width - 2 * cm, height - 2.18 * cm, generated_at)

    canvas.setStrokeColor(rl_colors.HexColor("#e2e9f0"))
    canvas.setLineWidth(1)
    canvas.line(2 * cm, height - 2.55 * cm, width - 2 * cm, height - 2.55 * cm)
    canvas.restoreState()


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(rl_colors.HexColor("#e2e9f0"))
    canvas.setLineWidth(1)
    canvas.line(2 * cm, 1.7 * cm, width - 2 * cm, 1.7 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(rl_colors.HexColor("#a6b1bc"))
    canvas.drawString(
        2 * cm, 1.35 * cm,
        "Strumento di supporto alla ricerca — non sostituisce il giudizio clinico di personale qualificato.",
    )
    canvas.drawRightString(width - 2 * cm, 1.35 * cm, f"Pagina {canvas.getPageNumber()}")
    canvas.restoreState()


def generate_pdf_report(
    filename: str,
    model_label: str,
    chrono_age: float,
    predicted_age: float,
    mae: float,
    volume_for_preview: np.ndarray,
    cam: np.ndarray | None,
) -> bytes:
    """
    Genera il referto PDF in formato clinico professionale: letterhead e
    footer ricorrenti su ogni pagina (ID referto, numerazione), sezioni
    strutturate (dati esame, risultato, interpretazione, metodologia,
    visualizzazione) e un riquadro di validazione/firma — sul modello dei
    referti diagnostici reali, pur restando uno strumento di supporto alla
    ricerca (vedi disclaimer in calce a ogni pagina).
    """
    report_id = _report_id(filename)
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=3.0 * cm, bottomMargin=2.2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()

    section_style = ParagraphStyle(
        "Section", parent=styles["Normal"], fontSize=10.5, fontName="Helvetica-Bold",
        textColor=rl_colors.HexColor("#1d72c2"), spaceAfter=8, spaceBefore=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10,
        textColor=rl_colors.HexColor("#16222e"), leading=15,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=8.5,
        textColor=rl_colors.HexColor("#7c8b9a"), leading=12,
    )
    gap = predicted_age - chrono_age
    ci_low, ci_high = predicted_age - mae, predicted_age + mae

    def on_page(canvas, doc_):
        _draw_letterhead(canvas, doc_, report_id, generated_at)
        _draw_footer(canvas, doc_)

    story = []

    story.append(Paragraph("DATI DELL'ESAME", section_style))
    info_table = Table(
        [
            ["File analizzato", filename],
            ["Data e ora analisi", generated_at],
            ["ID referto", report_id],
            ["Modello di stima utilizzato", model_label],
            ["Pipeline", "NeuroAge V2 — analisi computazionale automatizzata"],
        ],
        colWidths=[6.5 * cm, 9.5 * cm],
    )
    info_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), rl_colors.HexColor("#7c8b9a")),
        ("TEXTCOLOR", (1, 0), (1, -1), rl_colors.HexColor("#16222e")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, rl_colors.HexColor("#e2e9f0")),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("RISULTATO DELL'ANALISI", section_style))
    result_table = Table(
        [
            ["Età anagrafica dichiarata", f"{chrono_age:.1f} anni"],
            ["Età cerebrale stimata", f"{predicted_age:.1f} anni"],
            ["Intervallo plausibile (±MAE)", f"{ci_low:.1f} – {ci_high:.1f} anni"],
            ["Scostamento (brain-age gap)", f"{gap:+.1f} anni"],
            ["Errore atteso del modello (MAE)", f"± {mae:.1f} anni"],
        ],
        colWidths=[6.5 * cm, 9.5 * cm],
    )
    result_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), rl_colors.HexColor("#7c8b9a")),
        ("TEXTCOLOR", (1, 0), (1, -1), rl_colors.HexColor("#16222e")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (1, 1), (1, 1), 13),
        ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor("#f6fafc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, rl_colors.HexColor("#e2e9f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#e2e9f0")),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("INTERPRETAZIONE", section_style))
    story.append(Paragraph(_interpretation_text(gap, mae), body_style))
    story.append(Spacer(1, 18))

    story.append(Paragraph("METODOLOGIA", section_style))
    story.append(Paragraph(_methodology_text(model_label, mae), body_style))
    story.append(Spacer(1, 18))

    story.append(Paragraph("VISUALIZZAZIONE", section_style))
    slices_fig = build_slice_figure(volume_for_preview, cam)
    img_buffer = BytesIO()
    slices_fig.savefig(img_buffer, format="png", dpi=120, facecolor="#0a0e14")
    plt.close(slices_fig)
    img_buffer.seek(0)
    story.append(RLImage(img_buffer, width=16 * cm, height=5.6 * cm))
    story.append(Spacer(1, 4))
    caption = "Slice ortogonali centrali (assiale, coronale, sagittale)"
    if cam is not None:
        caption += " con overlay Grad-CAM: le aree rosse hanno influenzato di più la stima della CNN."
    story.append(Paragraph(caption, caption_style))
    story.append(Spacer(1, 26))

    story.append(Paragraph("VALIDAZIONE", section_style))
    sig_table = Table(
        [
            ["Referto generato automaticamente da sistema di supporto alla ricerca.", ""],
            ["Revisionato da", "_________________________________"],
            ["Data", "_________________________________"],
            ["Firma", "_________________________________"],
        ],
        colWidths=[5 * cm, 11 * cm],
    )
    sig_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("SPAN", (0, 0), (1, 0)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.HexColor("#7c8b9a")),
        ("TEXTCOLOR", (0, 1), (0, -1), rl_colors.HexColor("#7c8b9a")),
        ("TEXTCOLOR", (1, 1), (1, -1), rl_colors.HexColor("#16222e")),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(sig_table)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buffer.seek(0)
    return buffer.getvalue()


# ──────────────────────────────────────────────
# ROUTE
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "NeuroAge backend è vivo!"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = "cnn",  # "cnn" oppure "ensemble"
):
    """
    Riceve un volume MRI e restituisce l'età stimata dal modello scelto.

    Il parametro 'model' arriva come query string, es.:
        POST /predict?model=cnn
    """
    if model not in ("cnn", "ensemble"):
        raise HTTPException(status_code=400, detail="model deve essere 'cnn' o 'ensemble'")

    contents = await file.read()
    volume = load_volume_from_upload(contents, file.filename)

    if model == "cnn":
        if cnn_model is None:
            raise HTTPException(status_code=503, detail="Modello CNN non disponibile sul server")
        predicted_age = predict_with_cnn(volume)
    else:
        if classical_model is None:
            raise HTTPException(status_code=503, detail="Modello Ensemble non disponibile sul server")
        predicted_age = predict_with_classical(volume)

    mae = MODEL_INFO[model]["mae"]
    return {
        "filename": file.filename,
        "model": model,
        "predicted_age": round(predicted_age, 1),
        "mae": mae,
        "confidence_interval": [round(predicted_age - mae, 1), round(predicted_age + mae, 1)],
    }


@app.post("/mesh")
async def mesh(file: UploadFile = File(...), overlay: str | None = None):
    """
    Riceve un volume MRI e restituisce la mesh 3D della superficie cerebrale
    in formato GLB (binario), pronta per essere visualizzata e ruotata nel
    browser con three.js.

    Con `overlay=gradcam` (query string), i vertici della mesh vengono
    colorati in base alla mappa di salienza Grad-CAM della CNN 3D — richiede
    che il modello CNN sia disponibile sul server.
    """
    contents = await file.read()
    volume = load_volume_from_upload(contents, file.filename)

    cam = None
    if overlay == "gradcam":
        if cnn_model is None:
            raise HTTPException(status_code=503, detail="Grad-CAM richiede il modello CNN, non disponibile")
        cam, _ = compute_gradcam_volume(volume)

    try:
        glb_bytes = build_brain_mesh_glb(volume, cam=cam)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=f"Impossibile generare la mesh: {exc}")

    return Response(content=glb_bytes, media_type="model/gltf-binary")


@app.post("/report")
async def report(
    file: UploadFile = File(...),
    model: str = Form(...),
    predicted_age: float = Form(...),
    chrono_age: float = Form(...),
):
    """
    Genera il referto PDF dell'analisi: età stimata, brain-age gap, anteprima
    slice (con overlay Grad-CAM se il modello è la CNN), pronto da scaricare.
    """
    if model not in MODEL_INFO:
        raise HTTPException(status_code=400, detail="model deve essere 'cnn' o 'ensemble'")

    contents = await file.read()
    volume = load_volume_from_upload(contents, file.filename)

    cam = None
    volume_for_preview = downsample_volume(volume, target_size=CNN_VOLUME_SIZE)
    if model == "cnn" and cnn_model is not None:
        cam, volume_for_preview = compute_gradcam_volume(volume)

    pdf_bytes = generate_pdf_report(
        filename=file.filename,
        model_label=MODEL_INFO[model]["label"],
        chrono_age=chrono_age,
        predicted_age=predicted_age,
        mae=MODEL_INFO[model]["mae"],
        volume_for_preview=volume_for_preview,
        cam=cam,
    )

    safe_name = Path(file.filename).stem
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="neuroage_referto_{safe_name}.pdf"'},
    )
