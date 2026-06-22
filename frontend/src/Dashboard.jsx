import { useEffect, useRef, useState } from "react";
import BrainViewer3D from "./BrainViewer3D.jsx";
import TrendChart from "./TrendChart.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const HISTORY_KEY = "neuroage_history_v1";
const MAX_HISTORY = 8;

/** Stessa palette usata server-side in region_vertex_colors (backend/main.py). */
const REGION_LEGEND = [
  { label: "Frontale", color: "#4299e1" },
  { label: "Parietale", color: "#38b2ac" },
  { label: "Temporale", color: "#ed8936" },
  { label: "Occipitale", color: "#9f7aea" },
  { label: "Cervelletto / tronco", color: "#e53e3e" },
  { label: "Area centrale", color: "#cfe3f2" },
];

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(entries) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
}

/** Stessa logica/soglie della V1 Streamlit (gap_badge): 3 e 6 anni. */
function gapBadge(gap) {
  const abs = Math.abs(gap);
  if (abs >= 6) return { label: "elevato", color: "#c0392b", bg: "#fbe8e6" };
  if (abs >= 3) return { label: "lievemente elevato", color: "#b07b1a", bg: "#fdf3e0" };
  return { label: "nella norma", color: "#15976a", bg: "#e4f5ee" };
}

/**
 * Dashboard — pagina di analisi reale: upload della MRI, scelta del
 * modello, età anagrafica, predizione + brain-age gap, modello 3D (con
 * overlay Grad-CAM opzionale), referto PDF scaricabile e storico delle
 * analisi (persistito in localStorage).
 *
 * Layout a "bento": il visualizzatore 3D è l'elemento hero, a piena
 * larghezza in alto; upload, risultato e storico vivono in card sotto.
 */
export default function Dashboard({ onBackClick }) {
  const [file, setFile] = useState(null);
  const [modelChoice, setModelChoice] = useState("cnn");
  const [chronoAge, setChronoAge] = useState("");
  const [showGradcam, setShowGradcam] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [glbUrl, setGlbUrl] = useState(null);
  const [meshLoading, setMeshLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const inputRef = useRef();

  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const handleFileChange = (selected) => {
    if (!selected) return;
    setFile(selected);
    setResult(null);
    setGlbUrl(null);
    setShowGradcam(false);
    setStatus("idle");
    setErrorMsg("");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) handleFileChange(dropped);
  };

  const fetchMesh = async (targetFile, overlay) => {
    const fd = new FormData();
    fd.append("file", targetFile);
    const url = overlay ? `${API_BASE}/mesh?overlay=gradcam` : `${API_BASE}/mesh`;
    const res = await fetch(url, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Generazione modello 3D fallita (${res.status})`);
    return URL.createObjectURL(await res.blob());
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setStatus("loading");
    setErrorMsg("");

    try {
      const predictFd = new FormData();
      predictFd.append("file", file);

      const [predictRes, meshUrl] = await Promise.all([
        fetch(`${API_BASE}/predict?model=${modelChoice}`, { method: "POST", body: predictFd }),
        fetchMesh(file, showGradcam && modelChoice === "cnn"),
      ]);

      if (!predictRes.ok) throw new Error(`Predizione fallita (${predictRes.status})`);

      const predictionData = await predictRes.json();
      setResult(predictionData);
      setGlbUrl(meshUrl);
      setStatus("done");

      const chrono = parseFloat(chronoAge);
      const entry = {
        filename: predictionData.filename,
        model: predictionData.model,
        predicted_age: predictionData.predicted_age,
        chrono_age: Number.isFinite(chrono) ? chrono : null,
        gap: Number.isFinite(chrono) ? Math.round((predictionData.predicted_age - chrono) * 10) / 10 : null,
        date: new Date().toISOString(),
      };
      const next = [entry, ...history];
      setHistory(next);
      saveHistory(next);
    } catch (err) {
      setErrorMsg(err.message || "Errore durante l'analisi");
      setStatus("error");
    }
  };

  const handleToggleGradcam = async (checked) => {
    setShowGradcam(checked);
    if (!file || !result || modelChoice !== "cnn") return;
    setMeshLoading(true);
    try {
      const meshUrl = await fetchMesh(file, checked);
      setGlbUrl(meshUrl);
    } catch (err) {
      setErrorMsg(err.message || "Errore nel caricamento dell'overlay Grad-CAM");
    } finally {
      setMeshLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    if (!file || !result) return;
    setPdfLoading(true);
    setErrorMsg("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("model", result.model);
      fd.append("predicted_age", String(result.predicted_age));
      fd.append("chrono_age", String(Number.isFinite(parseFloat(chronoAge)) ? parseFloat(chronoAge) : result.predicted_age));

      const res = await fetch(`${API_BASE}/report`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Generazione referto fallita (${res.status})`);

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `neuroage_referto_${result.filename.split(".")[0]}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setErrorMsg(err.message || "Errore nella generazione del referto");
    } finally {
      setPdfLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    saveHistory([]);
  };

  const chronoNum = parseFloat(chronoAge);
  const gap = result && Number.isFinite(chronoNum) ? result.predicted_age - chronoNum : null;
  const badge = gap !== null ? gapBadge(gap) : null;

  return (
    <div
      style={{
        minHeight: "100vh",
        fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
        background: "radial-gradient(1100px 600px at 85% -8%, #eaf4fb 0%, #f6fafc 45%, #f6fafc 100%)",
        color: "#16222e",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        @keyframes naFadeUp { from { opacity:0; transform: translateY(10px); } to { opacity:1; transform:translateY(0); } }
        .na-focusable:focus-visible { outline: 2.5px solid #1d72c2; outline-offset: 2px; }
      `}</style>

      {/* ===== HEADER ===== */}
      <header
        style={{
          maxWidth: 1320, margin: "0 auto", padding: "22px 32px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
          <div
            style={{
              width: 34, height: 34, borderRadius: 9,
              background: "linear-gradient(135deg,#1d72c2,#15976a)",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 4px 12px rgba(29,114,194,.28)",
            }}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <circle cx="10" cy="10" r="7.4" stroke="#fff" strokeWidth="1.3" opacity="0.85" />
              <circle cx="10" cy="10" r="3.9" stroke="#fff" strokeWidth="1.3" />
              <circle cx="10" cy="10" r="1.4" fill="#fff" />
            </svg>
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-.3px" }}>
              NeuroAge<span style={{ color: "#1d72c2" }}>·</span>MRI
            </div>
            <div style={{ fontSize: 12, color: "#8c99a6" }}>Sezione di analisi</div>
          </div>
        </div>

        <button
          onClick={onBackClick}
          className="na-focusable"
          style={{
            fontSize: 13.5, fontWeight: 600, color: "#1d72c2", background: "#fff",
            border: "1.5px solid #d3e3f1", padding: "10px 18px", borderRadius: 10,
            cursor: "pointer", transition: "border-color .18s ease, box-shadow .18s ease",
          }}
        >
          ← Torna alla landing
        </button>
      </header>

      <main style={{ maxWidth: 1320, margin: "0 auto", padding: "4px 32px 64px" }}>
        {/* ===== HERO: MODELLO 3D ===== */}
        <section style={{ marginBottom: 26, animation: "naFadeUp .5s ease both" }} aria-label="Modello 3D del cervello">
          {!glbUrl && status !== "loading" && (
            <div
              style={{
                height: "min(64vh, 620px)", minHeight: 420, borderRadius: 24,
                border: "1.5px dashed #c8dceb", display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center", gap: 14,
                color: "#7c8b99", fontSize: 14.5, background: "#fff",
              }}
            >
              <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="#a9c2d4" strokeWidth="1.4" aria-hidden="true">
                <path d="M12 3c-3.5 0-5.5 2.2-5.5 4.6 0 1-.6 1.4-1.3 2-1 .8-1.4 1.8-1 3 .3 1 .2 1.6-.4 2.4-.8 1-.6 2.3.5 2.8.9.4 1.2.9 1.2 1.8 0 1.6 1.6 2.4 3 2 .9-.2 1.5.1 2 .9.7 1 2.4 1 3.1 0 .5-.8 1.1-1.1 2-.9 1.4.4 3-.4 3-2 0-.9.3-1.4 1.2-1.8 1.1-.5 1.3-1.8.5-2.8-.6-.8-.7-1.4-.4-2.4.4-1.2 0-2.2-1-3-.7-.6-1.3-1-1.3-2C17.5 5.2 15.5 3 12 3Z" />
                <path d="M12 6.2v12M9 8.6c-1 .8-1 2.2 0 3.4M15 8.6c1 .8 1 2.2 0 3.4M9.4 14.4c-1 .6-1 2 0 2.8M14.6 14.4c1 .6 1 2 0 2.8" />
              </svg>
              <div style={{ fontWeight: 600, color: "#5b6b79" }}>
                Carica una MRI per ricostruire il modello 3D
              </div>
              <div style={{ fontSize: 12.5, maxWidth: 360, textAlign: "center", lineHeight: 1.5 }}>
                Il modello sarà ruotabile, zoomabile e sezionabile per ispezionare le strutture
                interne del cervello.
              </div>
            </div>
          )}

          {status === "loading" && (
            <div
              role="status"
              style={{
                height: "min(64vh, 620px)", minHeight: 420, borderRadius: 24,
                border: "1px solid #e6eef4", display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center", gap: 14,
                color: "#5b6b79", fontSize: 14.5, background: "#fff",
              }}
            >
              <div
                style={{
                  width: 38, height: 38, borderRadius: "50%",
                  border: "3px solid #d3e3f1", borderTopColor: "#1d72c2",
                  animation: "naSpinLoad 0.9s linear infinite",
                }}
              />
              <style>{`@keyframes naSpinLoad { to { transform: rotate(360deg); } }`}</style>
              Ricostruzione della superficie cerebrale in corso…
            </div>
          )}

          {glbUrl && (
            <div style={{ position: "relative" }}>
              {meshLoading && (
                <div
                  role="status"
                  style={{
                    position: "absolute", inset: 0, zIndex: 5, borderRadius: 24,
                    background: "rgba(10,20,32,.55)", display: "flex", alignItems: "center",
                    justifyContent: "center", color: "#fff", fontSize: 13.5, fontWeight: 600,
                  }}
                >
                  Applico l'overlay Grad-CAM…
                </div>
              )}
              <BrainViewer3D glbUrl={glbUrl} />
            </div>
          )}

          {glbUrl && (
            <>
              <p style={{ fontSize: 12.5, color: "#8c99a6", marginTop: 12, lineHeight: 1.6, textAlign: "center" }}>
                Trascina per ruotare (movimento ammortizzato), scorri per zoomare. Usa lo slider
                "piano di sezione" per scavare nel modello e ispezionare le strutture interne lungo
                l'asse scelto.
              </p>

              <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", alignItems: "center", gap: 18, marginTop: 12 }}>
                {modelChoice === "cnn" && (
                  <label
                    htmlFor="gradcam-toggle"
                    style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 13, fontWeight: 600, color: "#5b6b79", cursor: "pointer" }}
                  >
                    <input
                      id="gradcam-toggle"
                      type="checkbox"
                      checked={showGradcam}
                      onChange={(e) => handleToggleGradcam(e.target.checked)}
                      aria-describedby="gradcam-help"
                    />
                    Mostra overlay Grad-CAM (interpretabilità CNN)
                  </label>
                )}

                <a
                  href={glbUrl}
                  download={`neuroage_modello_${(result?.filename || "cervello").split(".")[0]}.glb`}
                  className="na-focusable"
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 7,
                    fontSize: 12.5, fontWeight: 600, color: "#1d72c2", textDecoration: "none",
                    border: "1.5px solid #d3e3f1", padding: "7px 14px", borderRadius: 9, background: "#fff",
                  }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1d72c2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M12 15V3" />
                    <path d="M7 10l5 5 5-5" />
                    <path d="M4 21h16" />
                  </svg>
                  Scarica modello 3D (.glb)
                </a>
              </div>

              <p id="gradcam-help" style={{ fontSize: 11.5, color: "#a9bccb", textAlign: "center", marginTop: 8 }}>
                {showGradcam
                  ? "Le zone rosse del modello sono quelle che hanno influenzato di più la stima della CNN."
                  : "Colorazione per zona geometrica (illustrativa, non un atlante anatomico validato), utile per distinguere le strutture quando si seziona il modello."}
              </p>

              <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 14, marginTop: 10 }}>
                {(showGradcam
                  ? [
                      { label: "Bassa salienza", color: "#cfe3f2" },
                      { label: "Salienza intermedia", color: "#e88a78" },
                      { label: "Alta salienza", color: "#e62d28" },
                    ]
                  : REGION_LEGEND
                ).map((item) => (
                  <div key={item.label} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 11.5, color: "#5b6b79" }}>
                    <span style={{ width: 10, height: 10, borderRadius: 3, background: item.color, display: "inline-block" }} />
                    {item.label}
                  </div>
                ))}
              </div>
            </>
          )}
        </section>

        {/* ===== BENTO ROW: UPLOAD + RISULTATO ===== */}
        <section style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 22 }}>
          {/* ---- CARD UPLOAD ---- */}
          <div style={{ background: "#fff", border: "1px solid #e6eef4", borderRadius: 20, padding: 24 }}>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: "#1d72c2", marginBottom: 14, fontFamily: "'IBM Plex Mono', monospace", letterSpacing: ".4px" }}>
              01 · CARICA E ANALIZZA
            </div>

            <div
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              aria-label="Carica un volume MRI, clicca o trascina un file .nii o .nii.gz"
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
              className="na-focusable"
              style={{
                border: dragActive ? "2px solid #1d72c2" : "2px dashed #c8dceb",
                borderRadius: 16, padding: "26px 16px", textAlign: "center", cursor: "pointer",
                color: "#5b6b79", fontSize: 13.5,
                background: dragActive ? "#eaf4fb" : "#fafdff",
                marginBottom: 18, transition: "border-color .18s ease, background .18s ease",
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".nii,.gz"
                aria-label="Seleziona file MRI"
                style={{ display: "none" }}
                onChange={(e) => handleFileChange(e.target.files?.[0])}
              />
              {file ? (
                <span style={{ color: "#16222e", fontWeight: 600 }}>{file.name}</span>
              ) : (
                <>Trascina qui il volume MRI (.nii / .nii.gz)<br />o clicca per selezionarlo</>
              )}
            </div>

            <div style={{ fontSize: 12.5, fontWeight: 600, color: "#5b6b79", marginBottom: 8 }} id="model-label">
              Modello
            </div>
            <div role="group" aria-labelledby="model-label" style={{ display: "flex", gap: 8, marginBottom: 18 }}>
              {[
                { id: "cnn", label: "CNN 3D" },
                { id: "ensemble", label: "Ensemble ML" },
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => setModelChoice(m.id)}
                  aria-pressed={modelChoice === m.id}
                  className="na-focusable"
                  style={{
                    flex: 1, fontSize: 13, fontWeight: 600, padding: "10px 0", borderRadius: 10,
                    border: modelChoice === m.id ? "1.5px solid #1d72c2" : "1.5px solid #d3e3f1",
                    background: modelChoice === m.id ? "#eaf4fb" : "#fff",
                    color: modelChoice === m.id ? "#1d72c2" : "#5b6b79", cursor: "pointer",
                    transition: "all .18s ease",
                  }}
                >
                  {m.label}
                </button>
              ))}
            </div>

            <label htmlFor="chrono-age" style={{ fontSize: 12.5, fontWeight: 600, color: "#5b6b79", marginBottom: 8, display: "block" }}>
              Età anagrafica (per il brain-age gap)
            </label>
            <input
              id="chrono-age"
              type="number"
              min="0"
              max="120"
              placeholder="es. 42"
              value={chronoAge}
              onChange={(e) => setChronoAge(e.target.value)}
              className="na-focusable"
              style={{
                width: "100%", fontSize: 14, padding: "10px 14px", borderRadius: 10,
                border: "1.5px solid #d3e3f1", marginBottom: 20, boxSizing: "border-box",
              }}
            />

            <button
              onClick={handleAnalyze}
              disabled={!file || status === "loading"}
              className="na-focusable"
              style={{
                width: "100%", fontSize: 14.5, fontWeight: 600, color: "#fff",
                background: !file || status === "loading"
                  ? "#a9bccb"
                  : "linear-gradient(135deg,#1d72c2,#1565ad)",
                padding: "14px 0", borderRadius: 12, border: "none",
                cursor: !file || status === "loading" ? "default" : "pointer",
                boxShadow: !file || status === "loading" ? "none" : "0 8px 22px rgba(29,114,194,.28)",
                transition: "box-shadow .18s ease",
              }}
            >
              {status === "loading" ? "Analisi in corso…" : "Analizza"}
            </button>

            {status === "error" && (
              <div role="alert" style={{ marginTop: 14, fontSize: 13, color: "#c0392b" }}>{errorMsg}</div>
            )}
          </div>

          {/* ---- CARD RISULTATO ---- */}
          <div style={{ background: "#fff", border: "1px solid #e6eef4", borderRadius: 20, padding: 24 }}>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: "#15976a", marginBottom: 14, fontFamily: "'IBM Plex Mono', monospace", letterSpacing: ".4px" }}>
              02 · RISULTATO
            </div>

            {!result && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "calc(100% - 30px)", color: "#a9bccb", fontSize: 13.5, gap: 10 }}>
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#c8dceb" strokeWidth="1.6" aria-hidden="true">
                  <path d="M3 12h4l3 8 4-16 3 8h4" />
                </svg>
                In attesa di un'analisi
              </div>
            )}

            {result && (
              <div>
                <div style={{ fontSize: 12.5, color: "#8c99a6", marginBottom: 4 }}>Età cerebrale stimata</div>
                <div style={{ fontSize: 42, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1, marginBottom: 6 }}>
                  {result.predicted_age} <span style={{ fontSize: 17, color: "#8c99a6", fontWeight: 500 }}>anni</span>
                </div>
                {result.confidence_interval && (
                  <div style={{ fontSize: 12.5, color: "#8c99a6", marginBottom: 14 }}>
                    Intervallo plausibile: <strong style={{ color: "#16222e" }}>{result.confidence_interval[0]}–{result.confidence_interval[1]} anni</strong> (±{result.mae} anni, MAE del modello)
                  </div>
                )}

                <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 14 }}>
                  <div
                    style={{
                      display: "inline-flex", alignItems: "center", gap: 7,
                      fontSize: 12, fontWeight: 600, color: "#15976a", background: "#e4f5ee",
                      padding: "6px 12px", borderRadius: 99,
                    }}
                  >
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#15976a" }} />
                    Modello: {result.model === "cnn" ? "CNN 3D" : "Ensemble ML"}
                  </div>

                  {badge && (
                    <div
                      style={{
                        display: "inline-flex", alignItems: "center", gap: 7,
                        fontSize: 12, fontWeight: 600, color: badge.color, background: badge.bg,
                        padding: "6px 12px", borderRadius: 99,
                      }}
                    >
                      Δ {gap >= 0 ? "+" : "−"}{Math.abs(gap).toFixed(1)} anni · {badge.label}
                    </div>
                  )}
                </div>

                {!badge && (
                  <div style={{ fontSize: 12, color: "#a9bccb", marginBottom: 14 }}>
                    Inserisci l'età anagrafica per calcolare il brain-age gap.
                  </div>
                )}

                <button
                  onClick={handleDownloadPdf}
                  disabled={pdfLoading}
                  className="na-focusable"
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 8,
                    fontSize: 13, fontWeight: 600, color: "#1d72c2", background: "#eaf4fb",
                    border: "1.5px solid #c8dceb", padding: "10px 16px", borderRadius: 10,
                    cursor: pdfLoading ? "default" : "pointer", opacity: pdfLoading ? 0.7 : 1,
                  }}
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#1d72c2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M12 15V3" />
                    <path d="M7 10l5 5 5-5" />
                    <path d="M4 21h16" />
                  </svg>
                  {pdfLoading ? "Genero il referto…" : "Scarica referto PDF"}
                </button>

                <div style={{ marginTop: 18, paddingTop: 16, borderTop: "1px solid #eef3f7", fontSize: 12.5, color: "#8c99a6", lineHeight: 1.6 }}>
                  File analizzato: <strong style={{ color: "#16222e" }}>{result.filename}</strong>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ===== STORICO ANALISI ===== */}
        {history.length > 0 && (
          <section style={{ marginTop: 22, background: "#fff", border: "1px solid #e6eef4", borderRadius: 20, padding: 24 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: "#e2761b", fontFamily: "'IBM Plex Mono', monospace", letterSpacing: ".4px" }}>
                03 · STORICO ANALISI
              </div>
              <button
                onClick={clearHistory}
                className="na-focusable"
                style={{ fontSize: 12, fontWeight: 600, color: "#8c99a6", background: "none", border: "none", cursor: "pointer" }}
              >
                Cancella storico
              </button>
            </div>

            {history.length >= 2 && (
              <div style={{ marginBottom: 18, paddingBottom: 18, borderBottom: "1px solid #f0f4f7" }}>
                <div style={{ fontSize: 11.5, fontWeight: 600, color: "#8c99a6", marginBottom: 8 }}>
                  Andamento dell'età cerebrale stimata nel tempo
                </div>
                <TrendChart entries={history} />
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
              {history.map((h, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "12px 0", borderTop: i > 0 ? "1px solid #f0f4f7" : "none",
                    fontSize: 13, flexWrap: "wrap", gap: 8,
                  }}
                >
                  <div style={{ color: "#16222e", fontWeight: 600, minWidth: 160 }}>{h.filename}</div>
                  <div style={{ color: "#8c99a6", fontSize: 12 }}>{h.model === "cnn" ? "CNN 3D" : "Ensemble ML"}</div>
                  <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>{h.predicted_age} anni</div>
                  <div style={{ color: h.gap === null ? "#a9bccb" : gapBadge(h.gap).color, fontSize: 12.5, fontWeight: 600 }}>
                    {h.gap === null ? "—" : `Δ ${h.gap >= 0 ? "+" : "−"}${Math.abs(h.gap).toFixed(1)}`}
                  </div>
                  <div style={{ color: "#a9bccb", fontSize: 11.5 }}>
                    {new Date(h.date).toLocaleString("it-IT", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
