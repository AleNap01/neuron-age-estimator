# NeuroAge · MRI — Frontend

Landing page React (Vite) per il progetto Brain Age Estimation. Design
originale creato con Claude Design, tradotto in componenti React funzionali.

## Setup

```bash
npm install
npm run dev
```

Apre su `http://localhost:5173`.

## Struttura

```
src/
├── main.jsx                  # entry point
├── App.jsx                   # navigazione landing/dashboard (useState)
├── NeuroAgeLanding.jsx       # landing page con cervello SVG animato e hotspot
└── DashboardPlaceholder.jsx  # segnaposto per la dashboard funzionale
```

## Stato attuale

- ✅ Landing page completa e funzionante (hero animato, hotspot interattivi,
  sezione "come funziona", CTA, footer)
- ⬜ Dashboard funzionale: oggi la vera logica (upload MRI, predizione,
  Grad-CAM, referto PDF) vive nell'app Streamlit del progetto
  [neuron-age-estimator](https://github.com/AleNap01/neuron-age-estimator).
  Da collegare tramite:
  1. iframe verso l'app Streamlit (più rapido, meno integrato), oppure
  2. un'API REST (FastAPI) che esponga i modelli Python, chiamata da
     questo frontend via fetch — la strada più solida per un prodotto reale

## Build per produzione

```bash
npm run build
```

Genera `dist/`, pronta per essere servita da qualsiasi hosting statico
(Vercel, Netlify, GitHub Pages, ecc.).
