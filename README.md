# Brain Age Estimation

Stima dell'età cronologica di un soggetto a partire da una scansione MRI
3D del cervello, confrontando tecniche di Machine Learning classico e una
Convolutional Neural Network 3D, addestrate interamente **su CPU** (nessuna
GPU NVIDIA/CUDA disponibile in fase di sviluppo).

Progetto nato come esercitazione per il corso di Machine Learning
(UniNa – Federico II, A.A. 2025/26), successivamente esteso ed industrializzato
come progetto di portfolio.

## V2 — Dashboard web e ricostruzione 3D

Oltre alla pipeline di training, il progetto include un'applicazione web
(`backend/` + `frontend/`) che serve i modelli addestrati:

- **Backend FastAPI** (`backend/main.py`): predizione (`/predict`), ricostruzione
  della superficie cerebrale in 3D via marching cubes + trimesh con
  colorazione per zone anatomiche illustrative o overlay Grad-CAM (`/mesh`),
  e referto PDF clinico (`/report`).
- **Frontend React** (`frontend/`, Vite + Tailwind + Framer Motion +
  react-three-fiber): landing page con sfondo a rete neurale 3D e dashboard
  di analisi con il modello 3D ruotabile/sezionabile, brain-age gap,
  intervallo di confidenza, storico e trend longitudinale.

Avvio rapido:

```bash
# Backend (richiede i modelli in outputs/, vedi sotto)
cd backend && uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

In alternativa, con Docker:

```bash
docker compose up --build
```

## Risultati

| Approccio                                   | Feature / Input        | MAE (anni) |
|----------------------------------------------|------------------------|-----------:|
| SVR (kernel RBF)                             | 60 feature statistiche | 5.38 (CV) / 5.77 (test) |
| Ensemble SVR + RandomForest (tuned)          | 155 feature statistiche| 4.76 (CV)  |
| **CNN 3D**                                   | Volumi 64×64×64        | **4.06** (validation) |

Il leaderboard pubblico della competizione originale riportava un MAE minimo
di circa 2.2 anni, presumibilmente ottenuto con architetture deep learning
addestrate su GPU dedicata.

## Perché niente GPU?

Lo sviluppo è avvenuto su CPU Ryzen 5 5600X con GPU AMD RX 6600 XT. PyTorch
richiede CUDA (NVIDIA) per l'accelerazione hardware; `torch-directml`
(il bridge DirectML per GPU AMD/Intel su Windows) è stato testato ma **non
supporta l'operatore `Conv3d`**, rendendolo inutilizzabile per questa
architettura. Il training della CNN avviene quindi su CPU, con volumi
ridotti a 64³ (da 128³ originali) per mantenere tempi di addestramento
ragionevoli.

## Struttura del progetto

```
.
├── src/brain_age/          # package principale (training/feature engineering)
│   ├── config.py           # path e costanti centralizzate
│   ├── data/                # caricamento NIfTI, downsampling
│   ├── features/            # feature engineering (statistiche su volumi 3D)
│   ├── models/               # pipeline ML classico, CNN 3D, Grad-CAM, Dataset
│   └── training/             # loop di training e valutazione
├── backend/                 # API FastAPI (predizione, mesh 3D, referto PDF)
├── frontend/                 # app React (landing + dashboard di analisi)
├── scripts/                 # entry-point eseguibili per il training (CLI)
├── tests/                   # test automatici (pytest, incl. test_api.py)
├── notebooks/                # script di prototipazione/training (EDA, v2)
├── .github/workflows/        # CI (test backend + build frontend)
├── data/                     # dataset MRI (non versionato, vedi sotto)
├── outputs/                  # feature, modelli e submission generati (non versionato)
├── docker-compose.yml
└── pyproject.toml
```

## Setup

Richiede **Python 3.11** (necessario per la compatibilità con `torch-directml`,
testato ma non utilizzato nella pipeline finale per i limiti descritti sopra).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"
```

### Dataset

Il dataset (MRI in formato NIfTI + CSV con le età) non è incluso nel
repository per dimensioni e licenza. Va scaricato dalla competizione Kaggle
originale e posizionato secondo questa struttura:

```
data/
├── train/train/subj_XXXX.nii
└── test/test/subj_XXXX.nii
train.csv
test.csv
```

Il path di base del progetto può essere personalizzato con la variabile
d'ambiente `BRAIN_AGE_HOME` (di default è la cartella che contiene `src/`).

## Utilizzo

```bash
# 1. Estrazione feature statistiche (richiede ~10-15 minuti su CPU)
python scripts/run_feature_extraction.py

# 2. Training e valutazione modelli ML classici (Ridge, RF, SVR, GB, Ensemble)
python scripts/run_train_classical.py

# 3. (separatamente) Downsampling volumi per la CNN — script in notebooks/
#    poi training:
python scripts/run_train_cnn.py
```

## Metodologia

### Feature engineering (ML classico)

Un volume MRI 128³ contiene oltre 2 milioni di voxel: troppi per un modello
classico con solo 2328 campioni di training (overfitting garantito). Ogni
volume viene quindi riassunto in **155 feature statistiche**, motivate dal
fatto che l'invecchiamento cerebrale si manifesta come atrofia tissutale e
dilatazione ventricolare — fenomeni misurabili attraverso intensità e
distribuzione spaziale dei voxel:

- statistiche globali (media, percentili, skewness, kurtosis...)
- istogramma di intensità (10 bin)
- gradiente spaziale (Sobel 3D) — texture / bordi materia grigia-bianca
- statistiche per 8 ottanti e 6 fasce assiali — pattern spaziali locali
- rapporto centro/periferia — proxy della dilatazione ventricolare

### CNN 3D

Rete leggera (~300k parametri): 4 blocchi Conv3D+BatchNorm+ReLU+MaxPool che
riducono il volume da 64³ a 4³, seguiti da Global Average Pooling e due
layer fully-connected. Loss L1 (MAE diretto), data augmentation tramite flip
sagittale (il cervello ha simmetria bilaterale approssimativa), early
stopping e learning rate scheduling.

## Test

```bash
pytest tests/ -v
```

I test coprono: risoluzione dei path dei file MRI, downsampling dei volumi,
estrazione delle feature (incluso il comportamento su input degenerati, es.
volumi completamente vuoti), forward pass della CNN, e il Dataset PyTorch.

## Limiti noti

- Le feature statistiche perdono informazione spaziale fine rispetto ai
  voxel grezzi
- Gli ottanti e le fasce assiali sono suddivisioni generiche, non basate su
  atlanti neuroanatomici specifici (es. ippocampo, ventricoli)
- Il dataset è sbilanciato verso la fascia 19-30 anni, penalizzando
  potenzialmente l'accuratezza sulle fasce meno rappresentate

## Sviluppi futuri

- CNN 3D più profonda, addestrata su GPU dedicata (es. Kaggle Notebooks con
  GPU T4) e a risoluzione piena (128³)
- Feature da region-of-interest neuroanatomiche tramite atlanti standard
  (oggi la colorazione 3D per zona è solo geometrica/illustrativa, non
  basata su un atlante reale)
- Confronto longitudinale per paziente reale (oggi lo storico è per
  browser/localStorage, non per identità del paziente)

## Licenza

MIT
