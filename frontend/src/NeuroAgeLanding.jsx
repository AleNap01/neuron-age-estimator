import { motion } from "framer-motion";
import NeuralBrainBackground from "./NeuralBrainBackground.jsx";
import { cn } from "./lib/utils.js";

/**
 * NeuroAgeLanding — V3, ricostruita in stile "21st.dev": Tailwind CSS +
 * Framer Motion, mesh gradient di sfondo, glass card, marquee dello
 * stack tecnico, bento di feature con hover-glow e reveal-on-scroll.
 * Il "cervello digitale" (NeuralBrainBackground, three.js: nodi disposti
 * a forma di cervello e collegati come una rete neurale) vive dietro al
 * testo della hero, come elemento atmosferico — non più un oggetto da
 * manipolare in primo piano.
 */

const STACK = [
  "PYTORCH", "FASTAPI", "REACT THREE FIBER", "SCIKIT-LEARN",
  "NIBABEL", "TRIMESH", "THREE.JS", "FRAMER MOTION",
];

const FEATURES = [
  {
    id: "ml",
    color: "text-brand-blue",
    glow: "from-brand-blue/25",
    eyebrow: "01 · ML CLASSICO",
    title: "Feature + ensemble",
    text: "Misure morfometriche del cervello date in pasto a un ensemble di SVR e Random Forest.",
    icon: <path d="M4 19V5M4 19h16M8 15l3-4 3 2 4-6" />,
  },
  {
    id: "cnn",
    color: "text-brand-green",
    glow: "from-brand-green/25",
    eyebrow: "02 · CNN 3D",
    title: "Rete volumetrica",
    text: "Una rete neurale 3D addestrata direttamente sui volumi cerebrali, end-to-end.",
    icon: (
      <>
        <rect x="4" y="4" width="7" height="7" rx="1.2" />
        <rect x="13" y="4" width="7" height="7" rx="1.2" />
        <rect x="4" y="13" width="7" height="7" rx="1.2" />
        <rect x="13" y="13" width="7" height="7" rx="1.2" />
      </>
    ),
  },
  {
    id: "3d",
    color: "text-brand-orange",
    glow: "from-brand-orange/25",
    eyebrow: "03 · MODELLO 3D",
    title: "Ricostruzione ruotabile",
    text: "Il volume MRI viene ricostruito in una mesh 3D: ruotala, zoomala, sezionala.",
    icon: (
      <>
        <circle cx="12" cy="12" r="8.2" />
        <ellipse cx="12" cy="12" rx="8.2" ry="3.4" />
      </>
    ),
  },
  {
    id: "pdf",
    color: "text-emerald-600",
    glow: "from-emerald-500/25",
    eyebrow: "04 · REFERTO",
    title: "Report scaricabile",
    text: "Un documento con età stimata, brain-age gap e mappe, pronto da condividere.",
    icon: (
      <>
        <path d="M6 3.5h8l4 4V20a.5.5 0 0 1-.5.5H6.5A.5.5 0 0 1 6 20V4a.5.5 0 0 1 .5-.5Z" />
        <path d="M14 3.5V8h4" />
      </>
    ),
  },
];

const STEPS = [
  {
    n: "01",
    color: "bg-brand-blue",
    title: "Carica la MRI",
    text: "Trascina un volume T1w (.nii, .nii.gz o DICOM). Skull-stripping e normalizzazione sono automatici.",
  },
  {
    n: "02",
    color: "bg-brand-green",
    title: "Scegli il modello AI",
    text: "Ensemble ML classico o CNN 3D: la rete elabora il volume e produce la stima dell'età cerebrale.",
  },
  {
    n: "03",
    color: "bg-brand-orange",
    title: "Esplora il modello 3D",
    text: "Ruota, zooma e sezione la ricostruzione 3D, poi leggi il referto con età stimata e brain-age gap.",
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 22 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } },
};

function Reveal({ children, className, delay = 0 }) {
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-60px" }}
      variants={fadeUp}
      transition={{ delay }}
    >
      {children}
    </motion.div>
  );
}

function GlowIcon({ children, colorClass }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" className={colorClass}>
      {children}
    </svg>
  );
}

export default function NeuroAgeLanding({ onUploadClick }) {
  const handleUploadClick = () => onUploadClick?.();

  return (
    <div className="min-h-screen overflow-x-hidden bg-mist text-ink font-sans selection:bg-brand-blue/20">
      {/* ===== BACKGROUND MESH ===== */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(1200px_700px_at_82%_-10%,#eaf4fb,#f6fafc_45%)]" />
        <div className="absolute -top-32 -left-24 h-[420px] w-[420px] rounded-full bg-brand-blue/15 blur-3xl animate-blob" />
        <div className="absolute top-[40%] -right-24 h-[460px] w-[460px] rounded-full bg-brand-green/15 blur-3xl animate-blob [animation-delay:3s]" />
        <div
          className="absolute inset-0 opacity-[0.035]"
          style={{
            backgroundImage:
              "linear-gradient(to right, #16222e 1px, transparent 1px), linear-gradient(to bottom, #16222e 1px, transparent 1px)",
            backgroundSize: "44px 44px",
          }}
        />
      </div>

      {/* ===== NAVBAR (floating, glass) ===== */}
      <header className="fixed inset-x-4 top-4 z-50 sm:inset-x-6">
        <div className="mx-auto flex max-w-5xl items-center justify-between rounded-2xl border border-white/60 bg-white/70 px-5 py-3 shadow-[0_8px_30px_rgba(22,34,46,.08)] backdrop-blur-xl">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-[10px] bg-gradient-to-br from-brand-blue to-brand-green shadow-[0_4px_12px_rgba(29,114,194,.3)]">
              <svg width="18" height="18" viewBox="0 0 20 20" fill="none" aria-hidden="true">
                <circle cx="10" cy="10" r="7.4" stroke="#fff" strokeWidth="1.3" opacity="0.85" />
                <circle cx="10" cy="10" r="3.9" stroke="#fff" strokeWidth="1.3" />
                <circle cx="10" cy="10" r="1.4" fill="#fff" />
              </svg>
            </div>
            <span className="text-[15px] font-bold tracking-tight">
              NeuroAge<span className="text-brand-blue">·</span>MRI
            </span>
          </div>

          <nav className="hidden items-center gap-7 sm:flex">
            <a href="#tecnologia" className="text-sm font-medium text-muted hover:text-ink transition-colors">Tecnologia</a>
            <a href="#come-funziona" className="text-sm font-medium text-muted hover:text-ink transition-colors">Come funziona</a>
          </nav>

          <button
            onClick={handleUploadClick}
            className="cursor-pointer rounded-xl bg-ink px-4 py-2 text-sm font-semibold text-white shadow-sm transition-transform hover:scale-[1.03] active:scale-[0.98]"
          >
            Apri la dashboard
          </button>
        </div>
      </header>

      {/* ===== HERO ===== */}
      <section className="relative overflow-hidden pt-40 pb-28 sm:pt-48 lg:pb-36">
        {/* Cervello digitale: rete neurale di sfondo, dietro al testo */}
        <NeuralBrainBackground className="absolute inset-0 -z-0 opacity-90" />
        {/* Scrim di leggibilità: sfuma la rete verso il colore di pagina ai bordi */}
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_55%_at_50%_38%,transparent_0%,rgba(246,250,252,.55)_72%,#f6fafc_100%)]" />

        <motion.div
          className="relative z-10 mx-auto max-w-3xl px-6 text-center"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-brand-green/20 bg-mint px-3.5 py-1.5 font-mono text-xs font-medium text-brand-green">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-brand-green" />
            AI · Neuroimaging · Cervello digitale
          </div>

          <h1 className="text-[clamp(2.4rem,6vw,3.9rem)] font-extrabold leading-[1.02] tracking-tight">
            Quanti anni ha
            <br />
            <span className="bg-gradient-to-r from-brand-blue to-brand-green bg-clip-text text-transparent">
              davvero il tuo cervello?
            </span>
          </h1>

          <p className="mx-auto mt-6 max-w-lg text-balance text-[clamp(0.95rem,1.6vw,1.125rem)] leading-relaxed text-muted">
            NeuroAge·MRI stima l'età biologica del cervello da una risonanza magnetica e ne
            ricostruisce un <strong className="text-ink">modello 3D ruotabile e sezionabile</strong>,
            combinando machine learning classico e reti neurali 3D.
          </p>

          <div className="mt-9 flex flex-wrap items-center justify-center gap-3.5">
            <button
              onClick={handleUploadClick}
              className="group inline-flex cursor-pointer items-center gap-2.5 rounded-2xl bg-gradient-to-br from-brand-blue to-brand-blue-dark px-7 py-4 text-[15px] font-semibold text-white shadow-[0_10px_30px_rgba(29,114,194,.32)] transition-transform hover:scale-[1.03] active:scale-[0.98]"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="transition-transform group-hover:-translate-y-0.5" aria-hidden="true">
                <path d="M12 15V3" />
                <path d="M7 8l5-5 5 5" />
                <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
              </svg>
              Carica la tua MRI
            </button>
            <a
              href="#come-funziona"
              className="inline-flex items-center gap-2 rounded-2xl border border-black/10 bg-white/80 px-6 py-4 text-[15px] font-semibold text-ink backdrop-blur transition-colors hover:border-brand-blue/30"
            >
              Scopri come funziona
            </a>
          </div>

          <div className="mt-12 flex flex-wrap items-center justify-center gap-7">
            {[
              ["± 4.06", "anni MAE", "errore medio CNN 3D"],
              ["2.3k", "", "volumi di addestramento"],
              ["100%", "CPU", "nessuna GPU richiesta"],
            ].map(([big, suffix, caption], i) => (
              <div key={i} className="flex items-center gap-7">
                {i > 0 && <div className="hidden h-9 w-px bg-black/10 sm:block" />}
                <div>
                  <div className="font-mono text-[22px] font-semibold">
                    {big} {suffix && <span className="text-[13px] font-medium text-faint">{suffix}</span>}
                  </div>
                  <div className="mt-0.5 text-xs text-faint">{caption}</div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* ===== MARQUEE STACK ===== */}
      <div className="relative border-y border-black/5 bg-white/60 py-4 backdrop-blur">
        <div className="flex overflow-hidden">
          <div className="flex shrink-0 animate-marquee items-center gap-12 pr-12">
            {[...STACK, ...STACK].map((item, i) => (
              <span key={i} className="font-mono text-xs font-medium tracking-wide text-faint whitespace-nowrap">
                {item}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ===== TECNOLOGIA (bento) ===== */}
      <section id="tecnologia" className="mx-auto max-w-6xl px-6 py-24">
        <Reveal className="mb-12 text-center">
          <div className="mb-2.5 font-mono text-xs font-medium uppercase tracking-wider text-brand-green">Tecnologia</div>
          <h2 className="text-[clamp(1.6rem,3.4vw,2.1rem)] font-bold tracking-tight">Quattro pezzi, una pipeline</h2>
        </Reveal>

        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((f, i) => (
            <Reveal key={f.id} delay={i * 0.08}>
              <div className="group relative h-full overflow-hidden rounded-2xl border border-black/5 bg-white p-6 transition-all hover:-translate-y-1 hover:shadow-[0_18px_40px_rgba(22,34,46,.10)]">
                <div className={cn("absolute -right-8 -top-8 h-28 w-28 rounded-full bg-gradient-to-br to-transparent opacity-0 blur-2xl transition-opacity group-hover:opacity-100", f.glow)} />
                <div className={cn("mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-black/[0.03]", f.color)}>
                  <GlowIcon colorClass={f.color}>{f.icon}</GlowIcon>
                </div>
                <div className={cn("mb-2 font-mono text-[11px] font-medium tracking-wide", f.color)}>{f.eyebrow}</div>
                <div className="mb-1.5 text-[15px] font-semibold">{f.title}</div>
                <div className="text-[13px] leading-relaxed text-muted">{f.text}</div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ===== COME FUNZIONA ===== */}
      <section id="come-funziona" className="mx-auto max-w-6xl px-6 py-12">
        <Reveal className="mb-14 text-center">
          <div className="mb-2.5 font-mono text-xs font-medium uppercase tracking-wider text-brand-blue">Come funziona</div>
          <h2 className="text-[clamp(1.6rem,3.4vw,2.1rem)] font-bold tracking-tight">Dalla risonanza al modello 3D in 3 passi</h2>
        </Reveal>

        <div className="grid gap-6 sm:grid-cols-3">
          {STEPS.map((step, i) => (
            <Reveal key={step.n} delay={i * 0.1}>
              <div className="relative h-full rounded-2xl border border-black/5 bg-white p-7">
                <div className={cn("mb-5 flex h-10 w-10 items-center justify-center rounded-[11px] font-mono text-[13px] font-semibold text-white", step.color)}>
                  {step.n}
                </div>
                <div className="mb-2 text-[16px] font-semibold">{step.title}</div>
                <div className="text-[13.5px] leading-relaxed text-muted">{step.text}</div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ===== CTA BAND ===== */}
      <section className="mx-auto max-w-6xl px-6 py-20">
        <Reveal>
          <div className="relative flex flex-col items-center gap-7 overflow-hidden rounded-3xl bg-gradient-to-br from-brand-blue to-brand-green px-8 py-14 text-center shadow-[0_24px_60px_rgba(29,114,194,.28)] sm:flex-row sm:justify-between sm:text-left">
            <div className="absolute -right-16 -top-16 h-72 w-72 rounded-full border border-white/15" />
            <div className="absolute -bottom-24 right-4 h-52 w-52 rounded-full border border-white/10" />

            <div className="relative max-w-md">
              <h2 className="text-[clamp(1.5rem,3.2vw,2rem)] font-bold leading-tight text-white">
                Pronto a scoprire l'età del tuo cervello?
              </h2>
              <p className="mt-3 text-[15px] leading-relaxed text-white/90">
                Carica la risonanza, ottieni una stima in meno di due minuti ed esplora il modello
                3D del tuo cervello, tutto nel browser.
              </p>
            </div>

            <button
              onClick={handleUploadClick}
              className="relative inline-flex cursor-pointer items-center gap-2.5 rounded-2xl bg-white px-7 py-4 text-[15px] font-semibold text-brand-blue shadow-[0_10px_26px_rgba(0,0,0,.18)] transition-transform hover:scale-[1.03] active:scale-[0.98]"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1d72c2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 15V3" />
                <path d="M7 8l5-5 5 5" />
                <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
              </svg>
              Carica la tua MRI
            </button>
          </div>
        </Reveal>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="border-t border-black/5">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-7">
          <div className="flex items-center gap-2.5 text-[13px] text-faint">
            <div className="h-6 w-6 rounded-[7px] bg-gradient-to-br from-brand-blue to-brand-green" />
            NeuroAge·MRI — strumento di supporto alla ricerca, non destinato all'uso diagnostico autonomo.
          </div>
          <div className="font-mono text-xs text-faint/80">© 2026 · v3.0</div>
        </div>
      </footer>
    </div>
  );
}
