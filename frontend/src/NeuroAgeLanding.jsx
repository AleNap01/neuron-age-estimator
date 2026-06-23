import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import NeuralBrainBackground from "./NeuralBrainBackground.jsx";
import { cn } from "./lib/utils.js";
import { useLenis } from "./useLenis.js";

gsap.registerPlugin(ScrollTrigger);

/**
 * NeuroAgeLanding — V3, ricostruita in stile "21st.dev": Tailwind CSS +
 * Framer Motion, mesh gradient di sfondo, glass card, marquee dello
 * stack tecnico, bento di feature con hover-glow e reveal-on-scroll.
 * Il "cervello digitale" (NeuralBrainBackground, three.js: nodi disposti
 * a forma di cervello e collegati come una rete neurale) vive dietro al
 * testo della hero, come elemento atmosferico — non più un oggetto da
 * manipolare in primo piano.
 */

const FEATURES = [
  {
    id: "3d",
    color: "text-brand-orange",
    glow: "from-brand-orange/25",
    tag: "MODELLO 3D",
    title: "Ricostruzione ruotabile",
    text: "Il volume MRI viene ricostruito in una mesh 3D che puoi ruotare, zoomare e sezionare per esplorare le strutture interne del cervello.",
    icon: (
      <>
        <circle cx="12" cy="12" r="8.2" />
        <ellipse cx="12" cy="12" rx="8.2" ry="3.4" />
      </>
    ),
    span: "sm:col-span-2",
  },
  {
    id: "ml",
    color: "text-brand-blue",
    glow: "from-brand-blue/25",
    tag: "ML CLASSICO",
    title: "Feature + ensemble",
    text: "Misure morfometriche date in pasto a un ensemble di SVR e Random Forest.",
    icon: <path d="M4 19V5M4 19h16M8 15l3-4 3 2 4-6" />,
  },
  {
    id: "cnn",
    color: "text-brand-green",
    glow: "from-brand-green/25",
    tag: "CNN 3D",
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
    id: "pdf",
    color: "text-emerald-600",
    glow: "from-emerald-500/25",
    tag: "REFERTO",
    title: "Report scaricabile",
    text: "Un documento con età stimata, brain-age gap e mappe, pronto da condividere.",
    icon: (
      <>
        <path d="M6 3.5h8l4 4V20a.5.5 0 0 1-.5.5H6.5A.5.5 0 0 1 6 20V4a.5.5 0 0 1 .5-.5Z" />
        <path d="M14 3.5V8h4" />
      </>
    ),
    span: "sm:col-span-2",
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

// Reveal — entrata marcata su scroll (blur + scale + risalita), non il fade-up
// uniforme da "AI slop": ogni elemento parte visibilmente fuori fuoco e si
// "messa a fuoco" mentre arriva, coerente con un prodotto che ricostruisce
// nitidezza da un volume MRI.
function Reveal({ children, className, delay = 0 }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    const mm = gsap.matchMedia();
    mm.add(
      { reduced: "(prefers-reduced-motion: reduce)", full: "(prefers-reduced-motion: no-preference)" },
      (ctx) => {
        if (ctx.conditions.reduced) {
          gsap.set(el, { opacity: 1, y: 0, scale: 1, filter: "blur(0px)" });
          return;
        }
        gsap.set(el, { opacity: 0, y: 56, scale: 0.92, filter: "blur(10px)" });
        const tween = gsap.to(el, {
          opacity: 1,
          y: 0,
          scale: 1,
          filter: "blur(0px)",
          duration: 0.9,
          delay,
          ease: "expo.out",
          scrollTrigger: {
            trigger: el,
            start: "top 85%",
            toggleActions: "play none none reverse",
          },
        });
        return () => tween.scrollTrigger?.kill();
      }
    );
    return () => mm.revert();
  }, [delay]);

  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}

// Tilt — inclinazione 3D che segue il mouse, per le card della tecnologia.
function useTilt() {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const onMove = (e) => {
      const rect = el.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width - 0.5;
      const py = (e.clientY - rect.top) / rect.height - 0.5;
      gsap.to(el, {
        rotateX: py * -10,
        rotateY: px * 12,
        scale: 1.03,
        duration: 0.4,
        ease: "expo.out",
        transformPerspective: 700,
      });
    };
    const onLeave = () => {
      gsap.to(el, { rotateX: 0, rotateY: 0, scale: 1, duration: 0.5, ease: "expo.out" });
    };

    el.addEventListener("mousemove", onMove);
    el.addEventListener("mouseleave", onLeave);
    return () => {
      el.removeEventListener("mousemove", onMove);
      el.removeEventListener("mouseleave", onLeave);
    };
  }, []);

  return ref;
}

function GlowIcon({ children, colorClass }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" className={colorClass}>
      {children}
    </svg>
  );
}

function FeatureCard({ f }) {
  const tiltRef = useTilt();
  return (
    <div
      ref={tiltRef}
      className="group relative h-full overflow-hidden rounded-2xl border border-black/5 bg-white p-6 transition-shadow will-change-transform hover:shadow-[0_18px_40px_rgba(22,34,46,.10)]"
      style={{ transformStyle: "preserve-3d" }}
    >
      <div className={cn("absolute -right-8 -top-8 h-28 w-28 rounded-full bg-gradient-to-br to-transparent opacity-0 blur-2xl transition-opacity group-hover:opacity-100", f.glow)} />
      <div className="mb-4 flex items-center gap-3">
        <div className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-black/[0.03]", f.color)}>
          <GlowIcon colorClass={f.color}>{f.icon}</GlowIcon>
        </div>
        <span className={cn("font-mono text-[11px] font-medium tracking-wide", f.color)}>{f.tag}</span>
      </div>
      <div className="mb-1.5 text-[15px] font-semibold">{f.title}</div>
      <div className="text-[13px] leading-relaxed text-muted">{f.text}</div>
    </div>
  );
}

export default function NeuroAgeLanding({ onUploadClick }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const heroRef = useRef(null);
  const heroSectionRef = useRef(null);
  const brainBgRef = useRef(null);
  const progressBarRef = useRef(null);
  const handleUploadClick = () => onUploadClick?.();
  const closeMenu = () => setMenuOpen(false);

  useLenis();

  // 1 · Hero load — timeline GSAP unica: headline → paragrafo → CTA → stat line.
  useEffect(() => {
    const mm = gsap.matchMedia();
    mm.add(
      { reduced: "(prefers-reduced-motion: reduce)", full: "(prefers-reduced-motion: no-preference)" },
      (ctx) => {
        const els = heroRef.current.querySelectorAll("[data-hero-item]");
        if (ctx.conditions.reduced) {
          gsap.set(els, { opacity: 1, y: 0 });
          return;
        }
        gsap.set(els, { opacity: 0, y: 18 });
        gsap.to(els, {
          opacity: 1,
          y: 0,
          duration: 0.7,
          ease: "expo.out",
          stagger: 0.09,
        });
      }
    );
    return () => mm.revert();
  }, []);

  // 2 · Scroll: parallax sul cervello digitale mentre la hero esce dal viewport.
  // 3 · Sticky progress — barra sottile nella navbar legata allo scroll dell'intera pagina.
  useEffect(() => {
    const mm = gsap.matchMedia();
    mm.add(
      { reduced: "(prefers-reduced-motion: reduce)", full: "(prefers-reduced-motion: no-preference)" },
      (ctx) => {
        if (ctx.conditions.reduced) return;

        const triggers = [];

        if (brainBgRef.current && heroSectionRef.current) {
          gsap.set(brainBgRef.current, { willChange: "transform" });
          triggers.push(
            ScrollTrigger.create({
              trigger: heroSectionRef.current,
              start: "top top",
              end: "bottom top",
              scrub: true,
              animation: gsap.to(brainBgRef.current, { yPercent: -32, scale: 1.18, ease: "none" }),
            })
          );
        }

        if (progressBarRef.current) {
          triggers.push(
            ScrollTrigger.create({
              trigger: document.documentElement,
              start: "top top",
              end: "bottom bottom",
              onUpdate: (self) => {
                gsap.set(progressBarRef.current, { scaleX: self.progress });
              },
            })
          );
        }

        return () => triggers.forEach((t) => t.kill());
      }
    );
    return () => mm.revert();
  }, []);

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
        <div className="relative mx-auto flex max-w-5xl items-center justify-between rounded-2xl border border-white/60 bg-white/70 px-5 py-3 shadow-[0_8px_30px_rgba(22,34,46,.08)] backdrop-blur-xl">
          <div className="absolute inset-x-5 bottom-0 h-px overflow-hidden rounded-full bg-black/5" aria-hidden="true">
            <div
              ref={progressBarRef}
              className="h-full w-full origin-left bg-gradient-to-r from-brand-blue to-brand-green"
              style={{ transform: "scaleX(0)" }}
            />
          </div>
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

          <div className="flex items-center gap-2">
            <button
              onClick={handleUploadClick}
              className="cursor-pointer rounded-xl bg-ink px-4 py-2 text-sm font-semibold text-white shadow-sm transition-transform hover:scale-[1.03] active:scale-[0.98]"
            >
              Apri la dashboard
            </button>
            <button
              onClick={() => setMenuOpen((v) => !v)}
              aria-label={menuOpen ? "Chiudi il menu" : "Apri il menu"}
              aria-expanded={menuOpen}
              aria-controls="mobile-nav-panel"
              className="flex h-9 w-9 cursor-pointer items-center justify-center rounded-xl text-ink transition-colors hover:bg-black/5 sm:hidden"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                {menuOpen ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
              </svg>
            </button>
          </div>
        </div>

        <AnimatePresence>
          {menuOpen && (
            <motion.nav
              id="mobile-nav-panel"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
              className="mx-auto mt-2 flex max-w-5xl flex-col gap-1 rounded-2xl border border-white/60 bg-white/90 p-3 shadow-[0_8px_30px_rgba(22,34,46,.08)] backdrop-blur-xl sm:hidden"
            >
              <a href="#tecnologia" onClick={closeMenu} className="rounded-xl px-3 py-2.5 text-sm font-medium text-muted hover:bg-black/5 hover:text-ink transition-colors">Tecnologia</a>
              <a href="#come-funziona" onClick={closeMenu} className="rounded-xl px-3 py-2.5 text-sm font-medium text-muted hover:bg-black/5 hover:text-ink transition-colors">Come funziona</a>
            </motion.nav>
          )}
        </AnimatePresence>
      </header>

      {/* ===== HERO ===== */}
      <section ref={heroSectionRef} className="relative overflow-hidden pt-40 pb-28 sm:pt-48 lg:pb-36">
        {/* Cervello digitale: rete neurale di sfondo, dietro al testo, con leggero parallax allo scroll */}
        <div ref={brainBgRef} className="absolute inset-0 -z-0">
          <NeuralBrainBackground className="absolute inset-0 opacity-90" />
        </div>
        {/* Scrim di leggibilità: sfuma la rete verso il colore di pagina ai bordi */}
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_55%_at_50%_38%,transparent_0%,rgba(246,250,252,.55)_72%,#f6fafc_100%)]" />

        <div ref={heroRef} className="relative z-10 mx-auto max-w-3xl px-6 text-center">
          <h1 data-hero-item className="text-[clamp(2.4rem,6vw,3.9rem)] font-extrabold leading-[1.02] tracking-tight text-balance">
            Quanti anni ha
            <br />
            <span className="text-brand-blue">davvero il tuo cervello?</span>
          </h1>

          <p data-hero-item className="mx-auto mt-6 max-w-lg text-balance text-[clamp(0.95rem,1.6vw,1.125rem)] leading-relaxed text-muted">
            NeuroAge·MRI stima l'età biologica del cervello da una risonanza magnetica e ne
            ricostruisce un <strong className="text-ink">modello 3D ruotabile e sezionabile</strong>,
            combinando machine learning classico e reti neurali 3D.
          </p>

          <div data-hero-item className="mt-9 flex flex-wrap items-center justify-center gap-3.5">
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

          <p data-hero-item className="mx-auto mt-10 max-w-xl text-balance text-[13px] leading-relaxed text-faint">
            Addestrato su <span className="font-mono text-ink">2.3k</span> volumi cerebrali, con un errore medio
            di <span className="font-mono text-ink">± 4.06 anni</span> (CNN 3D) e zero GPU richieste in fase di inferenza.
          </p>
        </div>
      </section>

      {/* ===== TECNOLOGIA (bento) ===== */}
      <section id="tecnologia" className="mx-auto max-w-6xl px-6 py-24">
        <Reveal className="mb-12 text-center">
          <h2 className="text-[clamp(1.6rem,3.4vw,2.1rem)] font-bold tracking-tight">
            La tecnologia dietro la stima
          </h2>
        </Reveal>

        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((f, i) => (
            <Reveal key={f.id} className={f.span} delay={i * 0.08}>
              <FeatureCard f={f} />
            </Reveal>
          ))}
        </div>
      </section>

      {/* ===== COME FUNZIONA ===== */}
      <section id="come-funziona" className="mx-auto max-w-6xl px-6 py-12">
        <Reveal className="mb-14 text-center">
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
          <div className="flex items-center gap-2.5 text-[13px] text-muted">
            <div className="h-6 w-6 shrink-0 rounded-[7px] bg-gradient-to-br from-brand-blue to-brand-green" />
            NeuroAge·MRI — strumento di supporto alla ricerca, non destinato all'uso diagnostico autonomo.
          </div>
          <div className="font-mono text-xs text-faint/80">© 2026 · v3.0</div>
        </div>
      </footer>
    </div>
  );
}
