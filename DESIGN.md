---
name: NeuroAge·MRI
description: Stima clinica dell'età biologica del cervello da risonanza magnetica, con ricostruzione 3D interattiva
colors:
  brand-blue: "#1d72c2"
  brand-blue-dark: "#1565ad"
  brand-green: "#15976a"
  brand-orange: "#e2761b"
  ink: "#16222e"
  muted: "#5b6b79"
  faint: "#8c99a6"
  mist: "#f6fafc"
  haze: "#eaf4fb"
  mint: "#e4f5ee"
typography:
  display:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "clamp(2.5rem, 6vw, 4rem)"
    fontWeight: 800
    lineHeight: 1.05
    letterSpacing: "-0.02em"
  body:
    fontFamily: "IBM Plex Sans, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "IBM Plex Mono, monospace"
    fontSize: "0.8125rem"
    fontWeight: 500
    letterSpacing: "0.02em"
rounded:
  sm: "10px"
  md: "12px"
  lg: "16px"
  xl: "24px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "28px"
  xl: "56px"
components:
  button-primary:
    backgroundColor: "{colors.brand-blue}"
    textColor: "#ffffff"
    rounded: "{rounded.xl}"
    padding: "16px 28px"
  button-primary-hover:
    backgroundColor: "{colors.brand-blue-dark}"
  button-secondary:
    backgroundColor: "{colors.ink}"
    textColor: "#ffffff"
    rounded: "{rounded.md}"
    padding: "8px 16px"
  card:
    backgroundColor: "#ffffff"
    textColor: "{colors.ink}"
    rounded: "{rounded.xl}"
    padding: "24px"
---

# Design System: NeuroAge·MRI

## 1. Overview

**Creative North Star: "Lo strumento da laboratorio di precisione"**

NeuroAge·MRI non è un gadget AI né un modulo sanitario asettico: è uno strumento clinico-scientifico che mostra il proprio lavoro. La superficie è chiara, quasi clinica nel suo ordine — sfondo quasi-bianco (mist), tipografia tecnica IBM Plex (Sans per la voce, Mono per i dati e i badge) — ma è animata da blob di colore brand-blue/brand-green sfocati e da un cervello digitale 3D in rotazione lenta, che portano calore e movimento senza intaccare la leggibilità dei numeri clinici. La fiducia nasce dalla precisione visibile (badge, intervalli di confidenza, microcopy mono per i dati) accompagnata da un'esecuzione morbida (radius ampi, ombre diffuse, transizioni leggere su hover) — non da decorazione gratuita.

Il sistema rifiuta esplicitamente lo stile SaaS da "AI slop" (niente sfondo crema/sand, niente eyebrow tutto-maiuscolo ripetuto su ogni sezione, niente gradient text, niente hero-metric template, niente card grid identiche) e rifiuta altrettanto l'estetica freddo-ospedaliera da modulistica sanitaria. Niente dark mode.

**Key Characteristics:**
- Sfondo quasi-bianco (`mist` #f6fafc) come base, mai crema/sand.
- Coppia cromatica blu/verde (`brand-blue` #1d72c2, `brand-green` #15976a) come accenti primario e di conferma, non come decorazione diffusa.
- IBM Plex Mono riservato a dati, badge, e label tecniche — mai per il corpo del testo.
- Radius generosi (xl: 24px) su CTA e card, mai spigoli vivi né radius minimi da "tool enterprise".
- Ombre sempre colorate e diffuse (tinte sul colore del CTA stesso), mai grigio neutro generico.

## 2. Colors

La palette è "Committed sobrio": due colori saturi (blue, green) portano significato clinico (azione primaria vs. stato positivo/conferma), su una base quasi-bianca che lascia parlare i numeri.

### Primary
- **Brand Blue** (#1d72c2): CTA primarie, link, focus ring, accento dominante della hero e della nav.
- **Brand Blue Dark** (#1565ad): stato hover/active di Brand Blue, mai un colore a sé stante nel resto della UI.

### Secondary
- **Brand Green** (#15976a): stato "nella norma"/conferma (badge gap sotto soglia, indicatori "ok"), bilancia il blu nei blob decorativi e nei gradient CTA.

### Tertiary
- **Brand Orange** (#e2761b): riservato a stati di attenzione/warning (es. badge gap sopra soglia). Usato con parsimonia, mai come accento decorativo.

### Neutral
- **Ink** (#16222e): testo principale, massimo contrasto su mist/white.
- **Muted** (#5b6b79): testo secondario, sottotitoli, paragrafi di supporto.
- **Faint** (#8c99a6): testo terziario, placeholder, metadata minori.
- **Mist** (#f6fafc): sfondo body di default. Non crema, non sand — bianco con un filo di blu.
- **Haze** (#eaf4fb): sfondo di sezioni alternate o blocchi informativi a basso contrasto.
- **Mint** (#e4f5ee): sfondo dei badge/pill in stato "verde" (coerente con brand-green).

### Named Rules
**The No-Cream Rule.** Il body background è sempre `mist` (#f6fafc) o bianco puro, mai un neutro tinto verso il caldo (crema/sand/parchment). Il calore del brand passa da blue+green, non dal background.

**The Mono-For-Data Rule.** IBM Plex Mono è riservato esclusivamente a numeri clinici, badge, timestamp e label tecniche (es. "ID referto", età stimata, gap). Il corpo testo e i titoli restano sempre in IBM Plex Sans.

## 3. Typography

**Display Font:** IBM Plex Sans (con fallback system-ui, sans-serif)
**Body Font:** IBM Plex Sans
**Label/Mono Font:** IBM Plex Mono

**Character:** Una sola famiglia umanista-tecnica (IBM Plex) in pesi diversi per display/body, contrastata da IBM Plex Mono per i dati — coppia che comunica precisione scientifica senza ricorrere a un serif "editoriale" fuori registro per un prodotto clinico.

### Hierarchy
- **Display** (800, `clamp(2.5rem, 6vw, 4rem)`, line-height 1.05, letter-spacing -0.02em): headline hero, claim principale.
- **Headline** (700, ~2rem, line-height 1.15): titoli di sezione (es. "Come funziona", "Risultati").
- **Title** (600, ~1.25rem, line-height 1.3): titoli di card/feature, nomi degli step.
- **Body** (400, 1rem, line-height 1.6, max 65–75ch): paragrafi descrittivi, copy della dashboard.
- **Label** (500, 0.8125rem, letter-spacing 0.02em, IBM Plex Mono): badge, pill di stato, ID referto, valori numerici nei risultati clinici.

### Named Rules
**The Display-Weight Ceiling Rule.** Il peso 800 e il clamp 4rem sono il tetto: oltre, l'headline grida invece di comunicare autorità clinica.

## 4. Elevation

Sistema flat-by-default con ombre come risposta a stato, non come decorazione statica. Niente shadow grigia generica: ogni ombra è colorata in coerenza con l'elemento che la genera (un CTA blu ha un'ombra blu translucida, non `rgba(0,0,0,.2)`).

### Shadow Vocabulary
- **Ambient card** (`box-shadow: 0 18px 40px rgba(22,34,46,.10)`): hover su card feature/step, suggerisce un leggero "lift" all'interazione.
- **CTA glow blue** (`box-shadow: 0 10px 30px rgba(29,114,194,.32)`): CTA primarie su sfondo chiaro.
- **CTA glow inverse** (`box-shadow: 0 10px 26px rgba(0,0,0,.18)`): CTA bianche su sfondo colorato (es. dentro la sezione CTA finale a sfondo gradiente blue→green).
- **Nav glass** (`box-shadow: 0 8px 30px rgba(22,34,46,.08)`): barra di navigazione flottante con backdrop-blur.

### Named Rules
**The Tinted Shadow Rule.** Nessuna ombra usa `rgba(0,0,0,...)` come colore dominante su elementi colorati: l'ombra eredita la tinta dell'elemento (blu, verde, ink), mai grigio neutro generico.

## 5. Components

### Buttons
- **Shape:** radius ampio (xl, 24px) per CTA primarie/secondarie su sfondo chiaro; radius medio (md, 12px) per azioni di nav/utility.
- **Primary:** sfondo `brand-blue`, testo bianco, padding `16px 28px`, ombra CTA glow blue; in gradiente blue→blue-dark nelle hero CTA.
- **Hover / Focus:** `scale(1.03)` su hover, `scale(0.98)` su active, transizione su transform; focus-visible con outline 2.5px `brand-blue` e offset 2px (definito globalmente, non per-componente).
- **Secondary / Ghost:** sfondo `ink` per azioni di nav secondarie; ghost con border `black/10` e bg `white/80` con backdrop-blur per CTA terziarie (es. "Scopri di più").

### Chips / Badge
- **Style:** pill (radius full), font mono, sfondo `mint` + testo `brand-green` per stato positivo/normale; bordo sottile `brand-green/20`.
- **State:** un punto pulsante (`animate-pulse`) accompagna i badge "live"/attivi.

### Cards / Containers
- **Corner Style:** radius xl (24px) costante su feature card, step card, container hero.
- **Background:** bianco puro su sfondo mist, mai grigio chiaro intermedio.
- **Shadow Strategy:** flat a riposo, ambient card shadow solo su hover (vedi Elevation).
- **Border:** `border-black/5` sottile come separazione, non come accento colorato.
- **Internal Padding:** 24-28px (lg), generoso per non affollare numeri clinici e copy.

### Navigation
- Barra flottante (non full-width fissa), pill arrotondata (rounded-2xl), bg bianco translucido con backdrop-blur-xl e ombra nav glass. Logo come quadrato gradient blue→green con radius 10px. CTA di nav in stile button-secondary (ink).

### Cervello digitale (Signature Component)
Il `NeuralBrainBackground` (rete neurale 3D procedurale, three.js) e il `BrainViewer3D` (mesh ricostruita da MRI con colorazione per zone anatomiche + overlay Grad-CAM opzionale) sono l'elemento visivo distintivo del brand. Vanno trattati come firma ricorrente — non un'icona intercambiabile — con rotazione sempre lenta/dampened, mai brusca, e sempre con un percorso `prefers-reduced-motion` gestito esplicitamente in JS (le media query CSS non bastano per three.js).

## 6. Do's and Don'ts

### Do:
- **Do** usare `mist` (#f6fafc) o bianco come sfondo body di default.
- **Do** riservare IBM Plex Mono a dati clinici, badge e label tecniche.
- **Do** tingere ogni ombra con il colore dell'elemento che la proietta (blue/green/ink), mai grigio neutro.
- **Do** mantenere radius ampi e coerenti (xl 24px su CTA/card) come firma visiva morbida ma precisa.
- **Do** gestire `prefers-reduced-motion` esplicitamente in JS per ogni animazione three.js/GSAP/Motion, non solo via CSS.
- **Do** comunicare i risultati clinici (gap, badge) con calma e precisione, mai con tono allarmistico anche quando il valore è fuori soglia.

### Don't:
- **Don't** usare uno sfondo crema/sand/parchment "per eleganza" — è il default AI del 2026, esplicitamente rifiutato da PRODUCT.md.
- **Don't** aggiungere eyebrow tutto-maiuscolo sopra ogni sezione, hero-metric template, gradient text, o card grid identiche — i tell da "AI slop" elencati come anti-riferimento.
- **Don't** introdurre dark mode (escluso esplicitamente in una sessione precedente).
- **Don't** dare al prodotto un'estetica freddo-ospedaliera/burocratica da modulistica sanitaria, nemmeno nella dashboard clinica.
- **Don't** usare `border-left`/`border-right` come accento colorato (side-stripe) su card o alert.
- **Don't** animare il cervello digitale o qualsiasi elemento 3D con movimenti bruschi o "bounce"/elastic easing — solo curve ease-out.
