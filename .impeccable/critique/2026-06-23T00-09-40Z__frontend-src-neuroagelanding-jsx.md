---
target: frontend/src/NeuroAgeLanding.jsx
total_score: 30
p0_count: 0
p1_count: 2
timestamp: 2026-06-23T00-09-40Z
slug: frontend-src-neuroagelanding-jsx
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | CTA non da feedback hover/click oltre a scale, accettabile per pagina statica |
| 2 | Match System / Real World | 4 | Linguaggio italiano chiaro, terminologia clinica accessibile |
| 3 | User Control and Freedom | 2 | Nav (Tecnologia, Come funziona) sparisce sotto sm: senza hamburger: su mobile non c'è modo di navigare alle sezioni |
| 4 | Consistency and Standards | 3 | Eyebrow numerato nel bento (01 - ML CLASSICO) non coerente con eyebrow non numerato delle sezioni (Tecnologia, Come funziona) |
| 5 | Error Prevention | 3 | n/a, pagina statica senza form |
| 6 | Recognition Rather Than Recall | 4 | Nav testuale, CTA ripetuta e riconoscibile in 3 punti |
| 7 | Flexibility and Efficiency | 3 | n/a per una landing |
| 8 | Aesthetic and Minimalist Design | 2 | Eyebrow + numerazione + gradient text + 4 card identiche appesantiscono una pagina altrimenti pulita |
| 9 | Error Recovery | 3 | n/a |
| 10 | Help and Documentation | 3 | Link "Scopri come funziona" come help contestuale |
| **Total** | | **30/40** | **Good — fondamenta solide, ma violazioni dirette delle proprie linee guida anti-slop** |

## Anti-Patterns Verdict

**Sembra fatto dall'AI?** Sì, in punti specifici — non per il concept (il cervello digitale è distintivo), ma per scelte di scaffolding che DESIGN.md/PRODUCT.md hanno appena vietato esplicitamente:

1. **Gradient text** (riga 196): bg-clip-text + bg-gradient-to-r from-brand-blue to-brand-green sulla headline hero ("davvero il tuo cervello?"). Confermato dal detector automatico (severity: warning).
2. **Eyebrow tutto-maiuscolo sopra ogni sezione** (righe 263, 287): "Tecnologia" e "Come funziona" sono entrambi introdotti da un eyebrow mono uppercase tracked — esattamente il tell bandito in DESIGN.md.
3. **Numbered section markers come scaffolding di default** (righe 25, 34, 50, 64, 78, 84, 90): ogni feature card e ogni step ha un "01 ·", "02 ·" ecc. La sequenza negli STEPS è legittima (sono davvero 3 passi ordinati), ma applicarla anche alle 4 FEATURES (categorie parallele, non sequenza) è scaffolding riflesso.
4. **Card grid identiche** (righe 267-281): 4 card stesso layout (icona + eyebrow + titolo + testo), stessa dimensione, stesso hover-glow — il "bento" è in realtà una card grid identica con un nome diverso.

**Scan deterministico**: 1 finding (gradient-text, riga 196, severity warning). Lo scanner non rileva eyebrow/numbered-markers/card-grid perché sono pattern strutturali, non singole classi CSS.

**Browser overlay**: non eseguito in questa passata (nessun dev server attivo in sessione); i risultati sopra vengono da lettura diretta del codice sorgente.

## Overall Impression

Il concept (cervello digitale 3D come firma visiva, palette blue/green su base quasi-bianca, tipografia IBM Plex) è solido e coerente con PRODUCT.md/DESIGN.md appena scritti. Il problema non è l'identità ma lo scaffolding preso in prestito da landing page SaaS generiche — eyebrow, numerazione, gradient text, card grid — che DESIGN.md ha appena vietato per nome.

## What's Working

- Microcopy onesta: il footer e gli STEPS numerati (legittimi, vero flusso a 3 passi) rispettano il principio di trasparenza metodologica di PRODUCT.md.
- Le ombre sono già tinte correttamente (es. CTA blu con ombra blu) — coerente con la "Tinted Shadow Rule" di DESIGN.md.
- Reduced-motion gestito globalmente, e il cervello digitale vive dietro al testo come elemento atmosferico, non gadget in primo piano.

## Priority Issues

- **[P1] Quattro anti-pattern AI-slop espliciti**: gradient text (r.196), eyebrow uppercase su 2 sezioni (r.263, 287), numerazione applicata anche a categorie non sequenziali (FEATURES, r.25-74), card grid identica (r.267-281).
  - Why it matters: DESIGN.md appena scritto vieta esplicitamente questi 4 pattern per nome; il codice attuale li contiene tutti.
  - Fix: testo hero in colore solido; eyebrow sostituito da trattamento tipografico diverso; numerazione rimossa dalle FEATURES (resta sugli STEPS); bento trasformato in layout asimmetrico o con varianti.
  - Suggested command: /impeccable quieter + /impeccable typeset + /impeccable layout

- **[P1] Nav mobile sparisce senza alternativa**: nav con link nascosta sotto sm: senza hamburger o menu mobile.
  - Why it matters: su mobile gli utenti perdono ogni modo di navigare alle sezioni, solo la CTA primaria resta accessibile.
  - Fix: aggiungere un menu mobile minimale, o documentare la scelta come deliberata.
  - Suggested command: /impeccable adapt

- **[P2] Hero stat row ricorda il "hero-metric template"**: 3 numeri grandi + caption piccola subito sotto la CTA.
  - Why it matters: pattern SaaS molto riconoscibile, anche se i numeri sono reali e utili.
  - Fix: integrare i numeri in un layout meno "card di metriche".
  - Suggested command: /impeccable delight o /impeccable layout

- **[P3] Footer disclaimer a basso contrasto**: text-faint su sfondo bianco per un disclaimer clinico importante.
  - Why it matters: un'informazione rilevante non dovrebbe avere il contrasto più basso della pagina.
  - Fix: passare a text-muted o text-ink per il disclaimer.
  - Suggested command: /impeccable audit

## Persona Red Flags

**Jordan (First-Timer)**: Vede "AI - Neuroimaging - Cervello digitale" come prima cosa — termine "Neuroimaging" non spiegato. Gli eyebrow "01 - ML CLASSICO" / "02 - CNN 3D" usano acronimi tecnici senza spiegazione in linguaggio semplice.

**Casey (Mobile)**: Perde la nav (vedi P1 sopra). La CTA hero richiede comunque scroll oltre la viewport iniziale su schermi piccoli data l'altezza di pt-40 pb-28.

## Minor Observations

- "2.3k" (volumi di addestramento) non ha unità/contesto immediato, solo nel caption piccolo sotto.
- selection:bg-brand-blue/20 sul body è un dettaglio di cura piacevole, da mantenere.
- Il pattern a griglia di sfondo (opacity 0.035) è sottile ma rischia moiré su schermi alta densità; verificarlo su device reale.

## Questions to Consider

- Le FEATURES hanno davvero bisogno di un numero se non sono una sequenza? Cosa succederebbe a rimuoverlo del tutto?
- Il gradient text sulla headline è l'unico modo per far percepire la coppia blue/green come "duo" del brand, o si può ottenere lo stesso effetto con un trattamento tipografico più sobrio?
- Se la nav scompare su mobile, è una scelta deliberata o un'omissione?
