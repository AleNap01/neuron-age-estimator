# Product

## Register

brand

## Users

Persone con accesso a una risonanza magnetica cerebrale (paziente, caregiver, o operatore clinico che li accompagna) che vogliono stimare l'età biologica del proprio cervello rispetto all'età anagrafica (brain-age gap) e capirne il significato. Contesto d'uso: spesso la prima volta che vedono uno strumento di questo tipo, possibile ansia legata a un risultato clinico-adiacente. La landing page deve guidarli con fiducia verso il caricamento dell'MRI e la lettura del referto; la dashboard (superficie secondaria, registro product) deve poi sostenere il workflow di analisi (upload, ricostruzione 3D, lettura del referto) senza intimidire.

## Product Purpose

NeuroAge·MRI stima l'età biologica del cervello da una risonanza magnetica, mostra lo scostamento (gap) rispetto all'età anagrafica con intervallo di confidenza, e genera un referto PDF clinico professionale. Il cuore tecnico è la ricostruzione 3D interattiva del cervello (marching cubes + three.js) con colorazione per zone anatomiche illustrative e overlay opzionale Grad-CAM. Successo = l'utente carica un MRI, esplora il modello 3D, capisce il proprio risultato (gap, confidenza, interpretazione) e si fida abbastanza del referto da condividerlo o conservarlo.

## Brand Personality

Clinico, preciso, rassicurante. Il tono è quello di uno strumento scientifico serio — non un giocattolo tech, non un modulo ospedaliero asettico. La fiducia si costruisce mostrando precisione (numeri, intervalli di confidenza, metodologia trasparente) e accompagnando l'utente con calma, non con urgenza o allarmismo. Il "cervello digitale" 3D è il elemento distintivo: deve comunicare competenza tecnica senza scivolare nell'estetica sci-fi/gadget.

## Anti-references

- Stile SaaS generico da "AI slop": niente sfondo crema/sand di default, niente eyebrow tutto-maiuscolo sopra ogni sezione, niente hero-metric template, niente gradient text, niente card grid identiche.
- Niente estetica freddo-ospedaliera/burocratica da modulistica sanitaria: il referto clinico deve essere professionale ma il resto del prodotto (landing, dashboard) non deve sembrare un form assicurativo.
- Niente dark mode (esplicitamente escluso dall'utente in una sessione precedente).
- Niente urgenza/allarmismo nella comunicazione del risultato clinico (badge e interpretazione testuale calibrati sulla gravità del gap, ma sempre con tono rassicurante, non drammatico).

## Design Principles

- **Precisione visibile**: ogni numero clinico (età stimata, gap, intervallo di confidenza) deve essere leggibile a colpo d'occhio e mai ambiguo — la fiducia nasce dalla chiarezza dei dati, non dalla decorazione.
- **Il cervello digitale è la firma visiva**, non un'icona generica: va trattato come l'elemento di brand ricorrente (hero, loading states, dashboard) coerente in stile across le superfici.
- **Calma sopra urgenza**: motion, colore e copy comunicano competenza e controllo, mai ansia — anche quando il risultato clinico è fuori soglia.
- **Trasparenza metodologica come elemento di fiducia**: la suddivisione anatomica è "illustrativa, non un atlante validato" — il prodotto è onesto sui propri limiti invece di vendersi come più di quello che è, e questo va riflesso anche nel tono della UI (microcopy, tooltip, referto).
- **Accessibilità non negoziabile**: WCAG 2.1 AA, motion sempre con alternativa reduced-motion, ruoli ARIA su tutti i controlli custom (slider, toggle, canvas 3D) — già parzialmente implementato, va mantenuto come standard per ogni nuova feature.

## Accessibility & Inclusion

WCAG 2.1 AA. Contrasto testo ≥4.5:1 (≥3:1 per testo grande), focus ring visibile e consistente, aria-label/aria-pressed/role="group" su tutti i controlli custom, role="img" con descrizione testuale sul canvas 3D. Ogni animazione (incluse quelle three.js/GSAP/Motion, non gestibili solo via media query CSS) deve rispettare `prefers-reduced-motion` con gestione esplicita in JS.
