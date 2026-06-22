/**
 * TrendChart — andamento longitudinale dell'età cerebrale stimata nelle
 * analisi passate (storico in localStorage). Mini line-chart SVG scritto a
 * mano: per un singolo grafico semplice non serve una libreria di charting
 * dedicata.
 */
export default function TrendChart({ entries }) {
  if (entries.length < 2) return null;

  // Storico più recente prima: lo ribaltiamo per leggere il tempo da sinistra a destra.
  const chrono = [...entries].reverse();
  const ages = chrono.map((e) => e.predicted_age);
  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);
  const range = Math.max(maxAge - minAge, 1);

  const width = 560;
  const height = 140;
  const padX = 28;
  const padY = 20;

  const x = (i) => padX + (i / (chrono.length - 1)) * (width - padX * 2);
  const y = (age) => height - padY - ((age - minAge) / range) * (height - padY * 2);

  const points = chrono.map((e, i) => [x(i), y(e.predicted_age)]);
  const linePath = points.map(([px, py], i) => `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${points[points.length - 1][0].toFixed(1)},${height - padY} L${points[0][0].toFixed(1)},${height - padY} Z`;

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height={height}
        role="img"
        aria-label={`Andamento dell'età cerebrale stimata su ${chrono.length} analisi, da ${chrono[0].predicted_age} a ${chrono[chrono.length - 1].predicted_age} anni`}
      >
        <defs>
          <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1d72c2" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#1d72c2" stopOpacity="0" />
          </linearGradient>
        </defs>

        <path d={areaPath} fill="url(#trendFill)" />
        <path d={linePath} fill="none" stroke="#1d72c2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

        {points.map(([px, py], i) => (
          <circle key={i} cx={px} cy={py} r="3.5" fill="#fff" stroke="#1d72c2" strokeWidth="2" />
        ))}
      </svg>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#8c99a6", marginTop: 2 }}>
        <span>{new Date(chrono[0].date).toLocaleDateString("it-IT")}</span>
        <span>{new Date(chrono[chrono.length - 1].date).toLocaleDateString("it-IT")}</span>
      </div>
    </div>
  );
}
