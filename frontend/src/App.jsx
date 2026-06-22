import { lazy, Suspense, useState } from "react";
import NeuroAgeLanding from "./NeuroAgeLanding.jsx";

// Caricata solo quando l'utente apre la dashboard: contiene three.js/fiber
// (BrainViewer3D) e tutta la logica di analisi, non serve nel bundle
// iniziale della landing.
const Dashboard = lazy(() => import("./Dashboard.jsx"));

function DashboardFallback() {
  return (
    <div
      style={{
        minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: "'IBM Plex Sans', system-ui, sans-serif", color: "#5b6b79", fontSize: 14,
        background: "#f6fafc",
      }}
      role="status"
    >
      Caricamento dashboard…
    </div>
  );
}

/**
 * App — gestisce la navigazione tra landing page e dashboard.
 *
 * Per ora usa un semplice useState invece di un router: con solo due
 * "pagine" è la soluzione più semplice. Se il progetto crescerà (più
 * route, URL condivisibili, back/forward del browser), vale la pena
 * introdurre react-router-dom.
 */
export default function App() {
  const [view, setView] = useState("landing"); // "landing" | "dashboard"

  if (view === "dashboard") {
    return (
      <Suspense fallback={<DashboardFallback />}>
        <Dashboard onBackClick={() => setView("landing")} />
      </Suspense>
    );
  }

  return <NeuroAgeLanding onUploadClick={() => setView("dashboard")} />;
}
