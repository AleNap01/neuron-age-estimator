import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import * as THREE from "three";

/**
 * BrainViewer3D — mostra la mesh 3D del cervello (GLB) ricostruita dal
 * backend a partire dal volume MRI, ruotabile/zoomabile col mouse e con
 * uno slider di sezione che "scava" nel modello lungo un asse per
 * rivelare le strutture interne (clipping plane di three.js, lato client).
 *
 * La rotazione è volutamente lenta e ammortizzata (rotateSpeed/damping
 * bassi): è l'elemento centrale della pagina di analisi, deve sembrare
 * un oggetto "pesante" e preciso da ispezionare, non un giocattolo.
 */

const AXES = {
  x: new THREE.Vector3(1, 0, 0),
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
};

function BrainMesh({ glbUrl, sliceAxis, sliceValue }) {
  const gltf = useLoader(GLTFLoader, glbUrl);
  const { gl } = useThree();

  const clippingPlane = useMemo(() => new THREE.Plane(AXES[sliceAxis].clone(), 0), [sliceAxis]);

  const scene = useMemo(() => {
    const cloned = gltf.scene.clone(true);
    cloned.traverse((child) => {
      if (child.isMesh) {
        // Se la mesh ha colori per vertice (overlay Grad-CAM dal backend),
        // li lasciamo parlare: colore base bianco neutro + vertexColors.
        const hasVertexColors = !!child.geometry.attributes.color;
        child.material = new THREE.MeshPhysicalMaterial({
          color: hasVertexColors ? "#ffffff" : "#dcebf7",
          vertexColors: hasVertexColors,
          metalness: 0.04,
          roughness: 0.42,
          clearcoat: 0.35,
          clearcoatRoughness: 0.5,
          sheen: hasVertexColors ? 0 : 0.6,
          sheenColor: new THREE.Color("#1d72c2"),
          side: THREE.DoubleSide,
          clippingPlanes: [clippingPlane],
          clipShadows: true,
        });
      }
    });
    return cloned;
  }, [gltf, clippingPlane]);

  useEffect(() => {
    gl.localClippingEnabled = true;
  }, [gl]);

  useEffect(() => {
    // sliceValue va da -1 (taglio completo) a 1 (nessun taglio): il piano
    // si muove lungo l'asse scelto e mostra ciò che resta "davanti" ad esso.
    clippingPlane.constant = sliceValue;
  }, [clippingPlane, sliceValue]);

  return <primitive object={scene} />;
}

function FitCamera() {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(2.7, 1.5, 2.7);
    camera.lookAt(0, 0, 0);
  }, [camera]);
  return null;
}

export default function BrainViewer3D({ glbUrl }) {
  const [sliceAxis, setSliceAxis] = useState("z");
  const [sliceValue, setSliceValue] = useState(1.2);
  const controlsRef = useRef();

  if (!glbUrl) return null;

  return (
    <div
      style={{
        background: "linear-gradient(165deg, #0c1620 0%, #0a1420 55%, #0e1a24 100%)",
        borderRadius: 24,
        border: "1px solid #1c2e3c",
        overflow: "hidden",
        boxShadow: "0 30px 70px rgba(10,20,30,.35)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "18px 24px 0",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#5fd9a8", boxShadow: "0 0 0 4px rgba(95,217,168,.18)" }} />
          <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: ".5px", color: "#a9c2d4", fontFamily: "'IBM Plex Mono', monospace" }}>
            MODELLO 3D RICOSTRUITO
          </span>
        </div>
      </div>

      <div
        role="img"
        aria-label="Modello 3D interattivo della superficie cerebrale ricostruita dalla risonanza magnetica caricata, ruotabile e sezionabile con i controlli sotto al riquadro"
        style={{ height: "min(64vh, 620px)", minHeight: 420, position: "relative" }}
      >
        <Canvas
          gl={{ localClippingEnabled: true, antialias: true }}
          camera={{ fov: 42 }}
          dpr={[1, 2]}
        >
          <color attach="background" args={["#0a1420"]} />
          <fog attach="fog" args={["#0a1420", 6, 11]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[3, 4, 5]} intensity={1.3} color="#eaf4fb" />
          <directionalLight position={[-4, -1, -3]} intensity={0.5} color="#15976a" />
          <pointLight position={[0, 2.5, 0]} intensity={0.35} color="#7fc2ec" />
          <FitCamera />
          <BrainMesh glbUrl={glbUrl} sliceAxis={sliceAxis} sliceValue={sliceValue} />
          <OrbitControls
            ref={controlsRef}
            enableDamping
            dampingFactor={0.045}
            rotateSpeed={0.35}
            zoomSpeed={0.5}
            panSpeed={0.4}
            minDistance={1.8}
            maxDistance={5.5}
          />
        </Canvas>
      </div>

      <div style={{ padding: "18px 24px 22px", borderTop: "1px solid #16242f" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 12 }}>
          <span id="slice-axis-label" style={{ fontSize: 12, fontWeight: 600, color: "#9db4c4", fontFamily: "'IBM Plex Mono', monospace", letterSpacing: ".4px" }}>
            PIANO DI SEZIONE
          </span>
          <div role="group" aria-labelledby="slice-axis-label" style={{ display: "flex", gap: 6 }}>
            {["x", "y", "z"].map((axis) => (
              <button
                key={axis}
                onClick={() => setSliceAxis(axis)}
                aria-pressed={sliceAxis === axis}
                aria-label={`Sezione lungo l'asse ${axis.toUpperCase()}`}
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  padding: "5px 12px",
                  borderRadius: 8,
                  border: sliceAxis === axis ? "1px solid #7fc2ec" : "1px solid #233646",
                  cursor: "pointer",
                  color: sliceAxis === axis ? "#06141d" : "#cfe3f2",
                  background: sliceAxis === axis
                    ? "linear-gradient(135deg,#7fc2ec,#5fd9a8)"
                    : "transparent",
                  transition: "all .18s ease",
                }}
              >
                {axis.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <input
          type="range"
          min={-1.2}
          max={1.2}
          step={0.01}
          value={sliceValue}
          aria-label="Posizione del piano di sezione: da cervello interno a superficie completa"
          onChange={(e) => setSliceValue(parseFloat(e.target.value))}
          style={{ width: "100%", accentColor: "#7fc2ec" }}
        />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, color: "#6d869a", marginTop: 6 }}>
          <span>cervello interno</span>
          <span>superficie completa</span>
        </div>
      </div>
    </div>
  );
}
