import { useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

/**
 * NeuralBrainBackground — "cervello digitale": una nuvola di nodi disposta
 * a forma di cervello (ellissoide con solco interemisferico), collegata
 * da segmenti come una rete neurale, in lenta rotazione con una leggera
 * parallasse al movimento del mouse. Pensato per stare DIETRO al testo
 * della hero, non come oggetto da manipolare (sostituisce il precedente
 * Brain3D "ad oggetto").
 */

const BRAND_BLUE = new THREE.Color("#1d72c2");
const BRAND_GREEN = new THREE.Color("#15976a");
const BRAND_CYAN = new THREE.Color("#7fc2ec");

function buildNetwork(count = 110, neighbors = 3) {
  const radii = { x: 2.5, y: 1.55, z: 1.85 };
  const points = [];

  // Campionamento per rigetto dentro un ellissoide, con un solco lungo x=0
  // per evocare le due emisfere — niente mesh, solo nodi.
  while (points.length < count) {
    const x = (Math.random() * 2 - 1) * radii.x;
    const y = (Math.random() * 2 - 1) * radii.y;
    const z = (Math.random() * 2 - 1) * radii.z;
    const norm = (x * x) / (radii.x * radii.x) + (y * y) / (radii.y * radii.y) + (z * z) / (radii.z * radii.z);
    if (norm > 1) continue;
    if (Math.abs(x) < 0.18 && Math.random() > 0.15) continue; // solco interemisferico
    points.push(new THREE.Vector3(x, y * 0.95, z));
  }

  // Edge verso i k vicini più prossimi, deduplicati.
  const edgeSet = new Set();
  const edges = [];
  points.forEach((p, i) => {
    const dists = points
      .map((q, j) => (j === i ? Infinity : p.distanceTo(q)))
      .map((d, j) => [d, j])
      .sort((a, b) => a[0] - b[0])
      .slice(0, neighbors);

    dists.forEach(([, j]) => {
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push([i, j]);
      }
    });
  });

  return { points, edges, radii };
}

function colorForPoint(p, radii) {
  const t = THREE.MathUtils.clamp((p.y / radii.y + 1) / 2, 0, 1);
  return BRAND_BLUE.clone().lerp(BRAND_GREEN, t);
}

function makeGlowSprite() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.35, "rgba(255,255,255,0.7)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function NeuralNetwork() {
  const groupRef = useRef();
  const { pointer } = useThree();
  const targetRot = useRef({ x: 0, y: 0 });
  const reducedMotion = useMemo(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    []
  );

  const { lineGeometry, pointsGeometry, pointsMaterial } = useMemo(() => {
    const { points, edges, radii } = buildNetwork();

    const linePositions = new Float32Array(edges.length * 6);
    const lineColors = new Float32Array(edges.length * 6);
    edges.forEach(([a, b], i) => {
      const pa = points[a];
      const pb = points[b];
      linePositions.set([pa.x, pa.y, pa.z, pb.x, pb.y, pb.z], i * 6);
      const ca = colorForPoint(pa, radii);
      const cb = colorForPoint(pb, radii);
      lineColors.set([ca.r, ca.g, ca.b, cb.r, cb.g, cb.b], i * 6);
    });
    const lGeo = new THREE.BufferGeometry();
    lGeo.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
    lGeo.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));

    const pointPositions = new Float32Array(points.length * 3);
    const pointColors = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      pointPositions.set([p.x, p.y, p.z], i * 3);
      const c = colorForPoint(p, radii).lerp(BRAND_CYAN, 0.25);
      pointColors.set([c.r, c.g, c.b], i * 3);
    });
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(pointPositions, 3));
    pGeo.setAttribute("color", new THREE.BufferAttribute(pointColors, 3));

    const pMat = new THREE.PointsMaterial({
      size: 0.085,
      map: makeGlowSprite(),
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    return { lineGeometry: lGeo, pointsGeometry: pGeo, pointsMaterial: pMat };
  }, []);

  useFrame((state, delta) => {
    if (!groupRef.current) return;

    if (reducedMotion) {
      // Niente rotazione automatica né parallasse: la rete resta ferma,
      // solo un pulse minimo dei nodi per non essere un'immagine statica morta.
      pointsMaterial.size = 0.085;
      return;
    }

    groupRef.current.rotation.y += delta * 0.05;

    targetRot.current.x = pointer.y * 0.12;
    targetRot.current.y += (pointer.x * 0.25 - targetRot.current.y) * 0.04;
    groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, targetRot.current.x, 0.04);

    const pulse = 0.085 + Math.sin(state.clock.elapsedTime * 1.4) * 0.012;
    pointsMaterial.size = pulse;
  });

  return (
    <group ref={groupRef} rotation={[0.1, 0.5, 0]}>
      <lineSegments geometry={lineGeometry}>
        <lineBasicMaterial vertexColors transparent opacity={0.4} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
      <points geometry={pointsGeometry} material={pointsMaterial} />
    </group>
  );
}

export default function NeuralBrainBackground({ className }) {
  return (
    <div className={className} aria-hidden="true">
      <Canvas
        camera={{ position: [0, 0, 6.5], fov: 42 }}
        dpr={[1, 1.6]}
        gl={{ antialias: true, alpha: true }}
        style={{ pointerEvents: "none" }}
      >
        <NeuralNetwork />
      </Canvas>
    </div>
  );
}
