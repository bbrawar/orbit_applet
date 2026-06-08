/**
 * OrbitalMechanicsApplet
 * ----------------------
 * Interactive 3D orbital mechanics visualization component.
 * Uses Three.js for 3D rendering with full Keplerian orbital mechanics.
 *
 * Embedding:
 *   import OrbitalMechanicsApplet from './orbital-mechanics-applet';
 *   <OrbitalMechanicsApplet defaultFrame="ECEF" theme="dark" height="700px" />
 *
 * Or as iframe: point to the standalone HTML version.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";

// ─────────────────────────────────────────────
// CONSTANTS & MATH HELPERS
// ─────────────────────────────────────────────
const MU = 398600.4418; // km³/s²  Earth gravitational parameter
const RE = 6371.0;       // km      Earth radius
const OMEGA_E = 7.2921150e-5; // rad/s  Earth rotation rate
const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

/** Clamp value between min and max */
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson */
function solveKepler(M, e, tol = 1e-10) {
  let E = M;
  for (let i = 0; i < 50; i++) {
    const dE = (M - E + e * Math.sin(E)) / (1 - e * Math.cos(E));
    E += dE;
    if (Math.abs(dE) < tol) break;
  }
  return E;
}

/** True anomaly from eccentric anomaly */
function eccentricToTrue(E, e) {
  return 2 * Math.atan2(
    Math.sqrt(1 + e) * Math.sin(E / 2),
    Math.sqrt(1 - e) * Math.cos(E / 2)
  );
}

/** True anomaly → mean anomaly */
function trueToMean(nu, e) {
  const E = 2 * Math.atan2(Math.sqrt(1 - e) * Math.sin(nu / 2), Math.sqrt(1 + e) * Math.cos(nu / 2));
  return E - e * Math.sin(E);
}

/**
 * Keplerian elements → ECI position & velocity (km, km/s)
 * a: semi-major axis (km), e: eccentricity, i: inclination (rad)
 * raan: RAAN (rad), omega: arg of perigee (rad), nu: true anomaly (rad)
 */
function elementsToECI(a, e, i, raan, omega, nu) {
  const p = a * (1 - e * e);
  const r = p / (1 + e * Math.cos(nu));
  // Perifocal coords
  const xp = r * Math.cos(nu);
  const yp = r * Math.sin(nu);
  const vxp = -Math.sqrt(MU / p) * Math.sin(nu);
  const vyp =  Math.sqrt(MU / p) * (e + Math.cos(nu));

  // Rotation matrices: R3(-raan) * R1(-i) * R3(-omega)
  const cO = Math.cos(raan), sO = Math.sin(raan);
  const co = Math.cos(omega), so = Math.sin(omega);
  const ci = Math.cos(i),    si = Math.sin(i);

  // Combined rotation (standard aerospace formulation)
  const l1 =  cO * co - sO * so * ci;
  const l2 = -cO * so - sO * co * ci;
  const m1 =  sO * co + cO * so * ci;
  const m2 = -sO * so + cO * co * ci;
  const n1 =  so * si;
  const n2 =  co * si;

  const pos = new THREE.Vector3(
    l1 * xp + l2 * yp,
    m1 * xp + m2 * yp,
    n1 * xp + n2 * yp
  );
  const vel = new THREE.Vector3(
    l1 * vxp + l2 * vyp,
    m1 * vxp + m2 * vyp,
    n1 * vxp + n2 * vyp
  );
  return { pos, vel, r };
}

/** ECI → ECEF (rotate by Earth sidereal angle θ_GMST ≈ omega_E * t) */
function eciToECEF(posECI, theta) {
  const c = Math.cos(-theta), s = Math.sin(-theta);
  return new THREE.Vector3(
    c * posECI.x - s * posECI.y,
    s * posECI.x + c * posECI.y,
    posECI.z
  );
}

/** ECEF position → geodetic lat/lon/alt */
function ecefToGeodetic(pos) {
  const lon = Math.atan2(pos.y, pos.x) * RAD;
  const p = Math.sqrt(pos.x * pos.x + pos.y * pos.y);
  const lat = Math.atan2(pos.z, p) * RAD;
  const alt = pos.length() - RE;
  return { lat, lon, alt };
}

/** Orbital period (seconds) */
const orbitalPeriod = (a) => 2 * Math.PI * Math.sqrt(a * a * a / MU);

/** Generate full orbit path points in ECI (N points) */
function generateOrbitPath(a, e, i, raan, omega, N = 256) {
  const pts = [];
  for (let k = 0; k <= N; k++) {
    const nu = (2 * Math.PI * k) / N;
    const { pos } = elementsToECI(a, e, i, raan, omega, nu);
    pts.push(pos.clone());
  }
  return pts;
}

// ─────────────────────────────────────────────
// PRESET ORBITS
// ─────────────────────────────────────────────
const PRESETS = {
  ISS:    { a: 6778,   e: 0.0007, i: 51.6, raan: 0,   omega: 0,   nu: 0,   label: "ISS" },
  GEO:    { a: 42164,  e: 0.0001, i: 0.07, raan: 0,   omega: 0,   nu: 0,   label: "GEO" },
  Molniya:{ a: 26560,  e: 0.74,   i: 63.4, raan: 0,   omega: 270, nu: 0,   label: "Molniya" },
  Polar:  { a: 7371,   e: 0.001,  i: 90,   raan: 0,   omega: 0,   nu: 0,   label: "Polar" },
  GPS:    { a: 26560,  e: 0.01,   i: 55,   raan: 0,   omega: 0,   nu: 0,   label: "GPS" },
  SSO:    { a: 7178,   e: 0.001,  i: 98.2, raan: 0,   omega: 0,   nu: 0,   label: "SSO" },
  HEO:    { a: 30000,  e: 0.85,   i: 28.5, raan: 0,   omega: 270, nu: 0,   label: "HEO" },
};
const DEFAULT_ORBIT = PRESETS.ISS;

// ─────────────────────────────────────────────
// THREE.JS SCALE (km → scene units, 1 unit = 1000 km)
// ─────────────────────────────────────────────
const SCENE_SCALE = 1 / 1000;
const toScene = (km) => km * SCENE_SCALE;

// ─────────────────────────────────────────────
// COLOR PALETTE (dark aerospace theme)
// ─────────────────────────────────────────────
const C = {
  bg:        "#04080f",
  bgPanel:   "#080d18",
  border:    "#1a2740",
  accent:    "#00d4ff",
  accentDim: "#0077aa",
  gold:      "#f5c842",
  green:     "#00ff88",
  red:       "#ff4466",
  white:     "#e8f4ff",
  dim:       "#4a6080",
  gridLine:  "#0d1f35",
  orbitLine: "#00d4ff",
  sat:       "#f5c842",
  velVec:    "#00ff88",
  hVec:      "#ff6633",
  nodeVec:   "#cc44ff",
  perigee:   "#ff4466",
  apogee:    "#44aaff",
  equator:   "#1e4080",
  prime:     "#1e4080",
};

// ─────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────
export default function OrbitalMechanicsApplet({
  defaultFrame = "ECI",
  theme = "dark",
  height = "700px",
}) {
  const mountRef = useRef(null);
  const threeRef = useRef({}); // stores all Three.js objects

  // Orbital elements state
  const [elems, setElems] = useState({ ...DEFAULT_ORBIT });
  const elemsRef = useRef({ ...DEFAULT_ORBIT });

  // Simulation state
  const [simTime, setSimTime] = useState(0); // seconds since epoch
  const [playing, setPlaying] = useState(false);
  const [timeScale, setTimeScale] = useState(200);
  const simRef = useRef({ time: 0, playing: false, timeScale: 200 });

  // Frame selection
  const [frame, setFrame] = useState(defaultFrame);
  const frameRef = useRef(defaultFrame);

  // Display toggles
  const [toggles, setToggles] = useState({
    orbitPath: true, satellite: true, velVec: true, hVec: true,
    orbitalPlane: true, nodeVec: true, perigeeMarker: true,
    apogeeMarker: true, equator: true, primeMeridian: true,
    axes: true, vernalEquinox: true, earthRotAxis: true,
    wireframe: false, trail: true,
  });
  const togglesRef = useRef({ ...toggles });

  // Telemetry
  const [telem, setTelem] = useState({
    pos: [0, 0, 0], vel: [0, 0, 0],
    lat: 0, lon: 0, alt: 0,
    period: 0, r: 0,
  });

  // Trail
  const trailRef = useRef([]);

  // ─── THREE.JS SETUP ───
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const W = el.clientWidth, H = el.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    renderer.setClearColor(0x04080f, 1);
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, W / H, 0.001, 1000);
    camera.position.set(0, toScene(RE * 3.5), toScene(RE * 3.5));
    camera.lookAt(0, 0, 0);

    // Ambient + directional light
    scene.add(new THREE.AmbientLight(0x223355, 2.5));
    const sun = new THREE.DirectionalLight(0xffffff, 3);
    sun.position.set(5, 3, 5);
    scene.add(sun);

    // ── Earth Sphere ──
    const earthGeo = new THREE.SphereGeometry(toScene(RE), 64, 64);
    const earthMat = new THREE.MeshPhongMaterial({
      color: 0x1a3f6b,
      emissive: 0x0a1a2f,
      shininess: 60,
      wireframe: false,
    });
    const earthMesh = new THREE.Mesh(earthGeo, earthMat);
    scene.add(earthMesh);

    // Subtle grid on earth
    const gridMat = new THREE.MeshBasicMaterial({ color: 0x0d2545, wireframe: true, transparent: true, opacity: 0.18 });
    const gridGeo = new THREE.SphereGeometry(toScene(RE * 1.001), 24, 24);
    const gridMesh = new THREE.Mesh(gridGeo, gridMat);
    scene.add(gridMesh);

    // ── Equator ring ──
    const equatorCurve = new THREE.EllipseCurve(0, 0, toScene(RE), toScene(RE), 0, 2 * Math.PI);
    const equatorPts = equatorCurve.getPoints(128);
    const equatorGeo = new THREE.BufferGeometry().setFromPoints(equatorPts.map(p => new THREE.Vector3(p.x, 0, -p.y)));
    const equatorLine = new THREE.Line(equatorGeo, new THREE.LineBasicMaterial({ color: 0x1e5090, linewidth: 1.5, transparent: true, opacity: 0.7 }));
    scene.add(equatorLine);

    // ── Prime meridian ──
    const pmCurve = new THREE.EllipseCurve(0, 0, toScene(RE), toScene(RE), 0, 2 * Math.PI);
    const pmPts = pmCurve.getPoints(128).map(p => new THREE.Vector3(p.x, -p.y, 0));
    const pmGeo = new THREE.BufferGeometry().setFromPoints(pmPts);
    const pmLine = new THREE.Line(pmGeo, new THREE.LineBasicMaterial({ color: 0x1e5090, linewidth: 1.5, transparent: true, opacity: 0.7 }));
    scene.add(pmLine);

    // ── Axes (ECI frame) ──
    const axisLen = toScene(RE * 2.5);
    const axisGroup = new THREE.Group();
    const makeAxis = (dir, color) => {
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), dir.clone().multiplyScalar(axisLen)]);
      return new THREE.Line(g, new THREE.LineBasicMaterial({ color, linewidth: 2 }));
    };
    axisGroup.add(makeAxis(new THREE.Vector3(1, 0, 0), 0xff3344));
    axisGroup.add(makeAxis(new THREE.Vector3(0, 1, 0), 0x33ff55));
    axisGroup.add(makeAxis(new THREE.Vector3(0, 0, 1), 0x3399ff));
    scene.add(axisGroup);

    // ── Vernal Equinox vector (ECI X, celestial) ──
    const veLen = toScene(RE * 3.2);
    const veGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(veLen, 0, 0)]);
    const veLine = new THREE.Line(veGeo, new THREE.LineBasicMaterial({ color: 0xffee00, linewidth: 3 }));
    scene.add(veLine);

    // ── Earth rotation axis ──
    const rotAxisGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, -axisLen, 0),
      new THREE.Vector3(0, axisLen, 0)
    ]);
    const rotAxisLine = new THREE.Line(rotAxisGeo, new THREE.LineBasicMaterial({ color: 0x88aaff, linewidth: 2, transparent: true, opacity: 0.6 }));
    scene.add(rotAxisLine);

    // ── Orbit path (dynamic, rebuilt each frame) ──
    const orbitGroup = new THREE.Group();
    scene.add(orbitGroup);

    // Orbital plane (transparent disc)
    const planeGeo = new THREE.CircleGeometry(toScene(45000), 64);
    const planeMat = new THREE.MeshBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.05, side: THREE.DoubleSide });
    const planeMesh = new THREE.Mesh(planeGeo, planeMat);
    orbitGroup.add(planeMesh);

    // Orbit line
    const orbitLineGeo = new THREE.BufferGeometry();
    const orbitLineMesh = new THREE.Line(orbitLineGeo, new THREE.LineBasicMaterial({ color: 0x00d4ff, linewidth: 2 }));
    orbitGroup.add(orbitLineMesh);

    // ── Satellite ──
    const satGeo = new THREE.SphereGeometry(toScene(RE * 0.1), 16, 16);
    const satMat = new THREE.MeshPhongMaterial({ color: 0xf5c842, emissive: 0x443300, shininess: 100 });
    const satMesh = new THREE.Mesh(satGeo, satMat);
    scene.add(satMesh);

    // ── Trail ──
    const maxTrailPts = 500;
    const trailPositions = new Float32Array(maxTrailPts * 3);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
    trailGeo.setDrawRange(0, 0);
    const trailLine = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({
      color: 0x00d4ff, transparent: true, opacity: 0.5, linewidth: 1.5
    }));
    scene.add(trailLine);

    // ── Velocity vector arrow ──
    const velArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), toScene(2000), 0x00ff88, toScene(400), toScene(200));
    scene.add(velArrow);

    // ── Angular momentum vector arrow ──
    const hArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), toScene(RE * 2), 0xff6633, toScene(400), toScene(200));
    scene.add(hArrow);

    // ── Node vector ──
    const nodeArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), toScene(RE * 2.2), 0xcc44ff, toScene(300), toScene(150));
    scene.add(nodeArrow);

    // ── Perigee & Apogee markers ──
    const mkGeo = new THREE.SphereGeometry(toScene(RE * 0.08), 12, 12);
    const perigeeMarker = new THREE.Mesh(mkGeo, new THREE.MeshPhongMaterial({ color: 0xff4466, emissive: 0x330011 }));
    const apogeeMarker  = new THREE.Mesh(mkGeo.clone(), new THREE.MeshPhongMaterial({ color: 0x44aaff, emissive: 0x001133 }));
    scene.add(perigeeMarker);
    scene.add(apogeeMarker);

    // ── Store all objects ──
    threeRef.current = {
      renderer, scene, camera,
      earthMesh, gridMesh, equatorLine, pmLine,
      axisGroup, veLine, rotAxisLine,
      orbitGroup, orbitLineMesh, planeMesh,
      satMesh, trailLine, trailPositions,
      velArrow, hArrow, nodeArrow,
      perigeeMarker, apogeeMarker,
    };

    // ── OrbitControls (manual, no external dep) ──
    let isDragging = false, prevMouse = { x: 0, y: 0 };
    let spherical = { theta: 0.6, phi: 0.9, radius: camera.position.length() };

    const updateCamera = () => {
      camera.position.set(
        spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta),
        spherical.radius * Math.cos(spherical.phi),
        spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta)
      );
      camera.lookAt(0, 0, 0);
    };
    updateCamera();

    renderer.domElement.addEventListener("mousedown", e => { isDragging = true; prevMouse = { x: e.clientX, y: e.clientY }; });
    renderer.domElement.addEventListener("mouseup", () => { isDragging = false; });
    renderer.domElement.addEventListener("mousemove", e => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      spherical.theta -= dx * 0.008;
      spherical.phi = clamp(spherical.phi - dy * 0.008, 0.05, Math.PI - 0.05);
      prevMouse = { x: e.clientX, y: e.clientY };
      updateCamera();
    });
    renderer.domElement.addEventListener("wheel", e => {
      spherical.radius = clamp(spherical.radius * (1 + e.deltaY * 0.001), toScene(RE * 1.2), toScene(200000));
      updateCamera();
    }, { passive: true });

    // ── Resize observer ──
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(el);

    // ── Animation loop ──
    let lastT = null;
    let animId;
    const animate = (ts) => {
      animId = requestAnimationFrame(animate);
      const dt = lastT ? Math.min((ts - lastT) / 1000, 0.1) : 0;
      lastT = ts;

      const s = simRef.current;
      if (s.playing) s.time += dt * s.timeScale;

      const t = s.time;
      const { a, e, i: iDeg, raan: raanDeg, omega: omegaDeg, nu: nuDeg } = elemsRef.current;
      const iRad = iDeg * DEG;
      const raanRad = raanDeg * DEG;
      const omegaRad = omegaDeg * DEG;
      const T = orbitalPeriod(a);

      // Mean anomaly at current time, starting from nuDeg
      const M0 = trueToMean(nuDeg * DEG, e);
      const M = (M0 + (2 * Math.PI / T) * t) % (2 * Math.PI);
      const E = solveKepler(M, e);
      const nuNow = eccentricToTrue(E, e);

      const { pos: posECI, vel: velECI, r } = elementsToECI(a, e, iRad, raanRad, omegaRad, nuNow);
      const theta = OMEGA_E * t; // GMST-like angle

      // Position in selected frame
      const fr = frameRef.current;
      let satPos;
      if (fr === "ECEF") {
        satPos = eciToECEF(posECI, theta).multiplyScalar(SCENE_SCALE);
      } else if (fr === "Perifocal") {
        // PQW: orbit plane coords
        const p_ = a * (1 - e * e);
        const rv = p_ / (1 + e * Math.cos(nuNow));
        satPos = new THREE.Vector3(rv * Math.cos(nuNow), rv * Math.sin(nuNow), 0).multiplyScalar(SCENE_SCALE);
      } else {
        satPos = posECI.clone().multiplyScalar(SCENE_SCALE);
      }

      // Update satellite position
      if (threeRef.current.satMesh) threeRef.current.satMesh.position.copy(satPos);

      // Earth rotation (ECEF frame rotates Earth)
      if (threeRef.current.earthMesh) {
        if (fr === "ECEF") {
          threeRef.current.earthMesh.rotation.y = theta;
          threeRef.current.gridMesh.rotation.y = theta;
          threeRef.current.pmLine.rotation.y = theta;
        } else {
          threeRef.current.earthMesh.rotation.y = 0;
          threeRef.current.gridMesh.rotation.y = 0;
          threeRef.current.pmLine.rotation.y = 0;
        }
      }

      // Orbit path
      const pathPts = generateOrbitPath(a, e, iRad, raanRad, omegaRad, 256);
      let scenePts;
      if (fr === "ECEF") {
        scenePts = pathPts.map(p => eciToECEF(p, theta).multiplyScalar(SCENE_SCALE));
      } else if (fr === "Perifocal") {
        scenePts = [];
        for (let k = 0; k <= 256; k++) {
          const nu_ = (2 * Math.PI * k) / 256;
          const p_ = a * (1 - e * e);
          const rv = p_ / (1 + e * Math.cos(nu_));
          scenePts.push(new THREE.Vector3(rv * Math.cos(nu_), rv * Math.sin(nu_), 0).multiplyScalar(SCENE_SCALE));
        }
      } else {
        scenePts = pathPts.map(p => p.clone().multiplyScalar(SCENE_SCALE));
      }

      const positions = new Float32Array(scenePts.length * 3);
      scenePts.forEach((p, idx) => { positions[idx * 3] = p.x; positions[idx * 3 + 1] = p.y; positions[idx * 3 + 2] = p.z; });
      threeRef.current.orbitLineMesh.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      threeRef.current.orbitLineMesh.geometry.computeBoundingSphere();

      // Orbital plane normal (ECI h vector)
      const hVec = new THREE.Vector3().crossVectors(posECI, velECI).normalize();
      const planeMeshRef = threeRef.current.planeMesh;
      if (planeMeshRef) {
        planeMeshRef.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), fr === "Perifocal" ? new THREE.Vector3(0, 0, 1) : hVec);
        const maxR = a * (1 + e);
        planeMeshRef.scale.setScalar(toScene(maxR * 1.15) / toScene(45000));
      }

      // Velocity arrow
      const velDir = velECI.clone().normalize();
      const velMag = velECI.length();
      const va = threeRef.current.velArrow;
      if (va && togglesRef.current.velVec) {
        va.position.copy(satPos);
        va.setDirection(fr === "ECEF" ? eciToECEF(velDir, theta) : velDir);
        va.setLength(toScene(velMag * 250), toScene(300), toScene(150));
        va.visible = true;
      } else if (va) va.visible = false;

      // Angular momentum arrow
      const ha = threeRef.current.hArrow;
      if (ha && togglesRef.current.hVec) {
        ha.position.set(0, 0, 0);
        ha.setDirection(fr === "Perifocal" ? new THREE.Vector3(0, 0, 1) : hVec);
        ha.setLength(toScene(RE * 1.8), toScene(300), toScene(150));
        ha.visible = true;
      } else if (ha) ha.visible = false;

      // Node vector (line of nodes: K × h normalized)
      const nodeVec = new THREE.Vector3(0, 0, 1).cross(hVec).normalize();
      const na = threeRef.current.nodeArrow;
      if (na && togglesRef.current.nodeVec && nodeVec.length() > 0.01) {
        na.position.set(0, 0, 0);
        na.setDirection(fr === "Perifocal" ? new THREE.Vector3(1, 0, 0) : nodeVec);
        na.setLength(toScene(RE * 2.0), toScene(250), toScene(125));
        na.visible = true;
      } else if (na) na.visible = false;

      // Perigee & Apogee
      const { pos: periPos } = elementsToECI(a, e, iRad, raanRad, omegaRad, 0);
      const { pos: apoPos  } = elementsToECI(a, e, iRad, raanRad, omegaRad, Math.PI);
      const peri = threeRef.current.perigeeMarker;
      const apo  = threeRef.current.apogeeMarker;
      if (peri) {
        peri.position.copy(fr === "ECEF" ? eciToECEF(periPos, theta).multiplyScalar(SCENE_SCALE) : periPos.clone().multiplyScalar(SCENE_SCALE));
        peri.visible = togglesRef.current.perigeeMarker;
      }
      if (apo) {
        apo.position.copy(fr === "ECEF" ? eciToECEF(apoPos, theta).multiplyScalar(SCENE_SCALE) : apoPos.clone().multiplyScalar(SCENE_SCALE));
        apo.visible = e > 0.01 && togglesRef.current.apogeeMarker;
      }

      // Vernal equinox (ECI X)
      if (threeRef.current.veLine) threeRef.current.veLine.visible = togglesRef.current.vernalEquinox;
      if (threeRef.current.axisGroup) threeRef.current.axisGroup.visible = togglesRef.current.axes;
      if (threeRef.current.rotAxisLine) threeRef.current.rotAxisLine.visible = togglesRef.current.earthRotAxis;
      if (threeRef.current.equatorLine) threeRef.current.equatorLine.visible = togglesRef.current.equator;
      if (threeRef.current.pmLine) threeRef.current.pmLine.visible = togglesRef.current.primeMeridian;
      if (threeRef.current.orbitLineMesh) threeRef.current.orbitLineMesh.visible = togglesRef.current.orbitPath;
      if (threeRef.current.planeMesh) threeRef.current.planeMesh.visible = togglesRef.current.orbitalPlane;
      if (threeRef.current.satMesh) threeRef.current.satMesh.visible = togglesRef.current.satellite;

      // Wireframe toggle
      if (threeRef.current.earthMesh) threeRef.current.earthMesh.material.wireframe = togglesRef.current.wireframe;

      // Trail
      const trailData = trailRef.current;
      if (togglesRef.current.trail && s.playing) {
        trailData.push(satPos.clone());
        if (trailData.length > 500) trailData.shift();
      }
      const tPos = threeRef.current.trailPositions;
      const tLine = threeRef.current.trailLine;
      if (tLine) {
        const N = Math.min(trailData.length, 500);
        for (let k = 0; k < N; k++) {
          const p = trailData[k];
          tPos[k * 3] = p.x; tPos[k * 3 + 1] = p.y; tPos[k * 3 + 2] = p.z;
        }
        tLine.geometry.attributes.position.needsUpdate = true;
        tLine.geometry.setDrawRange(0, N);
        tLine.visible = togglesRef.current.trail;
      }

      // Telemetry update (throttled to React state ~10/s)
      const geo = ecefToGeodetic(fr === "ECEF" ? eciToECEF(posECI, theta) : posECI);
      setSimTime(Math.round(s.time));
      setTelem({
        pos: [posECI.x.toFixed(1), posECI.y.toFixed(1), posECI.z.toFixed(1)],
        vel: [velECI.x.toFixed(3), velECI.y.toFixed(3), velECI.z.toFixed(3)],
        lat: geo.lat.toFixed(2), lon: geo.lon.toFixed(2), alt: geo.alt.toFixed(1),
        period: (T / 60).toFixed(1), r: r.toFixed(1),
        speed: velMag.toFixed(3),
      });

      renderer.render(scene, camera);
    };
    animId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, []);

  // Sync refs to state
  useEffect(() => { elemsRef.current = elems; trailRef.current = []; }, [elems]);
  useEffect(() => { simRef.current.playing = playing; }, [playing]);
  useEffect(() => { simRef.current.timeScale = timeScale; }, [timeScale]);
  useEffect(() => { frameRef.current = frame; }, [frame]);
  useEffect(() => { togglesRef.current = toggles; }, [toggles]);

  // Helpers
  const setElem = (k, v) => setElems(prev => ({ ...prev, [k]: v }));
  const applyPreset = (p) => { setElems({ ...PRESETS[p] }); simRef.current.time = 0; trailRef.current = []; };
  const resetSim = () => { simRef.current.time = 0; trailRef.current = []; setPlaying(false); };
  const toggleVis = (k) => setToggles(prev => ({ ...prev, [k]: !prev[k] }));

  const sliderStyle = (pct) => ({
    background: `linear-gradient(to right, ${C.accent} 0%, ${C.accent} ${pct}%, #0d1f35 ${pct}%, #0d1f35 100%)`
  });

  // ─────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────
  return (
    <div style={{ fontFamily: "'Courier New', monospace", background: C.bg, color: C.white, width: "100%", height, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      {/* Top bar */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 14px", background: C.bgPanel, borderBottom: `1px solid ${C.border}`, flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 13, letterSpacing: "0.2em", color: C.accent, fontWeight: "bold" }}>◉ ORBITAL MECHANICS</span>
          <span style={{ fontSize: 11, color: C.dim }}>/ interactive visualizer</span>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {Object.keys(PRESETS).map(k => (
            <button key={k} onClick={() => applyPreset(k)}
              style={{ fontSize: 10, padding: "2px 8px", background: "transparent", border: `1px solid ${C.border}`, color: C.dim, cursor: "pointer", letterSpacing: "0.1em" }}
              onMouseEnter={e => { e.target.style.borderColor = C.accent; e.target.style.color = C.accent; }}
              onMouseLeave={e => { e.target.style.borderColor = C.border; e.target.style.color = C.dim; }}>
              {PRESETS[k].label}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <FrameSelector frame={frame} setFrame={setFrame} />
        </div>
      </div>

      {/* Main body */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden", minHeight: 0 }}>

        {/* Left panel – orbital elements */}
        <div style={{ width: 210, background: C.bgPanel, borderRight: `1px solid ${C.border}`, padding: "10px 12px", overflowY: "auto", flexShrink: 0, display: "flex", flexDirection: "column", gap: 8 }}>
          <SectionLabel>Orbital Elements</SectionLabel>

          <ParamSlider label="a (km)" value={elems.a} min={RE + 100} max={50000} step={50}
            pct={((elems.a - (RE + 100)) / (50000 - (RE + 100))) * 100}
            onChange={v => setElem("a", v)} />
          <ParamSlider label="e" value={elems.e} min={0} max={0.99} step={0.001}
            pct={elems.e / 0.99 * 100}
            onChange={v => setElem("e", v)} decimals={4} />
          <ParamSlider label="i (°)" value={elems.i} min={0} max={180} step={0.1}
            pct={elems.i / 180 * 100}
            onChange={v => setElem("i", v)} />
          <ParamSlider label="Ω RAAN (°)" value={elems.raan} min={0} max={360} step={0.5}
            pct={elems.raan / 360 * 100}
            onChange={v => setElem("raan", v)} />
          <ParamSlider label="ω Arg. Per. (°)" value={elems.omega} min={0} max={360} step={0.5}
            pct={elems.omega / 360 * 100}
            onChange={v => setElem("omega", v)} />
          <ParamSlider label="ν True Anom. (°)" value={elems.nu} min={0} max={360} step={0.5}
            pct={elems.nu / 360 * 100}
            onChange={v => setElem("nu", v)} />

          <SectionLabel style={{ marginTop: 6 }}>Simulation</SectionLabel>
          <div style={{ display: "flex", gap: 5, marginBottom: 4 }}>
            <SimBtn active={playing} onClick={() => setPlaying(p => !p)}>{playing ? "⏸ PAUSE" : "▶ PLAY"}</SimBtn>
            <SimBtn onClick={resetSim}>↺ RST</SimBtn>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 10, color: C.dim, width: 52, flexShrink: 0 }}>Speed ×{timeScale}</span>
            <input type="range" min={1} max={10000} step={10} value={timeScale}
              onChange={e => setTimeScale(+e.target.value)}
              style={{ flex: 1, accentColor: C.accent, cursor: "pointer" }} />
          </div>
          <div style={{ fontSize: 10, color: C.dim }}>T+{simTime}s</div>

          <SectionLabel style={{ marginTop: 6 }}>Display</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {[
              ["orbitPath",     "Orbit Path"],
              ["satellite",     "Satellite"],
              ["trail",         "Trail"],
              ["velVec",        "Velocity Vec"],
              ["hVec",          "Ang. Mom. Vec"],
              ["orbitalPlane",  "Orbital Plane"],
              ["nodeVec",       "Line of Nodes"],
              ["perigeeMarker", "Perigee"],
              ["apogeeMarker",  "Apogee"],
              ["equator",       "Equator"],
              ["primeMeridian", "Prime Meridian"],
              ["axes",          "Frame Axes"],
              ["vernalEquinox", "Vernal Equinox ♈"],
              ["earthRotAxis",  "Rot. Axis"],
              ["wireframe",     "Wireframe"],
            ].map(([k, lbl]) => (
              <label key={k} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 10, color: toggles[k] ? C.white : C.dim }}>
                <input type="checkbox" checked={toggles[k]} onChange={() => toggleVis(k)}
                  style={{ accentColor: C.accent, width: 12, height: 12, cursor: "pointer" }} />
                {lbl}
              </label>
            ))}
          </div>
        </div>

        {/* Center – 3D viewport */}
        <div ref={mountRef} style={{ flex: 1, position: "relative", background: C.bg, minWidth: 0 }}>
          {/* Frame labels overlay */}
          <div style={{ position: "absolute", top: 8, left: 8, pointerEvents: "none" }}>
            <div style={{ fontSize: 11, color: C.accent, letterSpacing: "0.15em", opacity: 0.8 }}>
              FRAME: {frame}
            </div>
            <div style={{ fontSize: 9, color: C.dim, marginTop: 2 }}>
              <span style={{ color: "#ff3344" }}>■</span> X  <span style={{ color: "#33ff55" }}>■</span> Y  <span style={{ color: "#3399ff" }}>■</span> Z  <span style={{ color: "#ffee00" }}>■</span> ♈
            </div>
          </div>
          {/* Vector legend */}
          <div style={{ position: "absolute", bottom: 8, left: 8, pointerEvents: "none", display: "flex", flexDirection: "column", gap: 3 }}>
            <LegendItem color={C.orbitLine}>Orbit Path</LegendItem>
            <LegendItem color={C.velVec}>Velocity</LegendItem>
            <LegendItem color={C.hVec}>Ang. Momentum (h)</LegendItem>
            <LegendItem color={C.nodeVec}>Line of Nodes</LegendItem>
            <LegendItem color={C.perigee}>Perigee</LegendItem>
            <LegendItem color={C.apogee}>Apogee</LegendItem>
          </div>
        </div>

        {/* Right panel – telemetry */}
        <div style={{ width: 200, background: C.bgPanel, borderLeft: `1px solid ${C.border}`, padding: "10px 12px", overflowY: "auto", flexShrink: 0 }}>
          <SectionLabel>Telemetry</SectionLabel>
          <TelemRow label="Frame" value={frame} color={C.accent} />
          <TelemRow label="a" value={`${elems.a.toFixed(0)} km`} />
          <TelemRow label="e" value={elems.e.toFixed(4)} />
          <TelemRow label="i" value={`${elems.i.toFixed(2)}°`} />
          <TelemRow label="RAAN" value={`${elems.raan.toFixed(2)}°`} />
          <TelemRow label="ω" value={`${elems.omega.toFixed(2)}°`} />
          <hr style={{ borderColor: C.border, margin: "6px 0" }} />
          <TelemRow label="|r|" value={`${telem.r} km`} />
          <TelemRow label="|v|" value={`${telem.speed} km/s`} color={C.velVec} />
          <TelemRow label="Period" value={`${telem.period} min`} />
          <hr style={{ borderColor: C.border, margin: "6px 0" }} />
          <SectionLabel>Position (ECI)</SectionLabel>
          <TelemRow label="X" value={`${telem.pos[0]} km`} color="#ff7788" />
          <TelemRow label="Y" value={`${telem.pos[1]} km`} color="#77ff99" />
          <TelemRow label="Z" value={`${telem.pos[2]} km`} color="#77aaff" />
          <hr style={{ borderColor: C.border, margin: "6px 0" }} />
          <SectionLabel>Velocity (ECI)</SectionLabel>
          <TelemRow label="Vx" value={`${telem.vel[0]}`} color="#ff7788" />
          <TelemRow label="Vy" value={`${telem.vel[1]}`} color="#77ff99" />
          <TelemRow label="Vz" value={`${telem.vel[2]}`} color="#77aaff" />
          <hr style={{ borderColor: C.border, margin: "6px 0" }} />
          <SectionLabel>Geodetic</SectionLabel>
          <TelemRow label="Lat" value={`${telem.lat}°`} />
          <TelemRow label="Lon" value={`${telem.lon}°`} />
          <TelemRow label="Alt" value={`${telem.alt} km`} color={C.gold} />
          <hr style={{ borderColor: C.border, margin: "6px 0" }} />
          <SectionLabel>Orbit Type</SectionLabel>
          <div style={{ fontSize: 10, color: C.dim, lineHeight: 1.6 }}>
            {elems.e < 0.01 ? "Circular" : elems.e < 0.25 ? "Near-Circular" : "Elliptical"} <br />
            {elems.i > 170 ? "Retrograde" : elems.i > 80 ? "Polar" : elems.i < 10 ? "Equatorial" : "Inclined"} <br />
            {elems.a > 35786 + RE ? "High Earth" : elems.a > 20000 ? "MEO" : elems.a < RE + 2000 ? "LEO" : "GEO/MEO"}
          </div>
        </div>
      </div>

      {/* Status bar */}
      <div style={{ height: 22, background: C.bgPanel, borderTop: `1px solid ${C.border}`, display: "flex", alignItems: "center", padding: "0 14px", gap: 20, flexShrink: 0 }}>
        <StatusDot active={playing} />
        <span style={{ fontSize: 9, color: C.dim, letterSpacing: "0.1em" }}>TWO-BODY KEPLERIAN</span>
        <span style={{ fontSize: 9, color: C.dim }}>μ = {MU} km³/s²</span>
        <span style={{ fontSize: 9, color: C.dim }}>Rₑ = {RE} km</span>
        <span style={{ fontSize: 9, color: C.dim }}>ωₑ = {OMEGA_E.toExponential(2)} rad/s</span>
      </div>

      <style>{`
        input[type=range] { height: 4px; border-radius: 2px; }
        input[type=range]::-webkit-slider-thumb { width: 12px; height: 12px; border-radius: 50%; background: ${C.accent}; cursor: pointer; -webkit-appearance: none; border: 2px solid ${C.bg}; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
      `}</style>
    </div>
  );
}

// ─────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────

function SectionLabel({ children, style }) {
  return <div style={{ fontSize: 9, letterSpacing: "0.2em", color: C.accentDim, textTransform: "uppercase", marginBottom: 4, ...style }}>{children}</div>;
}

function TelemRow({ label, value, color }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 3 }}>
      <span style={{ color: C.dim }}>{label}</span>
      <span style={{ color: color || C.white, fontVariantNumeric: "tabular-nums" }}>{value}</span>
    </div>
  );
}

function LegendItem({ color, children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 9, color: C.dim }}>
      <span style={{ width: 12, height: 2, background: color, display: "inline-block" }} />
      {children}
    </div>
  );
}

function SimBtn({ children, onClick, active }) {
  return (
    <button onClick={onClick} style={{
      flex: 1, fontSize: 10, padding: "4px 0", background: active ? C.accent + "22" : "transparent",
      border: `1px solid ${active ? C.accent : C.border}`, color: active ? C.accent : C.dim,
      cursor: "pointer", letterSpacing: "0.1em", transition: "all 0.15s"
    }}>
      {children}
    </button>
  );
}

function StatusDot({ active }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <span style={{
        width: 6, height: 6, borderRadius: "50%", background: active ? C.green : C.dim,
        boxShadow: active ? `0 0 6px ${C.green}` : "none",
        display: "inline-block", transition: "all 0.3s"
      }} />
      <span style={{ fontSize: 9, color: active ? C.green : C.dim, letterSpacing: "0.15em" }}>
        {active ? "PROPAGATING" : "STANDBY"}
      </span>
    </span>
  );
}

function FrameSelector({ frame, setFrame }) {
  const frames = ["ECI", "ECEF", "Perifocal"];
  return (
    <div style={{ display: "flex", gap: 4 }}>
      {frames.map(f => (
        <button key={f} onClick={() => setFrame(f)} style={{
          fontSize: 10, padding: "2px 8px",
          background: frame === f ? C.accent + "22" : "transparent",
          border: `1px solid ${frame === f ? C.accent : C.border}`,
          color: frame === f ? C.accent : C.dim,
          cursor: "pointer", letterSpacing: "0.12em", transition: "all 0.15s"
        }}>
          {f}
        </button>
      ))}
    </div>
  );
}

function ParamSlider({ label, value, min, max, step, pct, onChange, decimals = 1 }) {
  return (
    <div style={{ marginBottom: 2 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 2 }}>
        <span style={{ color: C.dim }}>{label}</span>
        <input
          type="number" value={value} min={min} max={max} step={step}
          onChange={e => onChange(clamp(parseFloat(e.target.value) || 0, min, max))}
          style={{
            width: 72, background: "transparent", border: `1px solid ${C.border}`,
            color: C.white, fontSize: 10, padding: "1px 4px", textAlign: "right",
            outline: "none", fontFamily: "inherit"
          }}
        />
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: C.accent, cursor: "pointer", ...({}) }}
      />
    </div>
  );
}
