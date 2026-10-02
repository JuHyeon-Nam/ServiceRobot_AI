import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const FLOOR_GAP = 34; // 층 간 높이(world)
const S = 1.0; // 좌표 스케일
const CX = 150,
  CZ_OFF = 26; // 중심 보정(캔버스 300x196)

const app = document.getElementById("app");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf3f5f6);
scene.fog = new THREE.Fog(0xf3f5f6, 450, 900);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
const HOME_POS = new THREE.Vector3(185, 165, 260);
const HOME_TGT = new THREE.Vector3(-15, FLOOR_GAP, 0);
camera.position.copy(HOME_POS);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(app.clientWidth || 800, app.clientHeight || 600);
renderer.setClearColor(0xf3f5f6);
app.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.copy(HOME_TGT);
controls.minDistance = 60;
controls.maxDistance = 520;
controls.maxPolarAngle = Math.PI * 0.49;
controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
controls.autoRotate = true;
controls.autoRotateSpeed = 0.06;

scene.add(new THREE.AmbientLight(0xffffff, 0.85));
scene.add(new THREE.HemisphereLight(0xffffff, 0xc6d3e1, 0.95)); // 밝은 전시장/관제 조명
const dir = new THREE.DirectionalLight(0xffffff, 0.95);
dir.position.set(120, 240, 160);
scene.add(dir);
const fill = new THREE.DirectionalLight(0x9db7d8, 0.55); // 반대편 필 라이트(입체감)
fill.position.set(-140, 90, -120);
scene.add(fill);
const grid = new THREE.GridHelper(640, 40, 0xc3cfcd, 0xe1e7e6);
grid.position.y = -2;
scene.add(grid);

// floorY: build_agv_plan floor index 0=2F(top),1=1F,2=B1(bottom)
const floorY = (f) => (2 - f) * FLOOR_GAP;
const wx = (x) => (x - CX) * S;
const wz = (y, y0) => (y - y0 - CZ_OFF) * S;

const agvGroup = new THREE.Group();
scene.add(agvGroup);
const agvMesh = {}; // id -> {mesh, target:{x,y,z,ang}}
let LAYOUT = null,
  FLOOR_Y0 = {};

function buildLayout(L) {
  LAYOUT = L;
  L.floors.forEach((fl, fi) => {
    const deck = new THREE.Group();
    decks[fi] = deck;
    scene.add(deck);
    FLOOR_Y0[fi] = fl.y0;
    const ey = floorY(fi);
    // floor slab
    const x0 = 8,
      x1 = 252,
      zA = fl.geo.yA,
      zC = fl.geo.yC;
    const w = (x1 - x0) * S,
      d = (zC - zA) * S;
    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(w, 1.4, d),
      new THREE.MeshStandardMaterial({
        color: 0xdde8e4,
        transparent: true,
        opacity: 0.94,
        roughness: 0.9,
      }),
    );
    slab.position.set(
      ((x0 + x1) / 2 - CX) * S,
      ey,
      ((zA + zC) / 2 - fl.y0 - CZ_OFF) * S,
    );
    deck.add(slab);
    // 공정 설비 — 베이스 플린스 + 본체(베이별 색 변주·높이 변주) + 상단 서비스 유닛 + 로드포트(FOUP)
    const BAYCOL = [0xb8c8c6, 0xc5cfcc, 0xaac3bd, 0xd0d4d1, 0xb0c4c6];
    fl.tools.forEach((t) => {
      const th = 9 + (t.label.charCodeAt(t.label.length - 1) % 4) * 1.8; // 높이 변주(균일 격자 탈피)
      const bc = BAYCOL[t.label.charCodeAt(0) % BAYCOL.length]; // 베이별 색 변주
      const cx3 = wx(t.cx),
        cz3 = wz(t.cy, fl.y0);
      const base = new THREE.Mesh(
        new THREE.BoxGeometry(t.w * S + 2, 1.6, t.h * S + 2), // 바닥 받침
        new THREE.MeshStandardMaterial({ color: 0xc3cfdd, roughness: 0.9 }),
      );
      base.position.set(cx3, ey + 1.3, cz3);
      deck.add(base);
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(t.w * S, th, t.h * S), // 본체
        new THREE.MeshStandardMaterial({
          color: bc,
          roughness: 0.5,
          metalness: 0.32,
        }),
      );
      m.position.set(cx3, ey + th / 2 + 1.6, cz3);
      deck.add(m);
      const edge = new THREE.LineSegments(
        new THREE.EdgesGeometry(m.geometry),
        new THREE.LineBasicMaterial({ color: 0x65758b }),
      );
      edge.position.copy(m.position);
      deck.add(edge);
      const top = new THREE.Mesh(
        new THREE.BoxGeometry(t.w * S * 0.5, 3, t.h * S * 0.5), // 상단 유닛(EFEM/팬필터)
        new THREE.MeshStandardMaterial({
          color: 0x7f91a8,
          roughness: 0.6,
          metalness: 0.2,
        }),
      );
      top.position.set(cx3, ey + th + 1.6 + 1.5, cz3);
      deck.add(top);
      for (const s of [-0.26, 0.26]) {
        // 로드포트(FOUP) 앞면 큐브 2개
        const lp = new THREE.Mesh(
          new THREE.BoxGeometry(Math.max(t.w * S * 0.16, 2), 2.6, 2.2),
          new THREE.MeshStandardMaterial({
            color: 0x6b83a0,
            roughness: 0.45,
            metalness: 0.15,
          }),
        );
        lp.position.set(cx3 + t.w * S * s, ey + 2.9, cz3 - t.h * S * 0.5 - 1.1);
        deck.add(lp);
      }
    });
    // AGV 주행 레인(바닥 페인트 라인) + 중앙 가이드라인
    fl.tracks.forEach(([ax, ay, bx, by]) => {
      const x1 = wx(ax),
        z1 = wz(ay, fl.y0),
        x2 = wx(bx),
        z2 = wz(by, fl.y0);
      const len = Math.hypot(x2 - x1, z2 - z1);
      if (len < 0.5) return;
      const lane = new THREE.Mesh(
        new THREE.BoxGeometry(len, 0.3, 2.6),
        new THREE.MeshStandardMaterial({
          color: 0xb8c7d9,
          roughness: 0.85,
          emissive: 0x5f7ea8,
          emissiveIntensity: 0.12,
        }),
      );
      lane.position.set((x1 + x2) / 2, ey + 0.8, (z1 + z2) / 2);
      lane.rotation.y = -Math.atan2(z2 - z1, x2 - x1);
      deck.add(lane);
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x1, ey + 1.0, z1),
        new THREE.Vector3(x2, ey + 1.0, z2),
      ]);
      deck.add(
        new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x64748b })),
      );
    });
    // OHT 오버헤드 레일(천장 반송)
    (fl.oht || []).forEach(([ax, ay, bx, by]) => {
      const oy = ey + 11;
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(wx(ax), oy, wz(ay, fl.y0)),
        new THREE.Vector3(wx(bx), oy, wz(by, fl.y0)),
      ]);
      deck.add(
        new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x4a6098 })),
      );
    });
    // floor label
    deck.add(
      makeLabel(
        `${fl.short} / ${["공정", "이송", "유틸리티"][fi]}`,
        (x0 - CX) * S - 6,
        ey + 14,
        ((zA + zC) / 2 - fl.y0 - CZ_OFF) * S,
      ),
    );
  });

  // ===== 스토커 타워 + 층간 리프트 (전 층 관통 구조물) =====
  const yBot = floorY(2) - 3,
    yTop = floorY(0) + 16,
    TH = yTop - yBot,
    ymid = (yBot + yTop) / 2;
  function pillar(cx, cz, w, d, color, op, label) {
    const m = new THREE.Mesh(
      new THREE.BoxGeometry(w, TH, d),
      new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity: op,
        roughness: 0.75,
      }),
    );
    m.position.set(cx, ymid, cz);
    structures.add(m);
    const e = new THREE.LineSegments(
      new THREE.EdgesGeometry(m.geometry),
      new THREE.LineBasicMaterial({ color: 0x5a6ea8 }),
    );
    e.position.copy(m.position);
    structures.add(e);
    if (label) structures.add(makeLabel(label, cx - 3.5, yTop + 5, cz));
    return m;
  }
  if (LAYOUT.stocker) {
    const S2 = LAYOUT.stocker,
      cx = wx(S2.x + S2.w / 2);
    pillar(cx, 0, 9, 26, 0x2a3557, 0.92, "스토커");
    for (let k = 1; k < 6; k++) {
      // 선반 층 라인
      const yy = yBot + (TH * k) / 6;
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(cx - 4.5, yy, -13),
        new THREE.Vector3(cx - 4.5, yy, 13),
      ]);
      structures.add(
        new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x46587f })),
      );
    }
  }
  if (LAYOUT.lift) {
    const lx = wx(LAYOUT.lift.x + LAYOUT.lift.w / 2);
    pillar(lx, 0, 5, 20, 0x24304e, 0.5, "층간 리프트");
    L.floors.forEach((fl, fi) => {
      // 층별 리프트 캐빈
      const cab = new THREE.Mesh(
        new THREE.BoxGeometry(4, 5, 8),
        new THREE.MeshStandardMaterial({
          color: 0x3f7fb0,
          emissive: 0x143a52,
          emissiveIntensity: 0.6,
        }),
      );
      cab.position.set(lx, floorY(fi) + 5, 0);
      structures.add(cab);
    });
  }
}

function makeLabel(text, x, y, z) {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 72;
  const g = c.getContext("2d");
  g.fillStyle = "rgba(255,255,255,.95)";
  g.fillRect(0, 0, 512, 72);
  g.fillStyle = "#294740";
  g.font = "600 36px sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 256, 36, 480);
  const tex = new THREE.CanvasTexture(c);
  const sp = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, transparent: true }),
  );
  sp.scale.set(58, 8.2, 1);
  sp.position.set(x, y, z);
  return sp;
}

// 실제 팹 AGV(무인반송차) 형상: 하부 대차 + 바퀴 4 + 상단 FOUP 포드 + 상태 경광등(안돈)
function agvFor(a) {
  if (agvMesh[a.id]) return agvMesh[a.id];
  const grp = new THREE.Group();
  const steel = new THREE.MeshStandardMaterial({
    color: 0xc3ccdf,
    roughness: 0.5,
    metalness: 0.35,
  });
  const dark = new THREE.MeshStandardMaterial({
    color: 0x252d45,
    roughness: 0.75,
    metalness: 0.2,
  });
  const skirt = new THREE.Mesh(new THREE.BoxGeometry(7.6, 1.6, 10.6), dark); // 하부 대차
  skirt.position.y = 1.0;
  grp.add(skirt);
  const chassis = new THREE.Mesh(new THREE.BoxGeometry(7.0, 2.6, 9.8), steel); // 본체 섀시
  chassis.position.y = 3.0;
  grp.add(chassis);
  const wheelGeo = new THREE.CylinderGeometry(1.1, 1.1, 0.9, 14); // 바퀴 4개
  const wheelMat = new THREE.MeshStandardMaterial({
    color: 0x0e121c,
    roughness: 0.9,
  });
  for (const sx of [-3.5, 3.5])
    for (const sz of [-3.2, 3.2]) {
      const w = new THREE.Mesh(wheelGeo, wheelMat);
      w.rotation.z = Math.PI / 2;
      w.position.set(sx, 1.05, sz);
      grp.add(w);
    }
  const foup = new THREE.Mesh(
    new THREE.BoxGeometry(5.6, 4.4, 5.6), // 상단 반송 FOUP 포드
    new THREE.MeshStandardMaterial({
      color: 0x9fb3d6,
      roughness: 0.3,
      metalness: 0.15,
      transparent: true,
      opacity: 0.92,
    }),
  );
  foup.position.y = 6.6;
  grp.add(foup);
  const lid = new THREE.Mesh(new THREE.BoxGeometry(5.8, 0.8, 5.8), dark); // 포드 상단 커버
  lid.position.y = 9.2;
  grp.add(lid);
  const head = new THREE.Mesh(
    new THREE.BoxGeometry(4.2, 0.9, 0.7), // 전방 진행등(+Z가 앞)
    new THREE.MeshStandardMaterial({
      color: 0x22c55e,
      emissive: 0x22c55e,
      emissiveIntensity: 0.5,
    }),
  );
  head.position.set(0, 3.2, 5.0);
  grp.add(head);
  const beacon = new THREE.Mesh(
    new THREE.CylinderGeometry(0.95, 0.95, 2.6, 14), // 상단 상태 경광등(안돈)
    new THREE.MeshStandardMaterial({
      color: 0x22c55e,
      emissive: 0x22c55e,
      emissiveIntensity: 0.7,
      transparent: true,
      opacity: 0.95,
    }),
  );
  beacon.position.y = 10.8;
  grp.add(beacon);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(6.8, 0.7, 8, 30), // 바닥 경고 헤일로
    new THREE.MeshBasicMaterial({
      color: 0xff3b3b,
      transparent: true,
      opacity: 0.9,
    }),
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.5;
  ring.visible = false;
  grp.add(ring);
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.35, 0.35, 18, 8),
    new THREE.MeshBasicMaterial({
      color: 0xff5a5a,
      transparent: true,
      opacity: 0.24,
    }),
  );
  beam.position.y = 16;
  beam.visible = false;
  grp.add(beam);
  grp.userData = { agv: true, id: a.id };
  agvGroup.add(grp);
  const rec = {
    mesh: grp,
    beacon,
    head,
    foup,
    ring,
    beam,
    target: { x: 0, y: 0, z: 0, ang: 0 },
  };
  agvMesh[a.id] = rec;
  return rec;
}

let selected = null,
  focusing = false,
  homing = false,
  orbit = !matchMedia("(prefers-reduced-motion: reduce)").matches;
const decks = [],
  structures = new THREE.Group();
scene.add(structures);
let onSelect = () => {};
const selectedRing = new THREE.Mesh(
  new THREE.TorusGeometry(8.4, 0.35, 8, 48),
  new THREE.MeshBasicMaterial({ color: 0x007e80, depthTest: false }),
);
selectedRing.rotation.x = Math.PI / 2;
selectedRing.visible = false;
scene.add(selectedRing);

export function initLayout(layout, callback) {
  buildLayout(layout);
  onSelect = callback;
}
export function updateScene(assets) {
  for (const a of assets) {
    const fresh = !agvMesh[a.id];
    const rec = agvFor(a);
    rec.data = a;
    rec.target = {
      x: wx(a.x),
      y: floorY(a.floor) + 1.3,
      z: wz(a.y, FLOOR_Y0[a.floor] || 0),
      ang: a.ang,
    };
    if (fresh) rec.mesh.position.set(rec.target.x, rec.target.y, rec.target.z);
    const stage = a.phm.stage;
    const color = {
      normal: 0x168263,
      watch: 0xe0b52e,
      predicted_fault: 0xe18538,
      current_fault: 0xd44045,
    }[stage];
    for (const mat of [
      rec.beacon.material,
      rec.head.material,
      rec.foup.material,
    ]) {
      mat.color.setHex(color);
      mat.emissive.setHex(color);
      mat.emissiveIntensity = stage === "normal" ? 0.08 : 0.3;
    }
    rec.ring.visible = stage !== "normal";
    rec.beam.visible = stage === "current_fault";
    rec.ring.material.color.setHex(color);
    rec.beam.material.color.setHex(color);
  }
}
export function selectAsset(id) {
  selected = id;
  focusing = !!id;
  homing = !id;
  const floor = agvMesh[id]?.data.floor;
  decks.forEach(
    (deck, index) => (deck.visible = floor == null || floor === index),
  );
  for (const deck of decks)
    for (const child of deck.children) {
      if (child.isSprite) child.visible = floor == null;
      if (child.isMesh && child.geometry.parameters?.height > 2) {
        const mat = child.material;
        child.userData.baseOpacity ??= mat.opacity;
        mat.transparent = floor != null;
        mat.opacity = floor == null ? child.userData.baseOpacity : 0.24;
        mat.depthWrite = floor == null;
      }
    }
  structures.visible = floor == null;
  for (const rec of Object.values(agvMesh))
    rec.mesh.visible = floor == null || rec.data.floor === floor;
}
export function home() {
  selectAsset(null);
  focusing = false;
  homing = true;
}
export function setOrbit(value) {
  orbit = value;
}
export function getOrbit() {
  return orbit;
}
const ray = new THREE.Raycaster(),
  pointer = new THREE.Vector2();
let down = null;
renderer.domElement.addEventListener("pointerdown", (e) => {
  down = [e.clientX, e.clientY];
});
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    (-(e.clientY - rect.top) / rect.height) * 2 + 1,
  );
  ray.setFromCamera(pointer, camera);
  const hit = ray.intersectObjects(
    agvGroup.children.filter((g) => g.visible),
    true,
  )[0];
  let object = hit?.object;
  while (object && !object.userData.agv) object = object.parent;
  if (object) onSelect(object.userData.id);
  down = null;
});
controls.addEventListener("start", () => {
  focusing = false;
  homing = false;
});
function resize() {
  const width = app.clientWidth,
    height = app.clientHeight;
  if (!width || !height) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  if (!selected && !focusing) {
    // Keep all three decks in frame on narrow/mobile viewports.
    const factor = Math.max(1, 1.55 / camera.aspect);
    camera.position
      .copy(HOME_TGT)
      .add(HOME_POS.clone().sub(HOME_TGT).multiplyScalar(factor));
    controls.target.copy(HOME_TGT);
  }
}
new ResizeObserver(resize).observe(app);
const clk = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clk.getDelta(), 0.1),
    ease = 1 - Math.exp(-5 * dt);
  if (!app.clientWidth) return;
  for (const rec of Object.values(agvMesh)) {
    const t = rec.target;
    rec.mesh.position.lerp(
      new THREE.Vector3(t.x, t.y, t.z),
      1 - Math.exp(-10 * dt),
    );
    const want = (-t.ang * Math.PI) / 180;
    const diff = Math.atan2(
      Math.sin(want - rec.mesh.rotation.y),
      Math.cos(want - rec.mesh.rotation.y),
    );
    rec.mesh.rotation.y += diff * (1 - Math.exp(-8 * dt));
  }
  const asset = agvMesh[selected];
  selectedRing.visible = !!asset;
  if (asset) {
    selectedRing.position.copy(asset.mesh.position);
    selectedRing.position.y += 0.6;
    if (focusing) {
      controls.target.lerp(asset.mesh.position, ease);
      const want = asset.mesh.position
        .clone()
        .add(new THREE.Vector3(18, 82, asset.target.z >= 0 ? 60 : -60));
      camera.position.lerp(want, ease);
    }
  }
  if (homing) {
    const factor = Math.max(1, 1.55 / camera.aspect);
    const want = HOME_TGT.clone().add(
      HOME_POS.clone().sub(HOME_TGT).multiplyScalar(factor),
    );
    camera.position.lerp(want, ease);
    controls.target.lerp(HOME_TGT, ease);
    if (camera.position.distanceTo(want) < 1) homing = false;
  }
  controls.autoRotate = orbit && !selected && !homing;
  controls.update();
  renderer.render(scene, camera);
}
animate();
