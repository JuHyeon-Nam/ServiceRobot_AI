const $ = (id) => document.getElementById(id);
const text = (id, value) => {
  $(id).textContent = value;
};
const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const icons = () => window.lucide?.createIcons();
const stageName = {
  normal: "정상",
  watch: "주의",
  predicted_fault: "예측 이상",
  current_fault: "현재 이상",
};
const floorName = ["2F 공정", "1F 이송", "B1 유틸리티"];
const sourceName = {
  replay_model: "리플레이 + 모델",
  demo_scenario: "수동 시나리오",
  edge_ingest: "외부 Edge 입력",
  legacy: "기존 기록",
  live_booster: "모델 진단",
};
const statusName = {
  open: "접수 대기",
  acknowledged: "접수됨",
  in_progress: "점검 중",
  resolved: "조치 완료",
  closed: "종결",
};
const openStatuses = ["open", "acknowledged", "in_progress"];
let state = null,
  selected = null,
  view = "twin",
  scene = null,
  orders = [],
  lastReceived = 0;
let ws = null,
  retryDelay = 1000,
  toastTimer,
  sampleAt = 0,
  busy = false,
  ordersSignature = "";
let buffers = { batt: [], vib: [], temp: [] };

async function api(path, options = {}) {
  const response = await fetch(path, {
    signal: AbortSignal.timeout(8000),
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `요청 실패 (${response.status})`);
  }
  return response.json();
}
function toast(message) {
  text("toast", message);
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("toast").hidden = true), 4500);
}
function setView(next) {
  view = next;
  for (const name of ["twin", "orders", "evidence"])
    $(name + "View").hidden = name !== next;
  $("inspector").hidden = next !== "twin";
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === next);
    button.setAttribute("aria-pressed", String(button.dataset.view === next));
  });
  if (next === "orders") {
    renderOrders();
    pollOperations();
  }
  if (next === "evidence") loadEvidence();
}
function select(id) {
  selected = id;
  if (matchMedia("(max-width: 600px)").matches)
    window.scrollTo({ top: 0, behavior: "smooth" });
  buffers = { batt: [], vib: [], temp: [] };
  sampleAt = 0;
  setView("twin");
  scene?.selectAsset(id);
  $("selection").hidden = !id;
  $("selectionEmpty").hidden = !!id;
  $("inspector").classList.toggle("has-selection", !!id);
  if (id) {
    const a = state?.agvs.find((a) => a.id === id);
    $("scenarioSelect").value = state?.demo.overrides[id] || "replay";
    if (a) renderSelected(a);
    $("inspector").querySelector(".inspector-scroll").scrollTop = 0;
    text("historyList", "수집 중");
    pollHistory();
  }
  renderAssets();
}
function filteredAssets() {
  const search = $("assetSearch").value.trim().toLowerCase(),
    floor = $("floorFilter").value,
    filter = $("stateFilter").value;
  return (state?.agvs || []).filter(
    (a) =>
      a.id.toLowerCase().includes(search) &&
      (floor === "all" || String(a.floor) === floor) &&
      (filter === "all" ||
        (filter === "normal"
          ? a.phm.stage === "normal"
          : a.phm.stage !== "normal")),
  );
}
function renderAssets() {
  const rows = filteredAssets(),
    list = $("assetList");
  text("assetCount", `${rows.length} / ${state?.agvs.length || 0}`);
  if (list.querySelector(".empty")) list.replaceChildren();
  const seen = new Set();
  let added = false;
  for (const [index, a] of rows.entries()) {
    seen.add(a.id);
    let row = list.querySelector(`[data-asset="${CSS.escape(a.id)}"]`);
    if (!row) {
      row = document.createElement("button");
      row.dataset.asset = a.id;
      row.innerHTML =
        '<span class="asset-avatar"><i data-lucide="bot"></i></span><span><strong></strong><small></small></span><span class="asset-state"><b></b><span></span></span>';
      row.addEventListener("click", () => select(a.id));
      list.append(row);
      added = true;
    }
    if (list.children[index] !== row)
      list.insertBefore(row, list.children[index] || null);
    row.className = `asset-row ${a.phm.stage}${a.id === selected ? " selected" : ""}`;
    row.setAttribute("aria-pressed", String(a.id === selected));
    row.setAttribute("aria-label", `${a.id} ${stageName[a.phm.stage]}`);
    row.querySelector("strong").textContent = a.id;
    row.querySelector("small").textContent =
      `${floorName[a.floor]} · ${a.source === "demo_scenario" ? "시나리오" : a.source === "edge_ingest" ? "EDGE" : "REPLAY"}`;
    row.querySelector(".asset-state b").textContent = a.health;
    row.querySelector(".asset-state span").textContent = stageName[a.phm.stage];
  }
  for (const row of list.querySelectorAll("[data-asset]"))
    if (!seen.has(row.dataset.asset)) row.remove();
  if (!rows.length)
    list.innerHTML = '<p class="empty">조건에 맞는 자산이 없습니다.</p>';
  if (added) icons();
}
function spark(key, color, min, max) {
  const canvas = $(key + "Chart"),
    ctx = canvas.getContext("2d"),
    values = buffers[key];
  const w = canvas.width,
    h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#e2e9e7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h - 4);
  ctx.lineTo(w, h - 4);
  ctx.stroke();
  if (!values.length) return;
  const y = (value) =>
    h -
    5 -
    ((Math.max(min, Math.min(max, value)) - min) / (max - min)) * (h - 10);
  ctx.beginPath();
  values.forEach((value, i) => {
    const x = (i / Math.max(values.length - 1, 1)) * w;
    i ? ctx.lineTo(x, y(value)) : ctx.moveTo(x, y(value));
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}
function renderSelected(a) {
  text("selectedId", a.id);
  text("selectedZone", `${floorName[a.floor]} / ASSET DETAILS`);
  text("stageBadge", stageName[a.phm.stage]);
  $("stageBadge").className = `stage ${a.phm.stage}`;
  text("diagnosisSource", sourceName[a.source]);
  text(
    "diagnosis",
    a.label === "정상" && a.phm.stage !== "normal" ? "열화 징후 감지" : a.label,
  );
  text("action", a.phm.action);
  text("selectedHealth", `${a.health} / 100`);
  text("risk", a.phm.risk_score);
  $("healthBar").style.width = `${a.health}%`;
  $("healthBar").style.background =
    a.health < 30
      ? "var(--red)"
      : a.health < 65
        ? "var(--orange)"
        : "var(--green)";
  text(
    "confidenceLabel",
    a.source === "edge_ingest" ? "Edge 보고 신뢰도" : "모델 신뢰도",
  );
  text(
    "confidence",
    a.source === "demo_scenario"
      ? "적용 안 함 (수동 상태)"
      : `${(a.conf * 100).toFixed(1)}%`,
  );
  const rul = a.phm.rul_estimate_min;
  text(
    "rul",
    rul === 0 ? "즉시 확인" : rul ? `${rul}분 이내 점검` : "해당 없음",
  );
  text(
    "sensorSource",
    a.source === "edge_ingest" ? "외부 수신값" : "합성 입력",
  );
  if (performance.now() - sampleAt >= 500) {
    for (const key of Object.keys(buffers)) {
      buffers[key].push(a.sensors[key]);
      if (buffers[key].length > 60) buffers[key].shift();
    }
    sampleAt = performance.now();
  }
  for (const key of Object.keys(buffers))
    text(key + "Value", Number(a.sensors[key]).toFixed(key === "batt" ? 0 : 1));
  spark("batt", "#168263", 0, 100);
  spark("vib", "#d47b28", 0, 12);
  spark("temp", "#368897", 20, 80);
  $("reasons").innerHTML = [...new Set([...a.phm.reasons, ...a.cause])]
    .slice(0, 4)
    .map((reason) => `<li>${escape(reason)}</li>`)
    .join("");
  text(
    "priority",
    a.dispatch.priority === "NORMAL" ? "정상 운전" : a.dispatch.priority,
  );
  text("dispatchAction", a.dispatch.operator_action);
  text(
    "scenarioNotice",
    a.edge_input.active
      ? "외부 Edge 입력 우선 반영 중 · 수동 상태는 TTL 만료 후 적용"
      : `현재: ${a.scenario ? "수동 시나리오" : "모델 리플레이"} · AI 평가에서 시나리오 제외`,
  );
  $("csvExport").href = `/api/history?agv=${encodeURIComponent(a.id)}&fmt=csv`;
  renderAssetOrders();
}
function applyState(next) {
  if (state && next.generated_at < state.generated_at) return;
  state = next;
  lastReceived = Date.now();
  text("total", next.kpi.total);
  text("normal", next.agvs.filter((a) => a.phm.stage === "normal").length);
  text(
    "predicted",
    next.agvs.filter((a) => ["watch", "predicted_fault"].includes(a.phm.stage))
      .length,
  );
  text(
    "faults",
    next.agvs.filter((a) => a.phm.stage === "current_fault").length,
  );
  text("health", next.kpi.avg_health);
  text("latency", `${next.inference.last_latency_ms.toFixed(1)} ms`);
  text(
    "sourceLine",
    `AI-Hub 기반 합성 리플레이 · LightGBM 진단 · Edge ${next.data_source.edge_active}대`,
  );
  const overrides = Object.keys(next.demo.overrides).length;
  text(
    "streamSource",
    overrides
      ? "SCRIPTED SCENARIO ACTIVE"
      : next.data_source.edge_active
        ? "EDGE INPUT ACTIVE"
        : "REPLAY + LIVE MODEL",
  );
  text("scenarioCount", `수동 시나리오 ${overrides}대`);
  text(
    "motionState",
    next.demo.paused ? "주행 일시정지" : `${next.demo.speed}× 재생`,
  );
  if (document.activeElement !== $("speedSelect"))
    $("speedSelect").value = String(next.demo.speed);
  const paused = String(next.demo.paused);
  if ($("pauseBtn").dataset.paused !== paused) {
    $("pauseBtn").dataset.paused = paused;
    $("pauseBtn").innerHTML =
      `<i data-lucide="${next.demo.paused ? "play" : "pause"}"></i>`;
    const label = next.demo.paused ? "주행 재생" : "주행 일시정지";
    $("pauseBtn").title = label;
    $("pauseBtn").setAttribute("aria-label", label);
    icons();
  }
  scene?.updateScene(next.agvs);
  renderAssets();
  const asset = next.agvs.find((a) => a.id === selected);
  if (asset) renderSelected(asset);
  $("sceneError").hidden = !!scene;
  connectionStatus();
}
function connectionStatus() {
  const age = Date.now() - lastReceived,
    stale = age > 5000;
  text(
    "connectionText",
    stale
      ? "연결 확인 중"
      : ws?.readyState === WebSocket.OPEN
        ? "실시간 연결"
        : "폴링 연결",
  );
  $("connectionDot").style.background = stale
    ? "var(--orange)"
    : "var(--green)";
  text(
    "freshness",
    lastReceived ? `${Math.floor(age / 1000)}초 전 갱신` : "갱신 대기",
  );
  text("clock", new Date().toLocaleTimeString("en-GB", { hour12: false }));
  $("applyScenario").disabled = stale || busy;
}
function connect() {
  ws = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
  );
  ws.onopen = () => {
    retryDelay = 1000;
    connectionStatus();
  };
  ws.onmessage = (event) => {
    try {
      applyState(JSON.parse(event.data));
    } catch (e) {
      console.error(e);
    }
  };
  ws.onerror = () => ws.close();
  ws.onclose = () => {
    connectionStatus();
    setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 15000);
  };
}
async function control(payload) {
  if (busy) return;
  busy = true;
  connectionStatus();
  try {
    await api("/api/demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    applyState(await api("/api/snapshot"));
    if (payload.scenario) {
      toast(`${payload.asset_id} 시연 상태 적용됨`);
      await pollHistory();
    }
  } catch (error) {
    toast(error.message);
  } finally {
    busy = false;
    connectionStatus();
    pollOperations();
  }
}
function renderAssetOrders() {
  const rows = orders.filter(
    (o) => o.agv === selected && openStatuses.includes(o.status),
  );
  $("assetOrders").innerHTML = rows.length
    ? rows
        .map(
          (o) =>
            `<div class="asset-order"><strong>${escape(o.title)}</strong><span>${escape(o.priority)} · ${statusName[o.status]} · ${o.overdue ? "SLA 초과" : `기한 ${Math.max(0, Math.ceil(o.time_to_due_sec / 60))}분`}</span></div>`,
        )
        .join("")
    : '<p class="footnote">진행 중인 작업 없음</p>';
}
function renderOrders() {
  const filter = $("orderFilter").value;
  const rows = orders.filter(
    (o) =>
      filter === "all" ||
      (filter === "active"
        ? openStatuses.includes(o.status)
        : ["resolved", "closed"].includes(o.status)),
  );
  const signature = JSON.stringify([
    filter,
    rows.map((o) => [
      o.id,
      o.status,
      o.priority,
      o.health,
      o.overdue,
      Math.ceil(o.time_to_due_sec / 60),
    ]),
  ]);
  if (signature === ordersSignature) return;
  ordersSignature = signature;
  if (!rows.length) {
    $("ordersTable").innerHTML =
      '<p class="empty">해당 상태의 정비 작업이 없습니다.</p>';
    return;
  }
  $("ordersTable").innerHTML =
    `<table><thead><tr><th>자산 / 작업</th><th>우선순위</th><th>상태</th><th>점검 기한</th><th>처리</th></tr></thead><tbody>${rows
      .map((o) => {
        const action = {
          open: ["acknowledged", "접수"],
          acknowledged: ["in_progress", "점검 시작"],
          in_progress: ["resolved", "조치 완료"],
          resolved: ["closed", "종결"],
        }[o.status];
        return `<tr><td><button class="text-button" data-select="${escape(o.agv)}">${escape(o.title)}</button><small>${escape(sourceName[o.source] || o.source)} · ${escape(floorName[o.floor])}</small></td><td><span class="stage ${o.priority === "P1" ? "fault" : "predicted"}">${escape(o.priority)}</span></td><td>${statusName[o.status]}</td><td>${openStatuses.includes(o.status) ? (o.overdue ? "기한 초과" : `${Math.max(0, Math.ceil(o.time_to_due_sec / 60))}분 남음`) : "완료"}</td><td>${action ? `<button data-order="${escape(o.id)}" data-status="${action[0]}">${action[1]}</button>` : "종결됨"}</td></tr>`;
      })
      .join("")}</tbody></table>`;
}
let operationsBusy = false;
async function pollOperations() {
  if (operationsBusy || busy) return;
  operationsBusy = true;
  try {
    const result = await api("/api/work-orders?limit=200");
    orders = result.orders;
    text(
      "orderCount",
      Object.values(result.summary.open_by_priority).reduce((a, b) => a + b, 0),
    );
    renderAssetOrders();
    if (view === "orders") renderOrders();
    const stats = await api("/api/stats");
    text("storageStatus", `이벤트 ${stats.total.toLocaleString()}건`);
  } catch (error) {
    text("storageStatus", "운영 기록 연결 확인 필요");
  } finally {
    operationsBusy = false;
  }
}
async function pollHistory() {
  const id = selected;
  if (!id) return;
  try {
    const rows = await api(
      `/api/history?agv=${encodeURIComponent(id)}&limit=6`,
    );
    if (id !== selected) return;
    $("historyList").innerHTML = rows.length
      ? rows
          .map(
            (row) =>
              `<div class="history-row"><time>${new Date(row.ts * 1000).toLocaleTimeString("en-GB", { hour12: false })}</time><span>${escape(row.pred)} · 건전도 ${row.health}<small>${escape(sourceName[row.source] || row.source)}</small></span></div>`,
          )
          .join("")
      : '<p class="footnote">저장된 이상 이벤트 없음</p>';
  } catch (error) {
    if (id === selected) text("historyList", "이력을 불러올 수 없습니다.");
  }
}
async function loadEvidence() {
  $("evidenceContent").innerHTML =
    '<p class="empty">검증 정보를 불러오는 중</p>';
  try {
    const [card, source, rul] = await Promise.all([
      api("/api/model-card"),
      api("/api/data-source"),
      api("/api/rul-contract"),
    ]);
    const val = card.performance.official_validation;
    $("evidenceContent").innerHTML =
      `<section class="evidence-section"><h3>고장 진단 · LightGBM</h3><div class="evidence-grid"><div><span>검증 정확도</span><strong>${(val.accuracy * 100).toFixed(2)}<small>%</small></strong><span>Official validation split</span></div><div><span>Macro F1</span><strong>${val.macro_f1.toFixed(4)}</strong><span>희소 고장 클래스 개선 필요</span></div><div><span>모델 입력</span><strong>${card.feature_engineering.n_features}</strong><span>30-step 센서 window · 9-class</span></div></div><p>정확도는 저장된 공식 검증 결과입니다. 시연의 합성 센서 입력에 대한 현장 성능을 의미하지 않습니다. 정상 클래스 비중이 높아 Macro F1과 클래스별 성능을 함께 확인해야 합니다.</p></section>
    <section class="evidence-section"><h3>데이터 출처와 처리 범위</h3><table><thead><tr><th>구간</th><th>현재 입력 / 처리</th><th>상태</th></tr></thead><tbody><tr><td>학습 데이터</td><td>AI-Hub 실내공간 유지관리 서비스 로봇</td><td>원본 재배포 제외</td></tr><tr><td>3D 주행 · 센서</td><td>결정론적 주행 + 합성 센서 window</td><td>시뮬레이션</td></tr><tr><td>진단</td><td>LightGBM 실시간 추론 / 외부 보고 / 수동 상태</td><td>자산별 출처 표시</td></tr><tr><td>외부 입력</td><td>Edge API · MQTT → WebSocket</td><td>${source.edge_active}대 반영 중</td></tr><tr><td>물리 로봇</td><td>센서 어댑터 구현, 현장 연결 검증 필요</td><td>미연결</td></tr></tbody></table></section>
    <section class="evidence-section"><h3>PHM · 잔여수명 검증</h3><p>현재 위험도와 점검 시점은 건전도·추세·센서 임계값으로 계산하는 규칙 기반 추정입니다. 실제 고장 시각 라벨로 보정된 잔여수명 예측은 아직 검증되지 않았습니다.</p><p>RUL 학습은 자산별 holdout으로 분리합니다. 동일 자산의 고장 구간이 학습과 평가에 섞이지 않으며, 수동 시나리오 기록은 학습 입력에서 제외됩니다.</p><div class="evidence-grid"><div><span>실패 시각 라벨</span><strong>${rul.readiness.failure_events}</strong><span>독립 고장 사건</span></div><div><span>RUL 런타임</span><strong style="font-size:21px">규칙 기반</strong><span>현장 보정 필요</span></div><div><span>오프라인 학습 메타데이터</span><strong style="font-size:21px">${rul.trained_artifact?.available ? "있음" : "미등록"}</strong><span>운영 모델 자동 전환 없음</span></div></div></section>
    <section class="evidence-section"><h3>검증 자료</h3><div class="evidence-links"><a href="/api/model-card" target="_blank" rel="noopener">모델 카드</a><a href="/api/data-source" target="_blank" rel="noopener">데이터 출처</a><a href="/api/rul-contract" target="_blank" rel="noopener">RUL 계약</a><a href="/api/ops-report?fmt=md" download="operations.md">운영 리포트</a><a href="/api/tsdb-export?fmt=influx" download="telemetry.lp">시계열 내보내기</a><a href="/assets/twin_3d.gif" target="_blank" rel="noopener">기존 3D 시연</a></div></section>`;
  } catch (error) {
    $("evidenceContent").innerHTML =
      `<p class="empty">${escape(error.message)}</p>`;
  }
}

document
  .querySelectorAll("[data-view]")
  .forEach((button) => (button.onclick = () => setView(button.dataset.view)));
for (const id of ["assetSearch", "floorFilter", "stateFilter"])
  $(id).addEventListener("input", renderAssets);
$("closeSelection").onclick = () => select(null);
$("homeBtn").onclick = () => {
  select(null);
  scene?.home();
};
$("orbitBtn").onclick = () => {
  const enabled = !scene?.getOrbit();
  scene?.setOrbit(enabled);
  $("orbitBtn").classList.toggle("active", enabled);
  $("orbitBtn").setAttribute("aria-pressed", String(enabled));
};
$("focusBtn").onclick = () => {
  const first = [...(state?.agvs || [])].sort(
    (a, b) => b.phm.risk_score - a.phm.risk_score,
  )[0];
  if (first) select(first.id);
};
$("fullscreenBtn").onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch {
    toast("이 브라우저에서는 전체 화면을 사용할 수 없습니다.");
  }
};
$("pauseBtn").onclick = () => state && control({ paused: !state.demo.paused });
$("speedSelect").onchange = () =>
  control({ speed: Number($("speedSelect").value) });
$("applyScenario").onclick = () =>
  selected &&
  control({ asset_id: selected, scenario: $("scenarioSelect").value });
$("openOrdersBtn").onclick = () => setView("orders");
$("orderFilter").onchange = renderOrders;
$("ordersTable").addEventListener("click", async (event) => {
  const assetButton = event.target.closest("[data-select]");
  if (assetButton) {
    select(assetButton.dataset.select);
    return;
  }
  const button = event.target.closest("[data-order]");
  if (!button || busy) return;
  busy = true;
  button.disabled = true;
  try {
    await api(
      `/api/work-orders/${encodeURIComponent(button.dataset.order)}/status?status=${button.dataset.status}`,
      { method: "POST" },
    );
    toast(`작업 상태: ${statusName[button.dataset.status]}`);
    ordersSignature = "";
  } catch (error) {
    toast(error.message);
  } finally {
    busy = false;
    button.disabled = false;
    pollOperations();
  }
});
$("retryBtn").onclick = () => location.reload();
addEventListener("keydown", (event) => {
  if (event.key === "Escape") select(null);
});

async function start() {
  icons();
  try {
    scene = await import("./twin-scene.js");
    const layout = await api("/api/layout");
    scene.initLayout(layout, select);
    $("orbitBtn").classList.toggle("active", scene.getOrbit());
    $("orbitBtn").setAttribute("aria-pressed", String(scene.getOrbit()));
  } catch (error) {
    scene = null;
    $("sceneError").hidden = false;
    text("sceneErrorText", error.message);
  }
  try {
    applyState(await api("/api/snapshot"));
  } catch (error) {
    $("sceneError").hidden = false;
    text("sceneErrorText", error.message);
  }
  connect();
  pollOperations();
}
setInterval(connectionStatus, 1000);
setInterval(async () => {
  if (Date.now() - lastReceived < 2000) return;
  try {
    applyState(await api("/api/snapshot"));
  } catch {
    connectionStatus();
  }
}, 2000);
setInterval(pollOperations, 5000);
setInterval(pollHistory, 3000);
start();
