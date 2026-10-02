// Run against an isolated demo server: TWIN_URL=http://127.0.0.1:8765 node scripts/verify_twin.mjs
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs/promises";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const url = process.env.TWIN_URL || "http://127.0.0.1:8765";
const output = process.env.TWIN_SCREENSHOTS || "/tmp/servicerobot-ui";
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  channel: process.env.CHROME_CHANNEL || "chrome",
});
const errors = [];
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  deviceScaleFactor: 1,
});
const page = await context.newPage();
page.on("pageerror", (error) => errors.push(error.message));
const request = context.request;
const initial = await (await request.get(url + "/api/demo")).json();
async function control(body) {
  const response = await request.post(url + "/api/demo", { data: body });
  assert(response.ok(), await response.text());
}
async function waitForText(selector, value) {
  await page.waitForFunction(
    ([selector, value]) =>
      document.querySelector(selector)?.textContent.includes(value),
    [selector, value],
  );
}
async function canvasStats() {
  return page.locator("#app canvas").evaluate((canvas) => {
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(
      0,
      0,
      canvas.width,
      canvas.height,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
    let varied = 0,
      hash = 0;
    for (let i = 0; i < pixels.length; i += 32) {
      if (Math.min(pixels[i], pixels[i + 1], pixels[i + 2]) < 210) varied++;
      hash = (hash * 31 + pixels[i] + pixels[i + 1]) | 0;
    }
    return { varied, hash, width: canvas.width, height: canvas.height };
  });
}
try {
  await page.goto(url + "/twin");
  await page.waitForFunction(
    () => document.querySelector("#total").textContent !== "--",
  );
  await page.waitForSelector("#app canvas");
  await page.waitForTimeout(1500);
  let before = await canvasStats();
  assert(before.varied > 500, "blank 3D canvas");
  await page.screenshot({ path: output + "/desktop.png", fullPage: true });
  await page.waitForTimeout(1200);
  assert.notEqual(
    (await canvasStats()).hash,
    before.hash,
    "3D scene is not moving",
  );
  await page.locator('[data-asset="AGV-01"]').click();
  await waitForText("#selectedId", "AGV-01");
  for (const [scenario, stage] of [
    ["normal", "정상"],
    ["watch", "주의"],
    ["degrading", "예측 이상"],
    ["battery", "현재 이상"],
  ]) {
    await page.locator("#scenarioSelect").selectOption(scenario);
    await page.locator("#applyScenario").click();
    await waitForText("#stageBadge", stage);
  }
  await waitForText("#confidence", "적용 안 함");
  await page.locator(".inspector-scroll").evaluate((el) => (el.scrollTop = 0));
  await page.waitForTimeout(1500);
  await page.screenshot({ path: output + "/diagnosis.png", fullPage: true });
  await page.locator('[data-view="orders"]').click();
  await page
    .locator("#ordersTable tr")
    .filter({ hasText: "AGV-01 배터리 저하" })
    .first()
    .waitFor();
  const order = page
    .locator("#ordersTable tbody tr")
    .filter({ hasText: "AGV-01 배터리 저하" })
    .first();
  for (const label of ["접수", "점검 시작", "조치 완료"]) {
    await order.getByRole("button", { name: label, exact: true }).click();
    await page.waitForTimeout(600);
  }
  await page.locator("#orderFilter").selectOption("resolved");
  await page.waitForFunction(() =>
    document.querySelector("#ordersTable").textContent.includes("조치 완료"),
  );
  await page.screenshot({ path: output + "/orders.png", fullPage: true });
  await page.locator('[data-view="evidence"]').click();
  await waitForText("#evidenceContent", "Macro F1");
  await page.screenshot({ path: output + "/evidence.png", fullPage: true });
  await page.locator('[data-view="twin"]').click();
  await page.locator("#homeBtn").click();
  await page.locator("#assetSearch").fill("NO-SUCH-ASSET");
  await waitForText("#assetList", "조건에 맞는 자산이 없습니다");
  await page.locator("#assetSearch").fill("");
  await page.locator("#floorFilter").selectOption("1");
  assert.equal(await page.locator("[data-asset]").count(), 9);
  await page.locator("#floorFilter").selectOption("all");
  assert.equal(
    await page.locator("[data-asset]").first().getAttribute("data-asset"),
    "AGV-01",
  );
  await control({ paused: true });
  const paused = await (await request.get(url + "/api/snapshot")).json();
  await page.waitForTimeout(500);
  assert.equal(
    (await (await request.get(url + "/api/snapshot")).json()).p,
    paused.p,
    "pause did not stop route progress",
  );
  if ((await page.locator("#orbitBtn").getAttribute("aria-pressed")) === "true")
    await page.locator("#orbitBtn").click();
  // Read back actual WebGL pixels and click a fault-colored mesh, not an HTML proxy.
  await page.waitForTimeout(1800);
  const point = await page.locator("#app canvas").evaluate((canvas) => {
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    const w = canvas.width,
      h = canvas.height,
      p = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, p);
    for (let y = Math.floor(h * 0.25); y < h * 0.8; y++)
      for (let x = Math.floor(w * 0.15); x < w * 0.9; x++) {
        const i = (y * w + x) * 4;
        const red = (offset) =>
          (p[offset] > 110 && p[offset] > p[offset + 1] * 1.3 && p[offset] > p[offset + 2] * 1.2)
          || (p[offset + 1] > 80 && p[offset + 1] > p[offset] * 1.4 && p[offset + 1] > p[offset + 2] * 1.12);
        if (
          red(i) &&
          red(i + 4) &&
          red(i - 4) &&
          red(i + w * 4) &&
          red(i - w * 4)
        )
          return {
            x: (x * canvas.clientWidth) / w,
            y: ((h - 1 - y) * canvas.clientHeight) / h,
          };
      }
    return null;
  });
  assert(point, "no visible status-colored 3D asset");
  await page.locator("#app canvas").click({ position: point });
  await page.waitForFunction(
    () => !document.querySelector("#selection").hidden,
  );
  await page.locator("#closeSelection").click();
  await control({ paused: false });
  for (const [name, width, height] of [
    ["wide", 1920, 1080],
    ["tablet", 1024, 768],
    ["portrait", 768, 1024],
    ["mobile", 390, 844],
    ["small", 360, 800],
  ]) {
    await page.setViewportSize({ width, height });
    await page.locator("#homeBtn").click();
    await page.waitForTimeout(1800);
    assert((await canvasStats()).varied > 200, `${name}: blank canvas`);
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      `${name}: horizontal overflow`,
    );
    await page.screenshot({ path: `${output}/${name}.png`, fullPage: true });
    await page.locator('[data-asset="AGV-01"]').click();
    await waitForText("#selectedId", "AGV-01");
    await page.waitForTimeout(1800);
    await page.screenshot({
      path: `${output}/${name}-selected.png`,
      fullPage: true,
    });
    await page.locator("#closeSelection").click();
    for (const nextView of ["orders", "evidence"]) {
      await page.locator(`[data-view="${nextView}"]`).click();
      if (nextView === "evidence")
        await waitForText("#evidenceContent", "Macro F1");
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        `${name}/${nextView}: horizontal overflow`,
      );
    }
    await page.locator('[data-view="twin"]').click();
  }
  // WebSocket outage must fall back to HTTP polling and preserve operability.
  const fallback = await context.newPage();
  await fallback.routeWebSocket("**/ws", (socket) => socket.close());
  await fallback.goto(url + "/twin");
  await fallback.waitForFunction(
    () => document.querySelector("#connectionText").textContent === "폴링 연결",
  );
  await fallback.close();
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      passed: true,
      viewports: 6,
      canvasChecks: true,
      scenarioStages: 4,
      maintenanceWorkflow: true,
      websocketFallback: true,
      screenshots: output,
    }),
  );
} catch (error) {
  await page.screenshot({path:output+'/failure.png',fullPage:true});
  throw error;
} finally {
  const now = await (await request.get(url + "/api/demo")).json();
  for (const asset_id of Object.keys(now.overrides))
    await control({ asset_id, scenario: "replay" });
  for (const [asset_id, scenario] of Object.entries(initial.overrides))
    await control({ asset_id, scenario });
  await control({ paused: initial.paused, speed: initial.speed });
  await browser.close();
}
