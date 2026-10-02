// Capture actual browser frames, then encode with ffmpeg (see docs/DEMO_CAPTURE_CHECKLIST.md).
import { createRequire } from "node:module";
import fs from "node:fs/promises";
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const url = process.env.TWIN_URL || "http://127.0.0.1:8765";
const output = process.env.TWIN_CAPTURE_DIR || "/tmp/servicerobot-capture";
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  channel: process.env.CHROME_CHANNEL || "chrome",
});
const page = await browser.newPage({
  viewport: { width: 1440, height: 960 },
  deviceScaleFactor: 1,
});
const request = page.request;
const initial = await (await request.get(url + "/api/demo")).json();
let frame = 0;
async function capture(seconds) {
  for (let i = 0; i < seconds * 5; i++) {
    const start = Date.now();
    await page.screenshot({
      path: `${output}/frame-${String(frame++).padStart(4, "0")}.png`,
    });
    await page.waitForTimeout(Math.max(1, 200 - (Date.now() - start)));
  }
}
async function control(body) {
  const response = await request.post(url + "/api/demo", { data: body });
  if (!response.ok()) throw new Error(await response.text());
}
try {
  await page.goto(url + "/twin");
  await page.waitForFunction(
    () => document.querySelector("#total").textContent !== "--",
  );
  await page.waitForTimeout(1500);
  await capture(4);
  await page.locator('[data-asset="AGV-01"]').click();
  for (const scenario of ["normal", "watch", "degrading", "battery"]) {
    await page.locator("#scenarioSelect").selectOption(scenario);
    await page.locator("#applyScenario").click();
    await page.waitForTimeout(350);
    await page
      .locator(".inspector-scroll")
      .evaluate((el) => (el.scrollTop = 0));
    await capture(4);
  }
  await page.locator('[data-view="orders"]').click();
  await capture(3);
  const row = page
    .locator("#ordersTable tbody tr")
    .filter({ hasText: "AGV-01 배터리 저하" })
    .first();
  for (const label of ["접수", "점검 시작", "조치 완료"]) {
    await row.getByRole("button", { name: label, exact: true }).click();
    await capture(2);
  }
  await page.locator("#orderFilter").selectOption("resolved");
  await capture(2);
  await page.locator('[data-view="evidence"]').click();
  await page.waitForFunction(() =>
    document.querySelector("#evidenceContent").textContent.includes("Macro F1"),
  );
  await capture(4);
  await page.locator('[data-view="twin"]').click();
  await page.locator("#homeBtn").click();
  await control({ asset_id: "AGV-01", scenario: "replay" });
  await capture(3);
  console.log(
    JSON.stringify({
      frames: frame,
      fps: 5,
      durationSeconds: frame / 5,
      directory: output,
    }),
  );
} finally {
  await control({
    asset_id: "AGV-01",
    scenario: initial.overrides["AGV-01"] || "replay",
  });
  await browser.close();
}
