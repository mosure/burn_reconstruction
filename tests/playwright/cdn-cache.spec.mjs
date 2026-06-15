import { expect, test } from "@playwright/test";

const CDN_CACHE_E2E = process.env.BURN_RECONSTRUCTION_CDN_CACHE_E2E === "1";

const CDN_URLS = [
  "https://aberration.technology/model/zipsplat/zipsplat_f16.bpk.parts.json",
  "https://aberration.technology/model/zipsplat/zipsplat_f16.bpk.part-00000.bpk",
  "https://aberration.technology/model/yono/yono_backbone_f16.bpk.parts.json",
];

test.skip(!CDN_CACHE_E2E, "set BURN_RECONSTRUCTION_CDN_CACHE_E2E=1 for live CDN cache validation");

test("service worker caches live CDN model shards and serves them offline", async ({
  context,
  page,
}) => {
  await page.goto("/infer.html", { waitUntil: "domcontentloaded" });
  await page.evaluate(async () => {
    for (const reg of await navigator.serviceWorker.getRegistrations()) {
      await reg.unregister();
    }
    for (const key of await caches.keys()) {
      await caches.delete(key);
    }
  });

  await page.reload({ waitUntil: "domcontentloaded" });
  await page.evaluate(async () => {
    await navigator.serviceWorker.register(`/burn_reconstruction_sw.js?cache-test=${Date.now()}`, {
      scope: "/",
    });
    await navigator.serviceWorker.ready;
    if (!navigator.serviceWorker.controller) {
      await new Promise((resolve) => {
        navigator.serviceWorker.addEventListener("controllerchange", resolve, { once: true });
        setTimeout(resolve, 2000);
      });
    }
  });
  if (!(await page.evaluate(() => Boolean(navigator.serviceWorker.controller)))) {
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.evaluate(() => navigator.serviceWorker.ready);
  }
  await expect
    .poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller)))
    .toBe(true);

  const fetchResults = await page.evaluate(async (urls) => {
    const out = [];
    for (const url of urls) {
      const response = await fetch(url);
      const bytes = (await response.arrayBuffer()).byteLength;
      out.push({ url, status: response.status, type: response.type, bytes });
    }
    return out;
  }, CDN_URLS);
  for (const result of fetchResults) {
    expect(result.status, result.url).toBe(200);
    expect(result.bytes, result.url).toBeGreaterThan(0);
  }

  const cacheResults = await page.evaluate(async (urls) => {
    const keys = await caches.keys();
    const out = [];
    for (const key of keys) {
      const cache = await caches.open(key);
      for (const url of urls) {
        const match = await cache.match(url);
        if (match) {
          out.push({
            key,
            url,
            status: match.status,
            contentLength: match.headers.get("content-length"),
          });
        }
      }
    }
    return { keys, out };
  }, CDN_URLS);
  expect(cacheResults.out).toHaveLength(CDN_URLS.length);

  await context.setOffline(true);
  const offlineResults = await page.evaluate(async (urls) => {
    const out = [];
    for (const url of urls) {
      const response = await fetch(url);
      const bytes = (await response.arrayBuffer()).byteLength;
      out.push({ url, status: response.status, bytes });
    }
    return out;
  }, CDN_URLS);
  for (const result of offlineResults) {
    expect(result.status, result.url).toBe(200);
    expect(result.bytes, result.url).toBeGreaterThan(0);
  }
  await context.setOffline(false);
});
