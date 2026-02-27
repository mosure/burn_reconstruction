"use strict";

/**
 * burn_gaussian_splatting model/web-asset service worker.
 *
 * Source-of-truth location:
 * - `www/burn_gaussian_splatting_sw.js`
 *
 * Deployment location:
 * - copied to site root as `burn_gaussian_splatting_sw.js` so root pages are in scope.
 */

const CACHE_PREFIX = "burn-gaussian-splatting-web";
const CACHE_VERSION = "v1";
const CACHE_NAME = `${CACHE_PREFIX}-${CACHE_VERSION}`;
const ABERRATION_MODEL_ORIGIN = "https://aberration.technology";

function isCacheableModelPath(pathname) {
  return (
    pathname.endsWith(".bpk") ||
    pathname.endsWith(".bpk.parts.json") ||
    pathname.includes(".bpk.part-") ||
    pathname.endsWith(".json")
  );
}

function isCacheableWebAsset(pathname) {
  return (
    pathname.endsWith(".wasm") ||
    pathname.endsWith(".js") ||
    pathname.endsWith(".mjs") ||
    pathname.endsWith(".css") ||
    pathname.endsWith(".html")
  );
}

function shouldCacheRequest(requestUrl) {
  const url = new URL(requestUrl);
  const pathname = url.pathname;

  if (url.origin === self.location.origin) {
    if (isCacheableWebAsset(pathname)) {
      return true;
    }
    if (
      (pathname.includes("/assets/models/") ||
        pathname.includes("/www/assets/models/") ||
        pathname.includes("/assets/")) &&
      isCacheableModelPath(pathname)
    ) {
      return true;
    }
    return false;
  }

  return (
    url.origin === ABERRATION_MODEL_ORIGIN &&
    pathname.startsWith("/model/") &&
    isCacheableModelPath(pathname)
  );
}

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => key.startsWith(`${CACHE_PREFIX}-`) && key !== CACHE_NAME)
          .map((key) => caches.delete(key)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") {
    return;
  }
  if (!shouldCacheRequest(request.url)) {
    return;
  }

  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);
      if (cached) {
        return cached;
      }

      const response = await fetch(request);
      if (response && (response.ok || response.type === "opaque")) {
        await cache.put(request, response.clone());
      }
      return response;
    })(),
  );
});
