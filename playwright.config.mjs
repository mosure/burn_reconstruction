import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "tests/playwright",
  timeout: 60_000,
  webServer: {
    command: "python3 -m http.server 4174 -d www",
    url: "http://127.0.0.1:4174",
    reuseExistingServer: false,
    timeout: 15_000,
  },
  use: {
    baseURL: "http://127.0.0.1:4174",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
        },
      },
    },
  ],
});
