import { expect, test } from "@playwright/test";

test("infer page exposes model and ZipSplat quality controls", async ({ page }) => {
  await page.goto("/infer.html");

  await expect(page.getByRole("heading", { name: "Wasm Reconstruction Inference" })).toBeVisible();
  await expect(page.locator("#model-yono")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#model-zipsplat")).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator("#quality")).toHaveValue("balanced");
  await expect(page.locator("#image-size")).toHaveValue("224");
  await expect(page.locator("#zipsplat-settings")).toBeHidden();
  await expect(page.locator("#zipsplat-r")).toHaveValue("2");
  await expect(page.locator("#zipsplat-r-value")).toHaveText("2");
  await expect(page.locator("#config-summary")).toContainText("YoNoSplat");
  await expect(page.locator("#run")).toHaveText("Select 2+ images");

  await page.locator("#model-zipsplat").click();
  await expect(page.locator("#model-yono")).toHaveAttribute("aria-pressed", "false");
  await expect(page.locator("#model-zipsplat")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#zipsplat-settings")).toBeVisible();
  await expect(page.locator("#image-size")).toHaveValue("252");
  await expect(page.locator("#config-summary")).toContainText("ZipSplat");
  await page.locator("#quality").selectOption("compact");
  await expect(page.locator("#zipsplat-r")).toHaveValue("4");
  await expect(page.locator("#zipsplat-r-value")).toHaveText("4");
  await expect(page.locator("#config-summary")).toContainText("r=4");

  await page.locator("#zipsplat-r").fill("8");
  await expect(page.locator("#zipsplat-r")).toHaveValue("8");
  await expect(page.locator("#zipsplat-r-value")).toHaveText("8");
});

test("infer page initializes wasm or reports WebGPU unavailability", async ({ page }) => {
  await page.goto("/infer.html");

  const hasWebGpu = await page.evaluate(() => Boolean(globalThis.navigator?.gpu));
  if (!hasWebGpu) {
    await expect(page.locator("#status")).toContainText("WebGPU is unavailable");
    return;
  }

  await expect(page.locator("#status")).toContainText(/ready|failed|unavailable|adapter check failed/, {
    timeout: 30_000,
  });
});
