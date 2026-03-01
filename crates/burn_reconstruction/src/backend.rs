use burn::prelude::Backend;
#[cfg(target_arch = "wasm32")]
use std::cell::Cell;

/// Canonical runtime backend for this crate.
///
/// `burn_reconstruction` intentionally standardizes on WGPU + fusion to
/// keep one validated, production inference path.
pub type BackendImpl = burn::backend::Wgpu<f32, i32>;

/// Convenience alias for the default backend device.
pub type BackendDevice = <BackendImpl as Backend>::Device;

/// Returns the default WGPU device used by the public pipeline APIs.
pub fn default_device() -> BackendDevice {
    BackendDevice::default()
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WASM_WGPU_RUNTIME_READY: Cell<bool> = const { Cell::new(false) };
}

/// Ensures the wasm WebGPU runtime is initialized asynchronously before any sync tensor ops.
///
/// This avoids lazy sync runtime initialization paths that can panic on wasm.
#[cfg(target_arch = "wasm32")]
pub async fn ensure_wasm_wgpu_runtime(device: &BackendDevice) {
    if WASM_WGPU_RUNTIME_READY.with(Cell::get) {
        return;
    }
    burn::backend::wgpu::init_setup_async::<burn::backend::wgpu::graphics::WebGpu>(
        device,
        burn::backend::wgpu::RuntimeOptions::default(),
    )
    .await;
    WASM_WGPU_RUNTIME_READY.with(|flag| flag.set(true));
}
