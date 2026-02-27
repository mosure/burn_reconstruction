use burn::prelude::Backend;

/// Canonical runtime backend for this crate.
///
/// `burn_gaussian_splatting` intentionally standardizes on WGPU + fusion to
/// keep one validated, production inference path.
pub type BackendImpl = burn::backend::Wgpu<f32, i32>;

/// Convenience alias for the default backend device.
pub type BackendDevice = <BackendImpl as Backend>::Device;

/// Returns the default WGPU device used by the public pipeline APIs.
pub fn default_device() -> BackendDevice {
    BackendDevice::default()
}
