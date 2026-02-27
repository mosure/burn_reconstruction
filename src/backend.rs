#[cfg(feature = "backend_cuda")]
pub type BackendImpl = burn::backend::Cuda<f32, i32>;

#[cfg(all(not(feature = "backend_cuda"), feature = "backend_wgpu"))]
pub type BackendImpl = burn::backend::Wgpu<f32, i32>;

#[cfg(all(
    not(feature = "backend_cuda"),
    not(feature = "backend_wgpu"),
    feature = "backend_cpu"
))]
pub type BackendImpl = burn::backend::Cpu<f32, i32>;

#[cfg(all(
    not(feature = "backend_cuda"),
    not(feature = "backend_wgpu"),
    not(feature = "backend_cpu"),
    feature = "backend_ndarray"
))]
pub type BackendImpl = burn::backend::NdArray<f32>;

#[cfg(all(
    not(feature = "backend_cuda"),
    not(feature = "backend_wgpu"),
    not(feature = "backend_cpu"),
    not(feature = "backend_ndarray")
))]
compile_error!(
    "No backend feature enabled. Enable one of: backend_cuda, backend_wgpu, backend_cpu, backend_ndarray.",
);
