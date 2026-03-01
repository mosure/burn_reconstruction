#![recursion_limit = "1024"]
#![allow(clippy::too_many_arguments)]

pub mod backend;
pub mod bootstrap;
pub mod build_info;
pub mod pipeline;
pub mod utils;
#[cfg(target_arch = "wasm32")]
pub mod wasm_api;

pub use bootstrap::{
    default_cache_root, default_cache_root_with_config, resolve_or_bootstrap_yono_weights,
    resolve_or_bootstrap_yono_weights_with_config,
    resolve_or_bootstrap_yono_weights_with_config_and_progress,
    resolve_or_bootstrap_yono_weights_with_precision,
    resolve_or_bootstrap_yono_weights_with_precision_and_progress, BootstrapProgressCallback,
    ModelBootstrapError, YonoBootstrapConfig,
};
pub use build_info::{app_banner, build_label, git_revision_short, long_version};
#[cfg(target_arch = "wasm32")]
pub use burn_yono::glb::pack_gaussian_rows_full_async;
pub use burn_yono::{
    glb::{
        canonicalize_gaussian_transform_cv, cv_xyzw_to_canonical_wxyz, pack_gaussian_rows_full,
        sanitize_scale_for_viewer, CanonicalGaussianTransform, ExportGaussians, GlbExportOptions,
        GlbExportReport, GlbSortMode, PackedGaussianRows,
    },
    ForwardTimings, YonoWeightFormat, YonoWeightPrecision, YonoWeights,
};
pub use pipeline::{
    ComponentLoadReport, ImageToGaussianPipeline, PipelineConfig, PipelineError,
    PipelineExportReport, PipelineGaussians, PipelineInputImage, PipelineLoadReport, PipelineModel,
    PipelineQuality, PipelineRunOutput, PipelineRunWithCameras, PipelineWeights,
};
pub use utils::setup_hooks;

#[cfg(feature = "correctness")]
pub use burn_yono::correctness;

/// Optional access to lower-level YoNoSplat internals.
///
/// This is intentionally feature-gated so the default API surface remains concise.
#[cfg(feature = "raw_yono")]
pub use burn_yono;
