#![recursion_limit = "1024"]
#![allow(clippy::too_many_arguments)]

pub mod backend;
pub mod bootstrap;
pub mod pipeline;
pub mod utils;
#[cfg(target_arch = "wasm32")]
pub mod wasm_api;

pub use bootstrap::{
    default_cache_root, default_cache_root_with_config, resolve_or_bootstrap_yono_weights,
    resolve_or_bootstrap_yono_weights_with_config, ModelBootstrapError, YonoBootstrapConfig,
};
pub use burn_yono::{
    ForwardTimings, GlbExportOptions, GlbExportReport, GlbSortMode, YonoWeightFormat, YonoWeights,
};
pub use pipeline::{
    ComponentLoadReport, ImageToGaussianPipeline, PipelineConfig, PipelineError,
    PipelineExportReport, PipelineGaussians, PipelineLoadReport, PipelineModel, PipelineQuality,
    PipelineRunOutput, PipelineWeights,
};
pub use utils::setup_hooks;

#[cfg(feature = "correctness")]
pub use burn_yono::correctness;

/// Optional access to lower-level YoNoSplat internals.
///
/// This is intentionally feature-gated so the default API surface remains concise.
#[cfg(feature = "raw_yono")]
pub use burn_yono;
