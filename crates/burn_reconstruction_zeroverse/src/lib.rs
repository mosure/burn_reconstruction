#![forbid(unsafe_code)]

//! Stub crate for future YoNo fine-tuning using:
//! - `bevy_zeroverse` as synthetic 3D scene source
//! - `brush` as differentiable gaussian renderer / loss bridge
//!
//! The current implementation provides a stable orchestration API surface and
//! tests, without introducing hard runtime dependencies on either integration.

pub mod adapters;
pub mod config;
pub mod error;
pub mod workflow;

pub use burn_yono::{YonoWeightFormat, YonoWeightPrecision, YonoWeights};
pub use config::{BrushRenderConfig, YonoFineTuneConfig, ZeroverseDataConfig};
pub use error::ZeroverseError;
pub use workflow::{
    BrushDifferentiableRenderer, EpochMetrics, LossValue, StepMetrics, SyntheticBatch, YonoTrainer,
    ZeroverseFineTuneOrchestrator, ZeroverseSceneProvider,
};

/// High-level project goal of this crate.
pub const CRATE_GOAL: &str =
    "Train/fine-tune YoNo with bevy_zeroverse synthetic scenes + brush differentiable gaussian loss.";
