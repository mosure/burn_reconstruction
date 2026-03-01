use std::borrow::Cow;

/// Errors produced by the Zeroverse/Brush YoNo fine-tuning orchestration layer.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ZeroverseError {
    #[error("invalid training configuration: {0}")]
    InvalidConfig(Cow<'static, str>),
    #[error("integration unavailable: {0}")]
    IntegrationUnavailable(Cow<'static, str>),
    #[error("data pipeline failure: {0}")]
    Data(Cow<'static, str>),
    #[error("render pipeline failure: {0}")]
    Render(Cow<'static, str>),
    #[error("trainer failure: {0}")]
    Trainer(Cow<'static, str>),
}
