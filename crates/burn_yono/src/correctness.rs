use std::{fmt, fs, path::Path};

use burn::{prelude::*, tensor::Tensor};
use safetensors::tensor::{SafeTensors, TensorView};

#[derive(Debug, Clone)]
pub struct MetricStats {
    pub mean_abs: f32,
    pub max_abs: f32,
    pub max_rel: f32,
    pub mse: f32,
}

impl MetricStats {
    pub fn within(&self, mean_abs: f32, max_abs: f32, mse: f32) -> bool {
        self.mean_abs <= mean_abs && self.max_abs <= max_abs && self.mse <= mse
    }
}

pub fn compute_stats(lhs: &[f32], rhs: &[f32]) -> Result<MetricStats, CorrectnessError> {
    if lhs.len() != rhs.len() {
        return Err(CorrectnessError::LengthMismatch {
            expected: rhs.len(),
            actual: lhs.len(),
        });
    }

    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mse = 0.0f32;

    for (&l, &r) in lhs.iter().zip(rhs.iter()) {
        let diff = l - r;
        let abs = diff.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
        if r.abs() > f32::EPSILON {
            max_rel = max_rel.max(abs / r.abs());
        }
        mse += diff * diff;
    }

    let n = lhs.len() as f32;
    Ok(MetricStats {
        mean_abs: sum_abs / n,
        max_abs,
        max_rel,
        mse: mse / n,
    })
}

pub fn load_safetensors(path: impl AsRef<Path>) -> Result<SafeTensors<'static>, CorrectnessError> {
    let bytes = fs::read(path)?;
    let leaked = Box::leak(bytes.into_boxed_slice());
    SafeTensors::deserialize(leaked).map_err(CorrectnessError::Safetensors)
}

pub fn read_tensor(
    tensors: &SafeTensors<'_>,
    name: &'static str,
) -> Result<(Vec<f32>, Vec<usize>), CorrectnessError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| CorrectnessError::MissingTensor(name))?;
    Ok((tensor_view_to_vec(&view), view.shape().to_vec()))
}

pub fn tensor_view_to_vec(view: &TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk
                .try_into()
                .expect("safetensors view must encode f32 in 4-byte chunks");
            f32::from_le_bytes(bytes)
        })
        .collect()
}

pub fn tensor_to_vec<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Result<Vec<f32>, CorrectnessError> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| CorrectnessError::TensorData(format!("{err:?}")))
}

#[derive(Debug)]
pub enum CorrectnessError {
    Io(std::io::Error),
    Safetensors(safetensors::SafeTensorError),
    MissingTensor(&'static str),
    LengthMismatch { expected: usize, actual: usize },
    TensorData(String),
}

impl fmt::Display for CorrectnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Safetensors(err) => write!(f, "safetensors error: {err}"),
            Self::MissingTensor(name) => write!(f, "tensor `{name}` missing from reference"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "tensor length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::TensorData(err) => write!(f, "tensor data conversion failed: {err}"),
        }
    }
}

impl std::error::Error for CorrectnessError {}

impl From<std::io::Error> for CorrectnessError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
