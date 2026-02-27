#![allow(clippy::too_many_arguments)]

pub mod backend;
pub mod pipeline;
pub mod utils;

pub use utils::setup_hooks;

#[cfg(feature = "yono")]
pub use burn_yono;

#[cfg(all(feature = "yono", feature = "correctness"))]
pub use burn_yono::correctness;

#[cfg(feature = "yono")]
pub use burn_yono::model;

#[cfg(all(feature = "yono", feature = "import"))]
pub use burn_yono::import;
