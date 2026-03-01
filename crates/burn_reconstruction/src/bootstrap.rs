use std::path::PathBuf;
use std::sync::Arc;

use burn_yono::{YonoWeightFormat, YonoWeightPrecision, YonoWeights};

/// Thread-safe callback used to report native bootstrap progress.
pub type BootstrapProgressCallback = Arc<dyn Fn(String) + Send + Sync + 'static>;

/// Model source and cache settings for YoNo bootstrap.
///
/// This is an explicit configuration surface that does not depend on process
/// environment variables, which makes it suitable for host-managed runtimes.
#[derive(Debug, Clone)]
pub struct YonoBootstrapConfig {
    /// Optional explicit cache directory where YoNo weights should be stored.
    ///
    /// When unset on native targets, the default is:
    /// `~/.burn_reconstruction/models/yono`.
    pub cache_root: Option<PathBuf>,
    /// Base model URL.
    pub model_base_url: String,
    /// Remote directory beneath `model_base_url`.
    pub yono_remote_root: String,
    /// Optional full URL override for backbone weights.
    pub backbone_url: Option<String>,
    /// Optional full URL override for head weights.
    pub head_url: Option<String>,
    /// Prefer `.bpk.parts.json` + `.bpk.part-*` bundles when using burnpack format.
    ///
    /// When true (default), bootstrap first tries to fetch parts manifests and parts and
    /// only falls back to monolithic `.bpk` files when no parts manifest is available.
    pub prefer_burnpack_parts: bool,
    /// Preferred burnpack precision when `format = Burnpack`.
    ///
    /// Defaults to `f16` to reduce transfer size and cache footprint.
    pub burnpack_precision: YonoWeightPrecision,
}

impl Default for YonoBootstrapConfig {
    fn default() -> Self {
        Self {
            cache_root: None,
            model_base_url: "https://aberration.technology/model".to_string(),
            yono_remote_root: "yono".to_string(),
            backbone_url: None,
            head_url: None,
            prefer_burnpack_parts: true,
            burnpack_precision: YonoWeightPrecision::F16,
        }
    }
}

/// Resolves YoNoSplat weights from local cache and bootstraps downloads on first use.
///
/// Native behavior:
/// - cache root: `~/.burn_reconstruction/models/yono`
/// - remote root: `https://aberration.technology/model/yono`
///
/// Environment overrides:
/// - `BURN_RECONSTRUCTION_CACHE_DIR` (absolute cache root)
/// - `BURN_RECONSTRUCTION_MODEL_BASE_URL`
/// - `BURN_RECONSTRUCTION_YONO_REMOTE_ROOT`
/// - `BURN_RECONSTRUCTION_YONO_BACKBONE_URL`
/// - `BURN_RECONSTRUCTION_YONO_HEAD_URL`
/// - `BURN_RECONSTRUCTION_YONO_PREFER_PARTS` (`1|true|yes|on` to enable)
/// - `BURN_RECONSTRUCTION_YONO_BURNPACK_PRECISION` (`f16` or `f32`)
pub fn resolve_or_bootstrap_yono_weights(
    format: YonoWeightFormat,
) -> Result<YonoWeights, ModelBootstrapError> {
    resolve_or_bootstrap_yono_weights_with_precision(format, YonoWeightPrecision::F16)
}

/// Resolves/cache-populates YoNo weights using explicit precision selection.
///
/// For burnpack format, this prefers the requested precision and transparently
/// falls back to the alternate precision when unavailable.
pub fn resolve_or_bootstrap_yono_weights_with_precision(
    format: YonoWeightFormat,
    precision: YonoWeightPrecision,
) -> Result<YonoWeights, ModelBootstrapError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (format, precision);
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut cfg = native::apply_env_overrides(YonoBootstrapConfig::default());
        cfg.burnpack_precision = precision;
        native::resolve_or_bootstrap_yono_weights_native(format, &cfg, None)
    }
}

/// Resolves/cache-populates YoNo weights with precision and native progress callbacks.
pub fn resolve_or_bootstrap_yono_weights_with_precision_and_progress<F>(
    format: YonoWeightFormat,
    precision: YonoWeightPrecision,
    progress: F,
) -> Result<YonoWeights, ModelBootstrapError>
where
    F: Fn(String) + Send + Sync + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (format, precision, progress);
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut cfg = native::apply_env_overrides(YonoBootstrapConfig::default());
        cfg.burnpack_precision = precision;
        native::resolve_or_bootstrap_yono_weights_native(format, &cfg, Some(Arc::new(progress)))
    }
}

/// Resolves/cache-populates YoNo weights using explicit non-env configuration.
pub fn resolve_or_bootstrap_yono_weights_with_config(
    format: YonoWeightFormat,
    cfg: &YonoBootstrapConfig,
) -> Result<YonoWeights, ModelBootstrapError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (format, cfg);
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native::resolve_or_bootstrap_yono_weights_native(format, cfg, None)
    }
}

/// Resolves/cache-populates YoNo weights with config and native progress callbacks.
pub fn resolve_or_bootstrap_yono_weights_with_config_and_progress<F>(
    format: YonoWeightFormat,
    cfg: &YonoBootstrapConfig,
    progress: F,
) -> Result<YonoWeights, ModelBootstrapError>
where
    F: Fn(String) + Send + Sync + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (format, cfg, progress);
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native::resolve_or_bootstrap_yono_weights_native(format, cfg, Some(Arc::new(progress)))
    }
}

/// Returns the default on-disk cache root for this crate.
pub fn default_cache_root() -> Result<PathBuf, ModelBootstrapError> {
    #[cfg(target_arch = "wasm32")]
    {
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let cfg = native::apply_env_overrides(YonoBootstrapConfig::default());
        native::default_cache_root_native(&cfg)
    }
}

/// Resolves the on-disk cache root using explicit non-env configuration.
pub fn default_cache_root_with_config(
    cfg: &YonoBootstrapConfig,
) -> Result<PathBuf, ModelBootstrapError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = cfg;
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        native::default_cache_root_native(cfg)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelBootstrapError {
    #[error("automatic model bootstrap is not supported on wasm32 targets")]
    UnsupportedTarget,
    #[error("failed to resolve user home directory for model cache")]
    MissingHomeDir,
    #[error("failed to create cache directory `{path}`: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to download model `{url}`: {message}")]
    Download { url: String, message: String },
    #[error("failed to write model file `{path}`: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("download returned invalid content for `{url}`: {message}")]
    InvalidContent { url: String, message: String },
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::fs::{self, File};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::thread::sleep;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::{BootstrapProgressCallback, ModelBootstrapError, YonoBootstrapConfig};
    use burn_yono::parts::{
        burnpack_parts_manifest_path, manifest_is_complete, read_parts_manifest,
        resolve_part_entry_path, BurnpackPartEntry,
    };
    use burn_yono::{YonoWeightFormat, YonoWeightPrecision, YonoWeights};

    const CACHE_ROOT_DIR: &str = ".burn_reconstruction";
    const CACHE_MODELS_SUBDIR: &str = "models/yono";
    const DOWNLOAD_ATTEMPTS: u32 = 4;
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
    const READ_TIMEOUT: Duration = Duration::from_secs(60);
    const WRITE_TIMEOUT: Duration = Duration::from_secs(60);

    pub fn apply_env_overrides(mut cfg: YonoBootstrapConfig) -> YonoBootstrapConfig {
        if let Some(explicit) = std::env::var_os("BURN_RECONSTRUCTION_CACHE_DIR") {
            cfg.cache_root = Some(PathBuf::from(explicit).join("models/yono"));
        }
        if let Ok(value) = std::env::var("BURN_RECONSTRUCTION_MODEL_BASE_URL") {
            cfg.model_base_url = value;
        }
        if let Ok(value) = std::env::var("BURN_RECONSTRUCTION_YONO_REMOTE_ROOT") {
            cfg.yono_remote_root = value;
        }
        if let Ok(value) = std::env::var("BURN_RECONSTRUCTION_YONO_BACKBONE_URL") {
            cfg.backbone_url = Some(value);
        }
        if let Ok(value) = std::env::var("BURN_RECONSTRUCTION_YONO_HEAD_URL") {
            cfg.head_url = Some(value);
        }
        if let Ok(value) = std::env::var("BURN_RECONSTRUCTION_YONO_PREFER_PARTS") {
            cfg.prefer_burnpack_parts = parse_bool(&value).unwrap_or(cfg.prefer_burnpack_parts);
        }
        if let Ok(value) = std::env::var("BURN_RECONSTRUCTION_YONO_BURNPACK_PRECISION") {
            if let Some(precision) = parse_precision(&value) {
                cfg.burnpack_precision = precision;
            }
        }
        cfg
    }

    pub fn resolve_or_bootstrap_yono_weights_native(
        format: YonoWeightFormat,
        cfg: &YonoBootstrapConfig,
        progress: Option<BootstrapProgressCallback>,
    ) -> Result<YonoWeights, ModelBootstrapError> {
        let cache_root = default_cache_root_native(cfg)?;
        emit_progress(
            progress.as_ref(),
            format!(
                "resolving model cache under {}",
                cache_root.to_string_lossy()
            ),
        );
        fs::create_dir_all(&cache_root).map_err(|source| ModelBootstrapError::CreateDir {
            path: cache_root.clone(),
            source,
        })?;

        match format {
            YonoWeightFormat::Safetensors => {
                let (backbone_name, head_name) =
                    file_names_for_format(format, YonoWeightPrecision::F32);
                let backbone_path = cache_root.join(backbone_name);
                let head_path = cache_root.join(head_name);
                let (backbone_url, head_url) = resolve_remote_urls(backbone_name, head_name, cfg);
                emit_download_status(
                    progress.as_ref(),
                    "backbone weights",
                    backbone_path.as_path(),
                    backbone_url.as_str(),
                );
                ensure_file_cached(&backbone_path, &backbone_url)?;
                emit_download_status(
                    progress.as_ref(),
                    "head weights",
                    head_path.as_path(),
                    head_url.as_str(),
                );
                ensure_file_cached(&head_path, &head_url)?;
                Ok(YonoWeights::new(backbone_path, head_path)
                    .with_format(format)
                    .with_precision(YonoWeightPrecision::F32))
            }
            YonoWeightFormat::Burnpack => {
                let mut last_error: Option<ModelBootstrapError> = None;
                for precision in burnpack_precision_attempts(cfg.burnpack_precision) {
                    emit_progress(
                        progress.as_ref(),
                        format!(
                            "resolving burnpack weights (precision {})...",
                            precision_label(precision)
                        ),
                    );
                    let (backbone_name, head_name) = file_names_for_format(format, precision);
                    let backbone_path = cache_root.join(backbone_name);
                    let head_path = cache_root.join(head_name);
                    let (backbone_url, head_url) =
                        resolve_remote_urls(backbone_name, head_name, cfg);

                    let result = ensure_model_pair_cached(
                        &backbone_path,
                        &backbone_url,
                        &head_path,
                        &head_url,
                        cfg.prefer_burnpack_parts,
                        progress.clone(),
                    );

                    match result {
                        Ok(()) => {
                            return Ok(YonoWeights::new(backbone_path, head_path)
                                .with_format(format)
                                .with_precision(precision));
                        }
                        Err(err) => {
                            last_error = Some(err);
                        }
                    }
                }
                Err(
                    last_error.unwrap_or_else(|| ModelBootstrapError::InvalidContent {
                        url: "unknown".to_string(),
                        message: "failed to resolve burnpack precision candidates".to_string(),
                    }),
                )
            }
        }
    }

    pub fn default_cache_root_native(
        cfg: &YonoBootstrapConfig,
    ) -> Result<PathBuf, ModelBootstrapError> {
        if let Some(explicit) = cfg.cache_root.as_ref() {
            return Ok(explicit.clone());
        }
        let Some(home) = user_home_dir() else {
            return Err(ModelBootstrapError::MissingHomeDir);
        };
        Ok(home.join(CACHE_ROOT_DIR).join(CACHE_MODELS_SUBDIR))
    }

    fn resolve_remote_urls(
        backbone_name: &str,
        head_name: &str,
        cfg: &YonoBootstrapConfig,
    ) -> (String, String) {
        if let (Some(backbone), Some(head)) = (cfg.backbone_url.as_ref(), cfg.head_url.as_ref()) {
            return (backbone.clone(), head.clone());
        }

        let remote_root = join_url(cfg.model_base_url.as_str(), cfg.yono_remote_root.as_str());

        (
            cfg.backbone_url
                .clone()
                .unwrap_or_else(|| join_url(&remote_root, backbone_name)),
            cfg.head_url
                .clone()
                .unwrap_or_else(|| join_url(&remote_root, head_name)),
        )
    }

    fn emit_progress(progress: Option<&BootstrapProgressCallback>, message: String) {
        if let Some(progress) = progress {
            progress(message);
        }
    }

    fn emit_download_status(
        progress: Option<&BootstrapProgressCallback>,
        component: &str,
        path: &Path,
        url: &str,
    ) {
        if path.exists() {
            emit_progress(
                progress,
                format!(
                    "using cached {component}: {}",
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("weights")
                ),
            );
            return;
        }
        emit_progress(
            progress,
            format!("downloading {component}: {}", url_leaf_name(url)),
        );
    }

    fn precision_label(precision: YonoWeightPrecision) -> &'static str {
        match precision {
            YonoWeightPrecision::F16 => "f16",
            YonoWeightPrecision::F32 => "f32",
        }
    }

    fn url_leaf_name(url: &str) -> &str {
        url.rsplit('/').next().unwrap_or(url)
    }

    fn ensure_model_pair_cached(
        backbone_path: &Path,
        backbone_url: &str,
        head_path: &Path,
        head_url: &str,
        prefer_parts: bool,
        progress: Option<BootstrapProgressCallback>,
    ) -> Result<(), ModelBootstrapError> {
        let backbone_path = backbone_path.to_path_buf();
        let head_path = head_path.to_path_buf();
        let backbone_url = backbone_url.to_string();
        let head_url = head_url.to_string();

        let backbone_url_for_panic = backbone_url.clone();
        let head_url_for_panic = head_url.clone();
        let backbone_progress = progress.clone();
        let backbone_task = std::thread::spawn(move || {
            let component = "backbone";
            if prefer_parts {
                ensure_burnpack_bundle_cached(
                    backbone_path.as_path(),
                    backbone_url.as_str(),
                    component,
                    backbone_progress,
                )
            } else {
                emit_download_status(
                    backbone_progress.as_ref(),
                    "backbone weights",
                    backbone_path.as_path(),
                    backbone_url.as_str(),
                );
                ensure_file_cached(backbone_path.as_path(), backbone_url.as_str())
            }
        });
        let head_progress = progress;
        let head_task = std::thread::spawn(move || {
            let component = "head";
            if prefer_parts {
                ensure_burnpack_bundle_cached(
                    head_path.as_path(),
                    head_url.as_str(),
                    component,
                    head_progress,
                )
            } else {
                emit_download_status(
                    head_progress.as_ref(),
                    "head weights",
                    head_path.as_path(),
                    head_url.as_str(),
                );
                ensure_file_cached(head_path.as_path(), head_url.as_str())
            }
        });

        let backbone_result =
            backbone_task
                .join()
                .map_err(|_| ModelBootstrapError::InvalidContent {
                    url: backbone_url_for_panic,
                    message: "model bootstrap worker panicked".to_string(),
                })?;
        let head_result = head_task
            .join()
            .map_err(|_| ModelBootstrapError::InvalidContent {
                url: head_url_for_panic,
                message: "model bootstrap worker panicked".to_string(),
            })?;

        backbone_result?;
        head_result?;
        Ok(())
    }

    fn ensure_burnpack_bundle_cached(
        path: &Path,
        url: &str,
        component: &str,
        progress: Option<BootstrapProgressCallback>,
    ) -> Result<(), ModelBootstrapError> {
        let manifest_path = burnpack_parts_manifest_path(path);
        if manifest_is_complete(manifest_path.as_path()).unwrap_or(false) {
            emit_progress(
                progress.as_ref(),
                format!(
                    "using cached {component} parts manifest: {}",
                    manifest_path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("manifest")
                ),
            );
            return Ok(());
        }

        let manifest_url = format!("{url}.parts.json");
        emit_progress(
            progress.as_ref(),
            format!(
                "downloading {component} manifest: {}",
                url_leaf_name(manifest_url.as_str())
            ),
        );
        if let Some(manifest_bytes) = download_optional_bytes(manifest_url.as_str())? {
            if let Some(parent) = manifest_path.parent() {
                fs::create_dir_all(parent).map_err(|source| ModelBootstrapError::CreateDir {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
            write_bytes_atomically(manifest_path.as_path(), manifest_bytes.as_slice())?;

            let manifest = read_parts_manifest(manifest_path.as_path()).map_err(|message| {
                ModelBootstrapError::InvalidContent {
                    url: manifest_url.clone(),
                    message,
                }
            })?;
            if manifest.parts.is_empty() {
                return Err(ModelBootstrapError::InvalidContent {
                    url: manifest_url,
                    message: "parts manifest contains no parts".to_string(),
                });
            }

            let total_parts = manifest.parts.len();
            for (index, part) in manifest.parts.iter().enumerate() {
                let local_part_path =
                    resolve_part_entry_path(manifest_path.as_path(), part.path.as_str()).map_err(
                        |message| ModelBootstrapError::InvalidContent {
                            url: manifest_url.clone(),
                            message,
                        },
                    )?;
                if part_matches_cache(local_part_path.as_path(), part)? {
                    emit_progress(
                        progress.as_ref(),
                        format!(
                            "cached {component} part {}/{}: {}",
                            index + 1,
                            total_parts,
                            local_part_path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("part")
                        ),
                    );
                    continue;
                }
                let part_url =
                    resolve_manifest_entry_url(manifest_url.as_str(), part.path.as_str());
                emit_progress(
                    progress.as_ref(),
                    format!(
                        "downloading {component} part {}/{}: {}",
                        index + 1,
                        total_parts,
                        url_leaf_name(part_url.as_str())
                    ),
                );
                ensure_file_cached(local_part_path.as_path(), part_url.as_str())?;
                if !part_matches_cache(local_part_path.as_path(), part)? {
                    return Err(ModelBootstrapError::InvalidContent {
                        url: part_url,
                        message: format!(
                            "downloaded part did not match manifest bytes for {}",
                            local_part_path.display()
                        ),
                    });
                }
            }

            if manifest_is_complete(manifest_path.as_path()).unwrap_or(false) {
                emit_progress(
                    progress.as_ref(),
                    format!("downloaded {component} parts ({total_parts}/{total_parts})"),
                );
                return Ok(());
            }
            return Err(ModelBootstrapError::InvalidContent {
                url: manifest_url,
                message: format!(
                    "parts manifest remained incomplete after download: {}",
                    manifest_path.display()
                ),
            });
        }

        // Fallback to monolithic burnpack when no parts manifest is available.
        emit_progress(
            progress.as_ref(),
            format!(
                "{component} parts manifest unavailable; downloading monolithic {}",
                url_leaf_name(url)
            ),
        );
        emit_download_status(
            progress.as_ref(),
            &format!("{component} weights"),
            path,
            url,
        );
        ensure_file_cached(path, url)
    }

    fn download_optional_bytes(url: &str) -> Result<Option<Vec<u8>>, ModelBootstrapError> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(CONNECT_TIMEOUT)
            .timeout_read(READ_TIMEOUT)
            .timeout_write(WRITE_TIMEOUT)
            .build();
        let response = match agent.get(url).call() {
            Ok(value) => value,
            Err(ureq::Error::Status(code, _)) if code == 403 || code == 404 => {
                return Ok(None);
            }
            Err(err) => {
                return Err(ModelBootstrapError::Download {
                    url: url.to_string(),
                    message: match err {
                        ureq::Error::Status(code, response) => {
                            format!("HTTP {code} ({})", response.status_text())
                        }
                        ureq::Error::Transport(transport) => {
                            format!("transport error: {transport}")
                        }
                    },
                });
            }
        };

        let mut reader = response.into_reader();
        let mut out = Vec::new();
        reader
            .read_to_end(&mut out)
            .map_err(|err| ModelBootstrapError::Download {
                url: url.to_string(),
                message: format!("failed reading response body: {err}"),
            })?;
        if out.is_empty() {
            return Err(ModelBootstrapError::InvalidContent {
                url: url.to_string(),
                message: "downloaded file is empty".to_string(),
            });
        }
        Ok(Some(out))
    }

    fn resolve_manifest_entry_url(manifest_url: &str, entry_url: &str) -> String {
        if entry_url.contains("://") || entry_url.starts_with('/') {
            return entry_url.to_string();
        }
        let normalized = entry_url.replace('\\', "/");
        if let Some((parent, _)) = manifest_url.rsplit_once('/') {
            return format!("{}/{}", parent.trim_end_matches('/'), normalized);
        }
        normalized
    }

    fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<(), ModelBootstrapError> {
        let tmp = temp_download_path(path);
        let mut file = File::create(&tmp).map_err(|source| ModelBootstrapError::Write {
            path: tmp.clone(),
            source,
        })?;
        file.write_all(bytes)
            .map_err(|source| ModelBootstrapError::Write {
                path: tmp.clone(),
                source,
            })?;
        file.flush().map_err(|source| ModelBootstrapError::Write {
            path: tmp.clone(),
            source,
        })?;
        if path.exists() {
            fs::remove_file(path).map_err(|source| ModelBootstrapError::Write {
                path: path.to_path_buf(),
                source,
            })?;
        }
        fs::rename(&tmp, path).map_err(|source| ModelBootstrapError::Write {
            path: path.to_path_buf(),
            source,
        })
    }

    fn part_matches_cache(
        path: &Path,
        part: &BurnpackPartEntry,
    ) -> Result<bool, ModelBootstrapError> {
        if !path.exists() {
            return Ok(false);
        }
        if part.bytes == 0 {
            return Ok(true);
        }
        let bytes = fs::metadata(path)
            .map_err(|source| ModelBootstrapError::Write {
                path: path.to_path_buf(),
                source,
            })?
            .len();
        Ok(bytes == part.bytes)
    }

    fn ensure_file_cached(path: &Path, url: &str) -> Result<(), ModelBootstrapError> {
        if path.exists() {
            return Ok(());
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| ModelBootstrapError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let tmp = temp_download_path(path);
        let mut last_error: Option<String> = None;

        for attempt in 1..=DOWNLOAD_ATTEMPTS {
            match download_to_file(url, &tmp) {
                Ok(_) => {
                    if path.exists() {
                        fs::remove_file(path).map_err(|source| ModelBootstrapError::Write {
                            path: path.to_path_buf(),
                            source,
                        })?;
                    }
                    fs::rename(&tmp, path).map_err(|source| ModelBootstrapError::Write {
                        path: path.to_path_buf(),
                        source,
                    })?;
                    return Ok(());
                }
                Err(err) => {
                    let _ = fs::remove_file(&tmp);
                    if attempt == DOWNLOAD_ATTEMPTS {
                        return Err(ModelBootstrapError::Download {
                            url: url.to_string(),
                            message: err,
                        });
                    }
                    last_error = Some(err);
                    sleep(retry_delay(attempt));
                }
            }
        }

        Err(ModelBootstrapError::Download {
            url: url.to_string(),
            message: last_error.unwrap_or_else(|| "unknown download error".to_string()),
        })
    }

    fn download_to_file(url: &str, destination: &Path) -> Result<(), String> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(CONNECT_TIMEOUT)
            .timeout_read(READ_TIMEOUT)
            .timeout_write(WRITE_TIMEOUT)
            .build();

        let response = agent.get(url).call().map_err(|err| match err {
            ureq::Error::Status(code, response) => {
                format!("HTTP {code} ({})", response.status_text())
            }
            ureq::Error::Transport(transport) => format!("transport error: {transport}"),
        })?;

        let expected_len = response
            .header("Content-Length")
            .and_then(|value| value.parse::<u64>().ok());

        let mut reader = response.into_reader();
        let mut writer = File::create(destination)
            .map_err(|err| format!("failed to create {}: {err}", destination.display()))?;

        let mut total = 0u64;
        let mut buffer = [0u8; 1024 * 1024];
        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|err| format!("failed reading response body: {err}"))?;
            if read == 0 {
                break;
            }
            writer
                .write_all(&buffer[..read])
                .map_err(|err| format!("failed writing {}: {err}", destination.display()))?;
            total = total.saturating_add(read as u64);
        }
        writer
            .flush()
            .map_err(|err| format!("failed to flush {}: {err}", destination.display()))?;

        if total == 0 {
            return Err("downloaded file is empty".to_string());
        }
        if let Some(expected) = expected_len {
            if expected != total {
                return Err(format!(
                    "content-length mismatch (expected {expected} bytes, wrote {total} bytes)"
                ));
            }
        }
        Ok(())
    }

    fn retry_delay(attempt: u32) -> Duration {
        let capped = attempt.min(6);
        Duration::from_millis(600_u64.saturating_mul(1_u64 << capped))
    }

    fn file_names_for_format(
        format: YonoWeightFormat,
        precision: YonoWeightPrecision,
    ) -> (&'static str, &'static str) {
        match format {
            YonoWeightFormat::Safetensors => (
                "yono_backbone_weights.safetensors",
                "yono_head_weights.safetensors",
            ),
            YonoWeightFormat::Burnpack => match precision {
                YonoWeightPrecision::F16 => ("yono_backbone_f16.bpk", "yono_head_f16.bpk"),
                YonoWeightPrecision::F32 => ("yono_backbone.bpk", "yono_head.bpk"),
            },
        }
    }

    fn burnpack_precision_attempts(preferred: YonoWeightPrecision) -> [YonoWeightPrecision; 2] {
        match preferred {
            YonoWeightPrecision::F16 => [YonoWeightPrecision::F16, YonoWeightPrecision::F32],
            YonoWeightPrecision::F32 => [YonoWeightPrecision::F32, YonoWeightPrecision::F16],
        }
    }

    fn join_url(root: &str, rel: &str) -> String {
        let mut out = root.trim_end_matches('/').to_string();
        out.push('/');
        out.push_str(rel.trim_start_matches('/'));
        out
    }

    fn temp_download_path(path: &Path) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or_default();
        let file = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.bin");
        path.with_file_name(format!("{file}.download-{stamp}.tmp"))
    }

    fn parse_bool(value: &str) -> Option<bool> {
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    }

    fn parse_precision(value: &str) -> Option<YonoWeightPrecision> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => Some(YonoWeightPrecision::F16),
            "f32" | "fp32" | "full" => Some(YonoWeightPrecision::F32),
            _ => None,
        }
    }

    fn user_home_dir() -> Option<PathBuf> {
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            return Some(home);
        }
        #[cfg(target_os = "windows")]
        {
            if let Some(profile) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
                return Some(profile);
            }
            let drive = std::env::var_os("HOMEDRIVE");
            let path = std::env::var_os("HOMEPATH");
            if let (Some(drive), Some(path)) = (drive, path) {
                return Some(PathBuf::from(format!(
                    "{}{}",
                    drive.to_string_lossy(),
                    path.to_string_lossy()
                )));
            }
        }
        None
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use burn_yono::parts::{
        burnpack_parts_manifest_path, read_parts_manifest, resolve_part_entry_path,
    };
    use burn_yono::YonoWeightFormat;

    use crate::bootstrap::{
        resolve_or_bootstrap_yono_weights,
        resolve_or_bootstrap_yono_weights_with_precision_and_progress,
    };

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    #[test]
    fn resolves_from_cache_and_bootstraps_missing_weights() {
        let _lock = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock should succeed");

        let tmp = unique_tmp_dir();
        let cache_root = tmp.join("cache");
        let source_backbone = b"backbone-bytes".to_vec();
        let source_head = b"head-bytes".to_vec();

        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(AtomicUsize::new(0));
        let (base_url, server) = spawn_test_server(
            source_backbone.clone(),
            source_head.clone(),
            stop.clone(),
            requests.clone(),
        );

        std::env::set_var("BURN_RECONSTRUCTION_CACHE_DIR", &cache_root);
        std::env::set_var(
            "BURN_RECONSTRUCTION_YONO_BACKBONE_URL",
            format!("{base_url}/backbone"),
        );
        std::env::set_var(
            "BURN_RECONSTRUCTION_YONO_HEAD_URL",
            format!("{base_url}/head"),
        );

        let first = resolve_or_bootstrap_yono_weights(YonoWeightFormat::Safetensors)
            .expect("bootstrap should succeed");
        assert_eq!(
            requests.load(Ordering::SeqCst),
            2,
            "first resolve should fetch both files"
        );

        let cached_backbone =
            std::fs::read(&first.backbone).expect("cached backbone should be readable");
        let cached_head = std::fs::read(&first.head).expect("cached head should be readable");
        assert_eq!(cached_backbone, source_backbone);
        assert_eq!(cached_head, source_head);

        let second = resolve_or_bootstrap_yono_weights(YonoWeightFormat::Safetensors)
            .expect("second resolve should succeed");
        assert_eq!(
            requests.load(Ordering::SeqCst),
            2,
            "cached resolve should not re-download"
        );
        assert_eq!(first.backbone, second.backbone);
        assert_eq!(first.head, second.head);

        std::env::remove_var("BURN_RECONSTRUCTION_CACHE_DIR");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_BACKBONE_URL");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_HEAD_URL");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_BURNPACK_PRECISION");

        stop.store(true, Ordering::SeqCst);
        let _ = std::net::TcpStream::connect(
            base_url.trim_start_matches("http://").trim_end_matches('/'),
        );
        server.join().expect("server thread should exit cleanly");
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn bootstraps_burnpack_from_parts_manifest() {
        let _lock = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock should succeed");

        let tmp = unique_tmp_dir();
        let cache_root = tmp.join("cache");
        let backbone_part = b"backbone-part".to_vec();
        let head_part = b"head-part".to_vec();

        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(AtomicUsize::new(0));
        let (base_url, server) = spawn_parts_server(
            backbone_part.clone(),
            head_part.clone(),
            stop.clone(),
            requests.clone(),
        );

        std::env::set_var("BURN_RECONSTRUCTION_CACHE_DIR", &cache_root);
        std::env::set_var(
            "BURN_RECONSTRUCTION_YONO_BACKBONE_URL",
            format!("{base_url}/yono_backbone.bpk"),
        );
        std::env::set_var(
            "BURN_RECONSTRUCTION_YONO_HEAD_URL",
            format!("{base_url}/yono_head.bpk"),
        );
        std::env::set_var("BURN_RECONSTRUCTION_YONO_PREFER_PARTS", "1");

        let resolved = resolve_or_bootstrap_yono_weights(YonoWeightFormat::Burnpack)
            .expect("burnpack bootstrap should succeed");

        let backbone_manifest = burnpack_parts_manifest_path(resolved.backbone.as_path());
        let head_manifest = burnpack_parts_manifest_path(resolved.head.as_path());
        let backbone_manifest_data =
            read_parts_manifest(backbone_manifest.as_path()).expect("read backbone manifest");
        let head_manifest_data =
            read_parts_manifest(head_manifest.as_path()).expect("read head manifest");
        let backbone_part_path = resolve_part_entry_path(
            backbone_manifest.as_path(),
            backbone_manifest_data.parts[0].path.as_str(),
        )
        .expect("resolve backbone part path");
        let head_part_path = resolve_part_entry_path(
            head_manifest.as_path(),
            head_manifest_data.parts[0].path.as_str(),
        )
        .expect("resolve head part path");

        assert!(
            backbone_manifest.exists(),
            "expected backbone parts manifest"
        );
        assert!(head_manifest.exists(), "expected head parts manifest");
        assert!(backbone_part_path.exists(), "expected backbone part file");
        assert!(head_part_path.exists(), "expected head part file");
        assert_eq!(
            std::fs::read(backbone_part_path).expect("read backbone part"),
            backbone_part
        );
        assert_eq!(
            std::fs::read(head_part_path).expect("read head part"),
            head_part
        );
        assert!(
            requests.load(Ordering::SeqCst) >= 4,
            "expected at least manifest+part requests for both files"
        );

        std::env::remove_var("BURN_RECONSTRUCTION_CACHE_DIR");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_BACKBONE_URL");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_HEAD_URL");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_PREFER_PARTS");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_BURNPACK_PRECISION");

        stop.store(true, Ordering::SeqCst);
        let _ = std::net::TcpStream::connect(
            base_url.trim_start_matches("http://").trim_end_matches('/'),
        );
        server.join().expect("server thread should exit cleanly");
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn reports_native_progress_for_parts_bootstrap() {
        let _lock = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock should succeed");

        let tmp = unique_tmp_dir();
        let cache_root = tmp.join("cache");
        let backbone_part = b"backbone-part".to_vec();
        let head_part = b"head-part".to_vec();

        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(AtomicUsize::new(0));
        let (base_url, server) =
            spawn_parts_server(backbone_part, head_part, stop.clone(), requests);

        std::env::set_var("BURN_RECONSTRUCTION_CACHE_DIR", &cache_root);
        std::env::set_var(
            "BURN_RECONSTRUCTION_YONO_BACKBONE_URL",
            format!("{base_url}/yono_backbone.bpk"),
        );
        std::env::set_var(
            "BURN_RECONSTRUCTION_YONO_HEAD_URL",
            format!("{base_url}/yono_head.bpk"),
        );
        std::env::set_var("BURN_RECONSTRUCTION_YONO_PREFER_PARTS", "1");

        let progress = Arc::new(Mutex::new(Vec::<String>::new()));
        let progress_sink = progress.clone();
        let resolved = resolve_or_bootstrap_yono_weights_with_precision_and_progress(
            YonoWeightFormat::Burnpack,
            burn_yono::YonoWeightPrecision::F16,
            move |message| {
                progress_sink
                    .lock()
                    .expect("progress lock should succeed")
                    .push(message);
            },
        )
        .expect("burnpack bootstrap should succeed");
        let backbone_manifest = burnpack_parts_manifest_path(resolved.backbone.as_path());
        let head_manifest = burnpack_parts_manifest_path(resolved.head.as_path());
        assert!(
            backbone_manifest.exists(),
            "expected backbone parts manifest"
        );
        assert!(head_manifest.exists(), "expected head parts manifest");

        let collected = progress
            .lock()
            .expect("progress lock should succeed")
            .clone();
        let has_backbone_manifest = collected
            .iter()
            .any(|entry| entry.contains("downloading backbone manifest"));
        let has_head_manifest = collected
            .iter()
            .any(|entry| entry.contains("downloading head manifest"));
        let has_part_progress = collected.iter().any(|entry| entry.contains("part 1/1"));
        let has_backbone_complete = collected
            .iter()
            .any(|entry| entry.contains("downloaded backbone parts (1/1)"));
        let has_head_complete = collected
            .iter()
            .any(|entry| entry.contains("downloaded head parts (1/1)"));

        assert!(
            has_backbone_manifest,
            "missing backbone manifest progress; got {:?}",
            collected
        );
        assert!(
            has_head_manifest,
            "missing head manifest progress; got {:?}",
            collected
        );
        assert!(
            has_part_progress,
            "missing part progress; got {:?}",
            collected
        );
        assert!(
            has_backbone_complete,
            "missing backbone completion progress; got {:?}",
            collected
        );
        assert!(
            has_head_complete,
            "missing head completion progress; got {:?}",
            collected
        );

        std::env::remove_var("BURN_RECONSTRUCTION_CACHE_DIR");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_BACKBONE_URL");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_HEAD_URL");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_PREFER_PARTS");
        std::env::remove_var("BURN_RECONSTRUCTION_YONO_BURNPACK_PRECISION");

        stop.store(true, Ordering::SeqCst);
        let _ = std::net::TcpStream::connect(
            base_url.trim_start_matches("http://").trim_end_matches('/'),
        );
        server.join().expect("server thread should exit cleanly");
        let _ = std::fs::remove_dir_all(tmp);
    }

    fn spawn_test_server(
        backbone: Vec<u8>,
        head: Vec<u8>,
        stop: Arc<AtomicBool>,
        requests: Arc<AtomicUsize>,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        listener
            .set_nonblocking(true)
            .expect("set_nonblocking should succeed");
        let addr = listener.local_addr().expect("server addr");
        let base_url = format!("http://{}", addr);

        let handle = thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        requests.fetch_add(1, Ordering::SeqCst);
                        let mut buffer = [0u8; 1024];
                        let read = stream.read(&mut buffer).unwrap_or(0);
                        let req = String::from_utf8_lossy(&buffer[..read]);
                        let path = req.split_whitespace().nth(1).unwrap_or("/");

                        let (status, body) = match path {
                            "/backbone" => ("200 OK", backbone.as_slice()),
                            "/head" => ("200 OK", head.as_slice()),
                            _ => ("404 Not Found", b"".as_slice()),
                        };

                        let response = format!(
                            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(response.as_bytes());
                        let _ = stream.write_all(body);
                        let _ = stream.flush();
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });

        (base_url, handle)
    }

    fn spawn_parts_server(
        backbone_part: Vec<u8>,
        head_part: Vec<u8>,
        stop: Arc<AtomicBool>,
        requests: Arc<AtomicUsize>,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        listener
            .set_nonblocking(true)
            .expect("set_nonblocking should succeed");
        let addr = listener.local_addr().expect("server addr");
        let base_url = format!("http://{}", addr);

        let handle = thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        requests.fetch_add(1, Ordering::SeqCst);
                        let mut buffer = [0u8; 2048];
                        let read = stream.read(&mut buffer).unwrap_or(0);
                        let req = String::from_utf8_lossy(&buffer[..read]);
                        let path = req.split_whitespace().nth(1).unwrap_or("/");

                        let (status, body) = match path {
                            "/yono_backbone.bpk.parts.json" => (
                                "200 OK",
                                format!(
                                    "{{\"version\":1,\"source_file\":\"yono_backbone.bpk\",\"source_modified_unix_ms\":0,\"total_bytes\":0,\"max_part_bytes\":0,\"parts\":[{{\"path\":\"yono_backbone.bpk.part-00000.bpk\",\"bytes\":{},\"sha256\":\"\",\"tensors\":1}}]}}",
                                    backbone_part.len()
                                )
                                .into_bytes(),
                            ),
                            "/yono_head.bpk.parts.json" => (
                                "200 OK",
                                format!(
                                    "{{\"version\":1,\"source_file\":\"yono_head.bpk\",\"source_modified_unix_ms\":0,\"total_bytes\":0,\"max_part_bytes\":0,\"parts\":[{{\"path\":\"yono_head.bpk.part-00000.bpk\",\"bytes\":{},\"sha256\":\"\",\"tensors\":1}}]}}",
                                    head_part.len()
                                )
                                .into_bytes(),
                            ),
                            "/yono_backbone.bpk.part-00000.bpk" => {
                                ("200 OK", backbone_part.clone())
                            }
                            "/yono_head.bpk.part-00000.bpk" => ("200 OK", head_part.clone()),
                            _ => ("404 Not Found", Vec::new()),
                        };

                        let response = format!(
                            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(response.as_bytes());
                        let _ = stream.write_all(body.as_slice());
                        let _ = stream.flush();
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });

        (base_url, handle)
    }

    fn unique_tmp_dir() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("burn_gaussian_bootstrap_test_{stamp}"));
        std::fs::create_dir_all(&dir).expect("create tmp dir");
        dir
    }
}
