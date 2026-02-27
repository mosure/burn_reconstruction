use std::path::PathBuf;

use burn_yono::{YonoWeightFormat, YonoWeights};

/// Model source and cache settings for YoNo bootstrap.
///
/// This is an explicit configuration surface that does not depend on process
/// environment variables, which makes it suitable for host-managed runtimes.
#[derive(Debug, Clone)]
pub struct YonoBootstrapConfig {
    /// Optional explicit cache directory where YoNo weights should be stored.
    ///
    /// When unset on native targets, the default is:
    /// `~/.burn_gaussian_splatting/models/yono`.
    pub cache_root: Option<PathBuf>,
    /// Base model URL.
    pub model_base_url: String,
    /// Remote directory beneath `model_base_url`.
    pub yono_remote_root: String,
    /// Optional full URL override for backbone weights.
    pub backbone_url: Option<String>,
    /// Optional full URL override for head weights.
    pub head_url: Option<String>,
}

impl Default for YonoBootstrapConfig {
    fn default() -> Self {
        Self {
            cache_root: None,
            model_base_url: "https://aberration.technology/model".to_string(),
            yono_remote_root: "YoNoSplat".to_string(),
            backbone_url: None,
            head_url: None,
        }
    }
}

/// Resolves YoNoSplat weights from local cache and bootstraps downloads on first use.
///
/// Native behavior:
/// - cache root: `~/.burn_gaussian_splatting/models/yono`
/// - remote root: `https://aberration.technology/model/YoNoSplat`
///
/// Environment overrides:
/// - `BURN_GAUSSIAN_SPLATTING_CACHE_DIR` (absolute cache root)
/// - `BURN_GAUSSIAN_SPLATTING_MODEL_BASE_URL`
/// - `BURN_GAUSSIAN_SPLATTING_YONO_REMOTE_ROOT`
/// - `BURN_GAUSSIAN_SPLATTING_YONO_BACKBONE_URL`
/// - `BURN_GAUSSIAN_SPLATTING_YONO_HEAD_URL`
pub fn resolve_or_bootstrap_yono_weights(
    format: YonoWeightFormat,
) -> Result<YonoWeights, ModelBootstrapError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = format;
        Err(ModelBootstrapError::UnsupportedTarget)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let cfg = native::apply_env_overrides(YonoBootstrapConfig::default());
        native::resolve_or_bootstrap_yono_weights_native(format, &cfg)
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
        native::resolve_or_bootstrap_yono_weights_native(format, cfg)
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

    use super::{ModelBootstrapError, YonoBootstrapConfig};
    use burn_yono::{YonoWeightFormat, YonoWeights};

    const CACHE_ROOT_DIR: &str = ".burn_gaussian_splatting";
    const CACHE_MODELS_SUBDIR: &str = "models/yono";
    const DOWNLOAD_ATTEMPTS: u32 = 4;
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
    const READ_TIMEOUT: Duration = Duration::from_secs(60);
    const WRITE_TIMEOUT: Duration = Duration::from_secs(60);

    pub fn apply_env_overrides(mut cfg: YonoBootstrapConfig) -> YonoBootstrapConfig {
        if let Some(explicit) = std::env::var_os("BURN_GAUSSIAN_SPLATTING_CACHE_DIR") {
            cfg.cache_root = Some(PathBuf::from(explicit).join("models/yono"));
        }
        if let Ok(value) = std::env::var("BURN_GAUSSIAN_SPLATTING_MODEL_BASE_URL") {
            cfg.model_base_url = value;
        }
        if let Ok(value) = std::env::var("BURN_GAUSSIAN_SPLATTING_YONO_REMOTE_ROOT") {
            cfg.yono_remote_root = value;
        }
        if let Ok(value) = std::env::var("BURN_GAUSSIAN_SPLATTING_YONO_BACKBONE_URL") {
            cfg.backbone_url = Some(value);
        }
        if let Ok(value) = std::env::var("BURN_GAUSSIAN_SPLATTING_YONO_HEAD_URL") {
            cfg.head_url = Some(value);
        }
        cfg
    }

    pub fn resolve_or_bootstrap_yono_weights_native(
        format: YonoWeightFormat,
        cfg: &YonoBootstrapConfig,
    ) -> Result<YonoWeights, ModelBootstrapError> {
        let cache_root = default_cache_root_native(cfg)?;
        fs::create_dir_all(&cache_root).map_err(|source| ModelBootstrapError::CreateDir {
            path: cache_root.clone(),
            source,
        })?;

        let (backbone_name, head_name) = file_names_for_format(format);
        let backbone_path = cache_root.join(backbone_name);
        let head_path = cache_root.join(head_name);

        let (backbone_url, head_url) = resolve_remote_urls(format, cfg);

        ensure_file_cached(&backbone_path, &backbone_url)?;
        ensure_file_cached(&head_path, &head_url)?;

        Ok(YonoWeights::new(backbone_path, head_path).with_format(format))
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
        format: YonoWeightFormat,
        cfg: &YonoBootstrapConfig,
    ) -> (String, String) {
        let (backbone_name, head_name) = file_names_for_format(format);

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

    fn file_names_for_format(format: YonoWeightFormat) -> (&'static str, &'static str) {
        match format {
            YonoWeightFormat::Safetensors => (
                "yono_backbone_weights.safetensors",
                "yono_head_weights.safetensors",
            ),
            YonoWeightFormat::Burnpack => ("yono_backbone.bpk", "yono_head.bpk"),
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

    use burn_yono::YonoWeightFormat;

    use crate::bootstrap::resolve_or_bootstrap_yono_weights;

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

        std::env::set_var("BURN_GAUSSIAN_SPLATTING_CACHE_DIR", &cache_root);
        std::env::set_var(
            "BURN_GAUSSIAN_SPLATTING_YONO_BACKBONE_URL",
            format!("{base_url}/backbone"),
        );
        std::env::set_var(
            "BURN_GAUSSIAN_SPLATTING_YONO_HEAD_URL",
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

        std::env::remove_var("BURN_GAUSSIAN_SPLATTING_CACHE_DIR");
        std::env::remove_var("BURN_GAUSSIAN_SPLATTING_YONO_BACKBONE_URL");
        std::env::remove_var("BURN_GAUSSIAN_SPLATTING_YONO_HEAD_URL");

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
