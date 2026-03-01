use std::env;
use std::process::Command;

const FALLBACK_REV: &str = "unknown";

fn git_short_rev() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let rev = String::from_utf8(output.stdout).ok()?;
    let rev = rev.trim();
    if rev.is_empty() {
        None
    } else {
        Some(rev.chars().take(7).collect())
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BURN_RECONSTRUCTION_GIT_SHA");
    println!("cargo:rerun-if-changed=../../.git/HEAD");

    let rev = env::var("BURN_RECONSTRUCTION_GIT_SHA")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(git_short_rev)
        .unwrap_or_else(|| FALLBACK_REV.to_string());

    let version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    let build_label = format!("v{version}+{rev}");
    let long_version = format!("{version}\nrev: {rev}");

    println!("cargo:rustc-env=BURN_RECONSTRUCTION_GIT_SHA={rev}");
    println!("cargo:rustc-env=BURN_RECONSTRUCTION_BUILD_LABEL={build_label}");
    println!("cargo:rustc-env=BURN_RECONSTRUCTION_LONG_VERSION={long_version}");
}
