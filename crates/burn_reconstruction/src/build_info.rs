pub const PKG_NAME: &str = env!("CARGO_PKG_NAME");
pub const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_REV_SHORT: &str = match option_env!("BURN_RECONSTRUCTION_GIT_SHA") {
    Some(value) => value,
    None => "unknown",
};
pub const BUILD_LABEL: &str = match option_env!("BURN_RECONSTRUCTION_BUILD_LABEL") {
    Some(value) => value,
    None => "dev",
};
pub const LONG_VERSION: &str = match option_env!("BURN_RECONSTRUCTION_LONG_VERSION") {
    Some(value) => value,
    None => PKG_VERSION,
};

pub const fn git_revision_short() -> &'static str {
    GIT_REV_SHORT
}

pub const fn build_label() -> &'static str {
    BUILD_LABEL
}

pub const fn long_version() -> &'static str {
    LONG_VERSION
}

pub fn app_banner(app_name: &str) -> String {
    format!("{app_name} {PKG_VERSION} (rev {GIT_REV_SHORT})")
}
