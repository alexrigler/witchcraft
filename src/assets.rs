//! assets.rs — Simple asset file loader with lazy caching.

use once_cell::sync::OnceCell;
use std::{
    fs,
    path::PathBuf,
};

/// An asset file that is loaded at most once and cached for the program's lifetime.
pub struct Asset {
    path: &'static str,
    cached: OnceCell<Result<Vec<u8>, ()>>,
}

impl Asset {
    pub const fn new_file(path: &'static str) -> Self {
        Self {
            path,
            cached: OnceCell::new(),
        }
    }

    /// Return the file bytes; loads and caches on first call.
    pub fn bytes(&'static self, assets: &PathBuf) -> Result<&'static [u8], ()> {
        self.cached
            .get_or_init(|| {
                match fs::read(assets.join(self.path)) {
                    Ok(bytes) => Ok(bytes),
                    Err(e) => {
                        log::warn!("failed to read warp asset {}: {}", self.path, e);
                        Err(())
                    }
                }
            })
            .as_deref()
            .map_err(|_| ())
    }
}

#[macro_export]
macro_rules! embed_asset {
    ($vis:vis $name:ident, $path:literal) => {
        $vis static $name: $crate::assets::Asset =
            $crate::assets::Asset::new_file($path);
    };
}
