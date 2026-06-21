use pumpkin_i18n::get_translation;
use pumpkin_nbt::compound::NbtCompound;
use pumpkin_util::text::translation::get_translation_text;
use std::fs::{File, create_dir_all};
use std::io;
use std::path::PathBuf;
use tracing::{debug, error};
use uuid::Uuid;

/// Manages the storage and retrieval of player data from disk and memory cache.
///
/// This struct provides functions to load and save player data to/from NBT files,
/// with a memory cache to handle player disconnections temporarily.
pub struct PlayerDataStorage {
    /// Path to the directory where player data is stored
    data_path: PathBuf,
    /// Whether player data saving is enabled
    save_enabled: bool,
}

#[derive(Debug)]
pub enum PlayerDataError {
    Io(io::Error),
    Nbt(String),
}

impl From<io::Error> for PlayerDataError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl std::fmt::Display for PlayerDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let locale = crate::server_locale();
        match self {
            Self::Io(e) => {
                write!(
                    f,
                    "{}",
                    get_translation("pumpkin:world.player.io_error", locale)
                        .replace("%s", &e.to_string())
                )
            }
            Self::Nbt(e) => {
                write!(
                    f,
                    "{}",
                    get_translation("pumpkin:world.player.nbt_error", locale).replace("%s", e)
                )
            }
        }
    }
}

impl std::error::Error for PlayerDataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Nbt(_) => None,
        }
    }
}

impl PlayerDataStorage {
    /// Creates a new `PlayerDataStorage` with the specified data path and cache expiration time.
    pub fn new(data_path: impl Into<PathBuf>, enabled: bool) -> Self {
        let path = data_path.into();
        if !path.exists()
            && let Err(e) = create_dir_all(&path)
        {
            error!(
                "{}",
                get_translation_text(
                    "pumpkin:world.player.failed_create_dir",
                    crate::server_locale(),
                    vec![
                        pumpkin_util::text::TextComponent::text(format!("{}", path.display())).0,
                        pumpkin_util::text::TextComponent::text(e.to_string()).0,
                    ],
                )
            );
        }

        Self {
            data_path: path,
            save_enabled: enabled,
        }
    }

    #[must_use]
    pub const fn get_data_path(&self) -> &PathBuf {
        &self.data_path
    }

    #[must_use]
    pub const fn is_save_enabled(&self) -> bool {
        self.save_enabled
    }

    pub const fn set_save_enabled(&mut self, enabled: bool) {
        self.save_enabled = enabled;
    }

    /// Returns the path for a player's data file based on their UUID.
    #[must_use]
    pub fn get_player_data_path(&self, uuid: &Uuid) -> PathBuf {
        self.get_data_path().join(format!("{uuid}.dat"))
    }

    /// Loads player data from NBT file or cache.
    ///
    /// This function first checks if player data exists in the cache.
    /// If not, it attempts to load the data from a .dat file on disk.
    ///
    /// # Arguments
    ///
    /// * `uuid` - The UUID of the player to load data for.
    ///
    /// # Returns
    ///
    /// A Result containing either the player's NBT data or an error.
    pub fn load_player_data(&self, uuid: &Uuid) -> Result<(bool, NbtCompound), PlayerDataError> {
        // If player data saving is disabled, return empty data
        if !self.is_save_enabled() {
            return Ok((false, NbtCompound::new()));
        }

        // If not in cache, load from disk
        let path = self.get_player_data_path(uuid);
        if !path.exists() {
            debug!(
                "{}",
                get_translation_text(
                    "pumpkin:world.player.no_player_data_file",
                    crate::server_locale(),
                    vec![pumpkin_util::text::TextComponent::text(uuid.to_string()).0]
                )
            );
            return Ok((false, NbtCompound::new()));
        }

        let file = match File::open(&path) {
            Ok(file) => file,
            Err(e) => {
                error!(
                    "{}",
                    get_translation_text(
                        "pumpkin:world.player.failed_open_file",
                        crate::server_locale(),
                        vec![
                            pumpkin_util::text::TextComponent::text(uuid.to_string()).0,
                            pumpkin_util::text::TextComponent::text(e.to_string()).0
                        ]
                    )
                );
                return Err(PlayerDataError::Io(e));
            }
        };

        match pumpkin_nbt::nbt_compress::read_gzip_compound_tag(file) {
            Ok(nbt) => {
                debug!(
                    "{}",
                    get_translation_text(
                        "pumpkin:world.player.loaded_player_data",
                        crate::server_locale(),
                        vec![pumpkin_util::text::TextComponent::text(uuid.to_string()).0]
                    )
                );
                Ok((true, nbt))
            }
            Err(e) => {
                error!(
                    "{}",
                    get_translation_text(
                        "pumpkin:world.player.failed_read_data",
                        crate::server_locale(),
                        vec![
                            pumpkin_util::text::TextComponent::text(uuid.to_string()).0,
                            pumpkin_util::text::TextComponent::text(e.to_string()).0
                        ]
                    )
                );
                Err(PlayerDataError::Nbt(e.to_string()))
            }
        }
    }

    /// Saves player data to NBT file and updates cache.
    ///
    /// This function saves the player's data to a .dat file on disk and also
    /// updates the in-memory cache with the latest data.
    ///
    /// # Arguments
    ///
    /// * `uuid` - The UUID of the player to save data for.
    /// * `data` - The NBT compound data to save.
    ///
    /// # Returns
    ///
    /// A Result indicating success or the error that occurred.
    pub fn save_player_data(&self, uuid: &Uuid, data: NbtCompound) -> Result<(), PlayerDataError> {
        // Skip saving if disabled in config
        if !self.is_save_enabled() {
            return Ok(());
        }

        let path = self.get_player_data_path(uuid);

        // Ensure parent directory exists
        if let Some(parent) = path.parent()
            && let Err(e) = create_dir_all(parent)
        {
            error!(
                "{}",
                get_translation_text(
                    "pumpkin:world.player.failed_create_dir",
                    crate::server_locale(),
                    vec![
                        pumpkin_util::text::TextComponent::text(uuid.to_string()).0,
                        pumpkin_util::text::TextComponent::text(e.to_string()).0
                    ]
                )
            );
            return Err(PlayerDataError::Io(e));
        }

        // Create the file and write directly with GZip compression
        match File::create(&path) {
            Ok(file) => {
                if let Err(e) = pumpkin_nbt::nbt_compress::write_gzip_compound_tag(data, file) {
                    error!(
                        "{}",
                        get_translation_text(
                            "pumpkin:world.player.failed_write_compressed",
                            crate::server_locale(),
                            vec![
                                pumpkin_util::text::TextComponent::text(uuid.to_string()).0,
                                pumpkin_util::text::TextComponent::text(e.to_string()).0
                            ]
                        )
                    );
                    Err(PlayerDataError::Nbt(e.to_string()))
                } else {
                    debug!(
                        "{}",
                        get_translation_text(
                            "pumpkin:world.player.saved_player_data",
                            crate::server_locale(),
                            vec![pumpkin_util::text::TextComponent::text(uuid.to_string()).0]
                        )
                    );
                    Ok(())
                }
            }
            Err(e) => {
                error!(
                    "{}",
                    get_translation_text(
                        "pumpkin:world.player.failed_create_file",
                        crate::server_locale(),
                        vec![
                            pumpkin_util::text::TextComponent::text(uuid.to_string()).0,
                            pumpkin_util::text::TextComponent::text(e.to_string()).0
                        ]
                    )
                );
                Err(PlayerDataError::Io(e))
            }
        }
    }
}
