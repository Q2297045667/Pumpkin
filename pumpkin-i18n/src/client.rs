use std::str::FromStr;

use crate::locale::Locale;

/// Resolves the client locale for a player based on the configuration value
/// and the locale reported by the player's client.
///
/// # Arguments
/// * `player_locale` — The locale string reported by the client (e.g. `"en_us"`, `"zh_cn"`).
/// * `config_value` — The locale configuration value, either `"auto"` or a specific locale code.
///
/// # Returns
/// The resolved [`Locale`]. If `config_value` is `"auto"`, returns the player's locale.
/// Otherwise overrides with the configured locale.
#[must_use]
pub fn resolve_client_locale(player_locale: &str, config_value: &str) -> Locale {
    if config_value != "auto" {
        let normalized = config_value.replace('-', "_");
        return Locale::from_str(&normalized).unwrap_or(Locale::EnUs);
    }
    let normalized = player_locale.replace('-', "_");
    Locale::from_str(&normalized).unwrap_or(Locale::EnUs)
}

/// Formats a player join message that includes their detected locale.
///
/// # Arguments
/// * `player_name` — The player's display name.
/// * `locale` — The player's resolved locale.
///
/// # Returns
/// A log-friendly message like `"PlayerName joined the game language:zh_cn"`.
#[must_use]
pub fn format_join_locale(player_name: &str, locale: Locale) -> String {
    format!("{player_name} joined the game language:{locale:?}")
}

/// Returns the locale string in lowercase underscore format for logging.
///
/// # Arguments
/// * `locale` — The locale to format.
///
/// # Returns
/// A string like `"en_us"`, `"zh_cn"`, etc.
#[must_use]
pub fn locale_to_log_string(locale: Locale) -> String {
    format!("{locale:?}").to_lowercase()
}
