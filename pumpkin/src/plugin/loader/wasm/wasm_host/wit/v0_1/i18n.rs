use crate::plugin::loader::wasm::wasm_host::{
    state::PluginHostState,
    wit::v0_1::pumpkin::plugin::{common::Locale as WitLocale, i18n::Host},
};
use pumpkin_i18n::{Locale as UtilLocale, add_translation_file, get_translation};
use std::str::FromStr;

impl Host for PluginHostState {
    async fn translate(&mut self, key: String, locale: WitLocale) -> wasmtime::Result<String> {
        let util_locale = wit_to_util_locale(locale);
        Ok(get_translation(&key, util_locale))
    }

    async fn load_translations(
        &mut self,
        namespace: String,
        json: String,
        locale: WitLocale,
    ) -> wasmtime::Result<()> {
        let util_locale = wit_to_util_locale(locale);
        add_translation_file(namespace, json, util_locale);
        Ok(())
    }
}

/// Converts a WIT Locale to a pumpkin-i18n Locale.
///
/// WIT `Debug` produces CamelCase like `"EnUs"`, `"ZhCn"`, while
/// [`UtilLocale::from_str`] expects lowercase with underscores (`"en_us"`, `"zh_cn"`).
/// We insert underscores before uppercase letters (skipping the first character),
/// then lowercase the entire string.
fn wit_to_util_locale(wit: WitLocale) -> UtilLocale {
    let raw = format!("{wit:?}");
    let normalized: String = raw
        .chars()
        .enumerate()
        .flat_map(|(i, c)| {
            if i > 0 && c.is_uppercase() {
                vec!['_', c.to_ascii_lowercase()]
            } else {
                vec![c.to_ascii_lowercase()]
            }
        })
        .collect();
    UtilLocale::from_str(&normalized).unwrap_or(UtilLocale::EnUs)
}
