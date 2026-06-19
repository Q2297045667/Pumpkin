use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use crate::locale::Locale;

static VANILLA_EN_US_JSON: &str = include_str!("../../assets/en_us_java.json");
static PUMPKIN_EN_US_JSON: &str = include_str!("../../assets/translations/en_us.json");
static PUMPKIN_BRB_JSON: &str = include_str!("../../assets/translations/brb.json");
static PUMPKIN_DE_DE_JSON: &str = include_str!("../../assets/translations/de_de.json");
static PUMPKIN_ES_ES_JSON: &str = include_str!("../../assets/translations/es_es.json");
static PUMPKIN_FR_FR_JSON: &str = include_str!("../../assets/translations/fr_fr.json");
static PUMPKING_IT_IT_JSON: &str = include_str!("../../assets/translations/it_it.json");
static PUMPKIN_JA_JP_JSON: &str = include_str!("../../assets/translations/ja_jp.json");
static PUMPKIN_KA_GE_JSON: &str = include_str!("../../assets/translations/ka_ge.json");
static PUMPKIN_KO_KR_JSON: &str = include_str!("../../assets/translations/ko_kr.json");
static PUMPKIN_NDS_DE_JSON: &str = include_str!("../../assets/translations/nds_de.json");
static PUMPKIN_NL_BE_JSON: &str = include_str!("../../assets/translations/nl_be.json");
static PUMPKIN_NL_NL_JSON: &str = include_str!("../../assets/translations/nl_nl.json");
static PUMPKIN_RO_RO_JSON: &str = include_str!("../../assets/translations/ro_ro.json");
static PUMPKIN_RU_RU_JSON: &str = include_str!("../../assets/translations/ru_ru.json");
static PUMPKIN_SQ_AL_JSON: &str = include_str!("../../assets/translations/sq_al.json");
static PUMPKIN_ZH_CN_JSON: &str = include_str!("../../assets/translations/zh_cn.json");
static PUMPKIN_ZH_HK_JSON: &str = include_str!("../../assets/translations/zh_hk.json");
static PUMPKIN_ZH_TW_JSON: &str = include_str!("../../assets/translations/zh_tw.json");
static PUMPKIN_LZH_JSON: &str = include_str!("../../assets/translations/lzh.json");
static PUMPKIN_TR_TR_JSON: &str = include_str!("../../assets/translations/tr_tr.json");
static PUMPKIN_UK_UA_JSON: &str = include_str!("../../assets/translations/uk_ua.json");
static PUMPKIN_VI_VN_JSON: &str = include_str!("../../assets/translations/vi_vn.json");
static PUMPKIN_PT_BR_JSON: &str = include_str!("../../assets/translations/pt_br.json");
static PUMPKIN_PL_PL_JSON: &str = include_str!("../../assets/translations/pl_pl.json");

pub static TRANSLATIONS: LazyLock<Mutex<[HashMap<String, String>; Locale::COUNT]>> =
    LazyLock::new(|| {
        let mut array: [HashMap<String, String>; Locale::COUNT] =
            std::array::from_fn(|_| HashMap::new());
        let vanilla_en_us: HashMap<String, String> =
            serde_json::from_str(VANILLA_EN_US_JSON).expect("Could not parse en_us_java.json.");
        let pumpkin_en_us: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_EN_US_JSON).expect("Could not parse en_us.json.");
        let pumpkin_brb: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_BRB_JSON).expect("Could not parse brb.json.");
        let pumpkin_de_de: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_DE_DE_JSON).expect("Could not parse de_de.json.");
        let pumpkin_es_es: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_ES_ES_JSON).expect("Could not parse es_es.json.");
        let pumpkin_fr_fr: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_FR_FR_JSON).expect("Could not parse fr_fr.json.");
        let pumpkin_it_it: HashMap<String, String> =
            serde_json::from_str(PUMPKING_IT_IT_JSON).expect("Could not parse it_it.json.");
        let pumpkin_ja_jp: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_JA_JP_JSON).expect("Could not parse ja_jp.json.");
        let pumpkin_ka_ge: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_KA_GE_JSON).expect("Could not parse ka_ge.json.");
        let pumpkin_ko_kr: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_KO_KR_JSON).expect("Could not parse ko_kr.json.");
        let pumpkin_nds_de: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_NDS_DE_JSON).expect("Could not parse nds_de.json.");
        let pumpkin_nl_be: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_NL_BE_JSON).expect("Could not parse nl_be.json.");
        let pumpkin_nl_nl: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_NL_NL_JSON).expect("Could not parse nl_nl.json.");
        let pumpkin_ro_ro: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_RO_RO_JSON).expect("Could not parse ro_ro.json.");
        let pumpkin_ru_ru: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_RU_RU_JSON).expect("Could not parse ru_ru.json.");
        let pumpkin_sq_al: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_SQ_AL_JSON).expect("Could not parse sq_al.json.");
        let pumpkin_zh_cn: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_ZH_CN_JSON).expect("Could not parse zh_cn.json.");
        let pumpkin_zh_hk: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_ZH_HK_JSON).expect("Could not parse zh_hk.json.");
        let pumpkin_zh_tw: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_ZH_TW_JSON).expect("Could not parse zh_tw.json.");
        let pumpkin_lzh: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_LZH_JSON).expect("Could not parse lzh.json.");
        let pumpkin_tr_tr: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_TR_TR_JSON).expect("Could not parse tr_tr.json.");
        let pumpkin_uk_ua: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_UK_UA_JSON).expect("Could not parse uk_ua.json.");
        let pumpkin_vi_vn: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_VI_VN_JSON).expect("Could not parse vi_vn.json.");
        let pumpkin_pt_br: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_PT_BR_JSON).expect("Could not parse pt_br.json.");
        let pumpkin_pl_pl: HashMap<String, String> =
            serde_json::from_str(PUMPKIN_PL_PL_JSON).expect("Could not parse pl_pl.json.");

        for (key, value) in vanilla_en_us {
            array[Locale::EnUs as usize].insert(format!("minecraft:{key}"), value);
        }
        for (key, value) in pumpkin_en_us {
            array[Locale::EnUs as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_brb {
            array[Locale::Brb as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_de_de {
            array[Locale::DeDe as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_es_es {
            array[Locale::EsEs as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_fr_fr {
            array[Locale::FrFr as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_it_it {
            array[Locale::ItIt as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_ja_jp {
            array[Locale::JaJp as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_ka_ge {
            array[Locale::KaGe as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_ko_kr {
            array[Locale::KoKr as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_nds_de {
            array[Locale::NdsDe as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_nl_be {
            array[Locale::NlBe as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_nl_nl {
            array[Locale::NlNl as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_ro_ro {
            array[Locale::RoRo as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_ru_ru {
            array[Locale::RuRu as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_sq_al {
            array[Locale::SqAl as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_zh_cn {
            array[Locale::ZhCn as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_zh_hk {
            array[Locale::ZhHk as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_zh_tw {
            array[Locale::ZhTw as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_lzh {
            array[Locale::Lzh as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_tr_tr {
            array[Locale::TrTr as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_uk_ua {
            array[Locale::UkUa as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_vi_vn {
            array[Locale::ViVn as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_pt_br {
            array[Locale::PtBr as usize].insert(format!("pumpkin:{key}"), value);
        }
        for (key, value) in pumpkin_pl_pl {
            array[Locale::PlPl as usize].insert(format!("pumpkin:{key}"), value);
        }
        Mutex::new(array)
    });

/// Adds or overrides a single translation entry.
///
/// # Arguments
/// * `namespace`: The namespace of the translation key.
/// * `key`: The translation key without namespace.
/// * `translation`: The localized translation string.
/// * `locale`: The locale the translation belongs to.
pub fn add_translation<P: Into<String>>(namespace: P, key: P, translation: P, locale: Locale) {
    let mut translations = TRANSLATIONS.lock().unwrap();
    let namespaced_key = format!("{}:{}", namespace.into(), key.into()).to_lowercase();
    translations[locale as usize].insert(namespaced_key, translation.into());
}

/// Loads translations from a JSON string and registers them under a namespace.
///
/// # Arguments
/// * `namespace`: The namespace applied to all loaded keys.
/// * `file_path`: A JSON string containing a flat key-value translation map.
/// * `locale`: The locale the translations belong to.
pub fn add_translation_file<P: Into<String>>(namespace: P, file_path: P, locale: Locale) {
    let translations_map: HashMap<String, String> =
        serde_json::from_str(&file_path.into()).unwrap_or(HashMap::new());
    if translations_map.is_empty() {
        // TODO: Handle the case where the file is empty or not found properly
        return;
    }

    let mut translations = TRANSLATIONS.lock().unwrap();
    let namespace = namespace.into();
    for (key, translation) in translations_map {
        let namespaced_key = format!("{namespace}:{key}").to_lowercase();
        translations[locale as usize].insert(namespaced_key, translation);
    }
}

/// Retrieves a translation for the given key and locale.
///
/// # Arguments
/// * `key`: The fully qualified `namespace:key`.
/// * `locale`: The requested locale.
///
/// # Returns
/// The localized translation. Falls back to `en_us` or the key itself if not found.
pub fn get_translation(key: &str, locale: Locale) -> String {
    let translations = TRANSLATIONS.lock().unwrap();
    let key = key.to_lowercase();
    translations[locale as usize].get(&key).map_or_else(
        || {
            translations[Locale::EnUs as usize]
                .get(&key)
                .map_or(key, Clone::clone)
        },
        Clone::clone,
    )
}
