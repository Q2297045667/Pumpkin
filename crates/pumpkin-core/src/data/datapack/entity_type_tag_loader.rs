use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

use pumpkin_data::entity::EntityType;
use pumpkin_data::tag::{self, RegistryKey};
use pumpkin_util::identifier::Identifier;
use serde::Deserialize;
use tracing::warn;

use super::LoadedDatapack;

pub(super) type EntityTypeTagRegistry = HashMap<String, HashSet<u16>>;
type TagDefinitions = BTreeMap<String, Vec<TagEntry>>;
type Dependencies<'a> = HashMap<&'a str, BTreeSet<&'a str>>;

const MAX_TAG_FILE_BYTES: usize = 1024 * 1024;
const MAX_TAG_DATA_BYTES: usize = 16 * 1024 * 1024;
const MAX_TAG_FILES: usize = 4096;
const MAX_TAG_ENTRIES: usize = 65_536;
const MAX_DIRECTORY_ENTRIES: usize = 32_768;
const MAX_DIRECTORY_DEPTH: usize = 32;
const MAX_IDENTIFIER_BYTES: usize = 1024;
const MAX_REFERENCE_DEPTH: usize = 128;
const MAX_DEPENDENCY_STEPS: usize = 4_000_000;

#[derive(Deserialize)]
struct TagFile {
    #[serde(default)]
    replace: bool,
    values: Vec<TagFileEntry>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TagFileEntry {
    Id(String),
    Object {
        id: String,
        #[serde(default = "required_by_default")]
        required: bool,
    },
}

const fn required_by_default() -> bool {
    true
}

struct TagEntry {
    id: String,
    tag: bool,
    required: bool,
}

#[derive(Default)]
struct LoadBudget {
    directory_entries: usize,
    files: usize,
    bytes: usize,
    entries: usize,
}

impl LoadBudget {
    fn visit_directory_entry(&mut self) -> Result<(), String> {
        self.directory_entries += 1;
        if self.directory_entries > MAX_DIRECTORY_ENTRIES {
            return Err("Too many entity type tag directory entries".to_string());
        }
        Ok(())
    }
}

fn parse_identifier(id: &str) -> Result<String, String> {
    if id.len() > MAX_IDENTIFIER_BYTES {
        return Err("Entity type tag identifier is too long".to_string());
    }
    Identifier::parse(id)
        .map(|id| id.to_string())
        .map_err(|error| error.to_string())
}

fn parse_tag_file(content: &[u8]) -> Result<(bool, Vec<TagEntry>), String> {
    let file: TagFile = serde_json::from_slice(content).map_err(|error| error.to_string())?;
    if file.values.len() > MAX_TAG_ENTRIES {
        return Err("Too many entity type tag values".to_string());
    }
    let entries = file
        .values
        .into_iter()
        .map(|entry| {
            let (id, required) = match entry {
                TagFileEntry::Id(id) => (id, true),
                TagFileEntry::Object { id, required } => (id, required),
            };
            let tag = id.starts_with('#');
            Ok(TagEntry {
                id: parse_identifier(id.strip_prefix('#').unwrap_or(&id))?,
                tag,
                required,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok((file.replace, entries))
}

fn default_definitions() -> Result<TagDefinitions, String> {
    let mut definitions = TagDefinitions::new();
    for (&id, values) in tag::get_latest_map(RegistryKey::EntityType).entries() {
        let entries = values
            .0
            .iter()
            .map(|value| {
                Ok(TagEntry {
                    id: parse_identifier(value)?,
                    tag: false,
                    required: true,
                })
            })
            .collect::<Result<_, String>>()?;
        definitions.insert(id.to_string(), entries);
    }

    // The generated map flattens references; retain the assets' dependency edges.
    macro_rules! referenced_defaults {
        ($root:literal, $namespace:literal, $($name:literal),+ $(,)?) => {
            $(
                let (_, entries) = parse_tag_file(include_bytes!(concat!(
                    "../../../../../assets/", $root, "/data/", $namespace,
                    "/tags/entity_type/", $name, ".json"
                )))?;
                definitions.insert(concat!($namespace, ":", $name).to_string(), entries);
            )+
        };
    }
    referenced_defaults!(
        "datapack",
        "minecraft",
        "can_breathe_under_water",
        "candidate_for_iron_golem_gift",
        "ignores_poison_and_regen",
        "illager_friends",
        "impact_projectiles",
        "inverted_healing_and_harm",
        "sensitive_to_bane_of_arthropods",
        "sensitive_to_impaling",
        "sensitive_to_smite",
        "undead",
        "wither_friends",
    );
    referenced_defaults!("conventional_tags", "c", "boats");
    Ok(definitions)
}

pub(super) fn load(
    loaded_packs: &[LoadedDatapack],
    enabled_packs: &[String],
) -> Result<EntityTypeTagRegistry, String> {
    let defaults = default_definitions()?;
    let (mut definitions, mut defaults) = if enabled_packs.iter().any(|pack| pack == "vanilla") {
        (TagDefinitions::new(), Some(defaults))
    } else {
        (defaults, None)
    };
    let mut budget = LoadBudget::default();
    let mut visited_packs = HashSet::new();
    // Pack discovery uses filesystem order; only this registry follows enabled priority.
    for enabled in enabled_packs {
        if enabled == "vanilla" {
            if let Some(defaults) = defaults.take() {
                for (id, entries) in defaults {
                    definitions.entry(id).or_default().extend(entries);
                }
            }
            continue;
        }
        if let Some(pack) = loaded_packs
            .iter()
            .find(|pack| pack.id == *enabled || pack.name == *enabled)
            && visited_packs.insert(pack.id.as_str())
        {
            load_pack(&pack.root_path, &mut definitions, &mut budget)?;
        }
    }
    build(&definitions)
}

fn load_pack(
    pack: &Path,
    definitions: &mut TagDefinitions,
    budget: &mut LoadBudget,
) -> Result<(), String> {
    let namespaces = match fs::read_dir(pack.join("data")) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.to_string()),
    };
    for entry in namespaces {
        budget.visit_directory_entry()?;
        let entry = entry.map_err(|error| error.to_string())?;
        if !entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_dir()
        {
            continue;
        }
        let namespace = entry.file_name().to_string_lossy().into_owned();
        if Identifier::new(namespace.clone(), "").is_err() {
            warn!("Invalid entity type tag namespace '{namespace}'");
            continue;
        }
        let dir = entry.path().join("tags/entity_type");
        match fs::symlink_metadata(&dir) {
            Ok(metadata) if metadata.is_dir() => {
                load_directory(&namespace, &dir, &dir, 0, definitions, budget)?;
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("{}: {error}", dir.display())),
        }
    }
    Ok(())
}

fn load_directory(
    namespace: &str,
    base: &Path,
    dir: &Path,
    depth: usize,
    definitions: &mut TagDefinitions,
    budget: &mut LoadBudget,
) -> Result<(), String> {
    if depth > MAX_DIRECTORY_DEPTH {
        return Err("Entity type tag directory nesting is too deep".to_string());
    }
    for entry in fs::read_dir(dir).map_err(|error| error.to_string())? {
        budget.visit_directory_entry()?;
        let entry = entry.map_err(|error| error.to_string())?;
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        let path = entry.path();
        if kind.is_dir() {
            load_directory(namespace, base, &path, depth + 1, definitions, budget)?;
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "json") {
            budget.files += 1;
            if budget.files > MAX_TAG_FILES {
                return Err("Too many entity type tag files".to_string());
            }
            let relative = path.strip_prefix(base).map_err(|error| error.to_string())?;
            let name = relative
                .with_extension("")
                .to_string_lossy()
                .replace('\\', "/");
            let parsed: Result<_, String> = (|| {
                let id = parse_identifier(&format!("{namespace}:{name}"))?;
                let content = read_tag_file(&path)?;
                budget.bytes += content.len();
                if budget.bytes > MAX_TAG_DATA_BYTES {
                    return Err("Entity type tag data is too large".to_string());
                }
                let (replace, entries) = parse_tag_file(&content)?;
                Ok((id, replace, entries))
            })();
            match parsed {
                Ok((id, replace, entries)) => {
                    budget.entries += entries.len();
                    if budget.entries > MAX_TAG_ENTRIES {
                        return Err("Too many entity type tag values".to_string());
                    }
                    let values = definitions.entry(id).or_default();
                    if replace {
                        values.clear();
                    }
                    values.extend(entries);
                }
                Err(error) => warn!(
                    "Failed to read entity type tag '{}': {error}",
                    path.display()
                ),
            }
            if budget.bytes > MAX_TAG_DATA_BYTES {
                return Err("Entity type tag data is too large".to_string());
            }
        }
    }
    Ok(())
}

fn read_tag_file(path: &Path) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    if file.metadata().map_err(|error| error.to_string())?.len() > MAX_TAG_FILE_BYTES as u64 {
        return Err("Entity type tag file is too large".to_string());
    }
    let mut content = Vec::new();
    file.take(MAX_TAG_FILE_BYTES as u64 + 1)
        .read_to_end(&mut content)
        .map_err(|error| error.to_string())?;
    if content.len() > MAX_TAG_FILE_BYTES {
        return Err("Entity type tag file is too large".to_string());
    }
    Ok(content)
}

fn is_cyclic<'a>(
    dependencies: &Dependencies<'a>,
    from: &'a str,
    to: &'a str,
    steps: &mut usize,
) -> Result<bool, String> {
    let mut pending = vec![to];
    let mut visited = HashSet::new();
    while let Some(id) = pending.pop() {
        *steps += 1;
        if *steps > MAX_DEPENDENCY_STEPS {
            return Err("Entity type tag dependency graph is too complex".to_string());
        }
        if id == from {
            return Ok(true);
        }
        if visited.insert(id)
            && let Some(children) = dependencies.get(id)
        {
            pending.extend(children.iter().copied());
        }
    }
    Ok(false)
}

fn visit_dependencies<'a>(
    id: &'a str,
    dependencies: &Dependencies<'a>,
    visited: &mut HashSet<&'a str>,
    order: &mut Vec<&'a str>,
    depth: usize,
) -> Result<(), String> {
    if !visited.insert(id) {
        return Ok(());
    }
    if depth > MAX_REFERENCE_DEPTH {
        return Err("Entity type tag references are nested too deeply".to_string());
    }
    if let Some(children) = dependencies.get(id) {
        for child in children {
            visit_dependencies(child, dependencies, visited, order, depth + 1)?;
        }
    }
    order.push(id);
    Ok(())
}

fn build(definitions: &TagDefinitions) -> Result<EntityTypeTagRegistry, String> {
    let mut dependencies = Dependencies::new();
    let mut steps = 0;
    // DependencySorter adds required edges before optional ones, dropping cyclic edges.
    for required in [true, false] {
        for (id, entries) in definitions {
            for entry in entries {
                if entry.tag
                    && entry.required == required
                    && definitions.contains_key(&entry.id)
                    && !is_cyclic(&dependencies, id, &entry.id, &mut steps)?
                {
                    dependencies.entry(id).or_default().insert(&entry.id);
                }
            }
        }
    }
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    for id in definitions.keys() {
        visit_dependencies(id, &dependencies, &mut visited, &mut order, 0)?;
    }

    let mut registry = EntityTypeTagRegistry::new();
    for id in order {
        let mut values = HashSet::new();
        let mut missing = false;
        for entry in &definitions[id] {
            let found = if entry.tag {
                registry.get(&entry.id).is_some_and(|referenced| {
                    values.extend(referenced);
                    true
                })
            } else if let Some(name) = entry.id.strip_prefix("minecraft:")
                && let Some(entity_type) = EntityType::from_name(name)
            {
                values.insert(entity_type.id);
                true
            } else {
                false
            };
            if !found && entry.required {
                warn!(
                    "Entity type tag '{id}' is missing required reference '{}'",
                    entry.id
                );
                missing = true;
            }
        }
        if !missing {
            registry.insert(id.to_string(), values);
        }
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;
    use crate::data::datapack::DatapackManager;
    use crate::server::recipe::RecipeManager;

    const REDIRECTABLE: &str = "minecraft:redirectable_projectile";

    fn write_tag(pack: &Path, id: &str, content: &str) -> io::Result<()> {
        let (namespace, name) = id
            .split_once(':')
            .ok_or_else(|| io::Error::other("test tag must have a namespace"))?;
        let path = pack
            .join("data")
            .join(namespace)
            .join("tags/entity_type")
            .join(format!("{name}.json"));
        let parent = path
            .parent()
            .ok_or_else(|| io::Error::other("tag parent"))?;
        fs::create_dir_all(parent)?;
        fs::write(path, content)
    }

    fn pack(root: &Path, name: &str) -> LoadedDatapack {
        LoadedDatapack {
            id: format!("file/{name}"),
            name: name.to_string(),
            description: String::new(),
            pack_format: 61,
            root_path: root.join(name),
            recipe_count: 0,
            function_count: 0,
            known_packs: Vec::new(),
        }
    }

    #[test]
    fn packs_replace_append_and_resolve_final_namespaced_references() -> Result<(), Box<dyn Error>>
    {
        let temp = tempfile::tempdir()?;
        let low = pack(temp.path(), "low");
        let middle = pack(temp.path(), "middle");
        let high = pack(temp.path(), "high");
        write_tag(&low.root_path, REDIRECTABLE, r#"{"values":["egg"]}"#)?;
        write_tag(
            &low.root_path,
            "example:group/projectiles",
            r#"{"values":["arrow"]}"#,
        )?;
        let appended = load(std::slice::from_ref(&low), std::slice::from_ref(&low.id))?;
        assert!(appended[REDIRECTABLE].contains(&EntityType::FIREBALL.id));
        assert!(appended[REDIRECTABLE].contains(&EntityType::EGG.id));

        write_tag(
            &middle.root_path,
            REDIRECTABLE,
            r##"{"replace":true,"values":["#example:outer"]}"##,
        )?;
        write_tag(
            &middle.root_path,
            "example:outer",
            r##"{"values":["#example:group/projectiles","trident"]}"##,
        )?;
        write_tag(
            &high.root_path,
            "example:group/projectiles",
            r#"{"replace":true,"values":["snowball"]}"#,
        )?;
        write_tag(&high.root_path, REDIRECTABLE, r#"{"values":["egg","egg"]}"#)?;
        // Discovery order is deliberately different from enabled pack priority.
        let enabled = vec![low.id.clone(), middle.id.clone(), high.id.clone()];
        let loaded = [high, low, middle];
        let tags = load(&loaded, &enabled)?;
        assert_eq!(
            tags[REDIRECTABLE],
            HashSet::from([
                EntityType::SNOWBALL.id,
                EntityType::TRIDENT.id,
                EntityType::EGG.id
            ])
        );
        assert!(!tags[REDIRECTABLE].contains(&EntityType::FIREBALL.id));
        assert!(!tags[REDIRECTABLE].contains(&EntityType::ARROW.id));
        Ok(())
    }

    #[test]
    fn vanilla_respects_its_enabled_pack_priority() -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let custom = pack(temp.path(), "custom");
        write_tag(
            &custom.root_path,
            REDIRECTABLE,
            r#"{"replace":true,"values":[]}"#,
        )?;
        let enabled = [custom.id.clone(), "vanilla".to_string()];
        let tags = load(std::slice::from_ref(&custom), &enabled)?;
        assert!(tags[REDIRECTABLE].contains(&EntityType::FIREBALL.id));
        let enabled = ["vanilla".to_string(), custom.id.clone()];
        let tags = load(std::slice::from_ref(&custom), &enabled)?;
        assert!(tags[REDIRECTABLE].is_empty());
        Ok(())
    }

    #[test]
    fn optional_missing_entries_are_skipped_but_required_missing_invalidates_the_tag()
    -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let custom = pack(temp.path(), "custom");
        write_tag(
            &custom.root_path,
            REDIRECTABLE,
            r##"{"replace":true,"values":[
                {"id":"example:arrow","required":false},
                {"id":"minecraft:not_an_entity","required":false},
                {"id":"#example:missing","required":false},
                "snowball"
            ]}"##,
        )?;
        let tags = load(
            std::slice::from_ref(&custom),
            std::slice::from_ref(&custom.id),
        )?;
        assert_eq!(tags[REDIRECTABLE], HashSet::from([EntityType::SNOWBALL.id]));
        write_tag(
            &custom.root_path,
            REDIRECTABLE,
            r##"{"replace":true,"values":["snowball","#example:missing"]}"##,
        )?;
        let tags = load(
            std::slice::from_ref(&custom),
            std::slice::from_ref(&custom.id),
        )?;
        assert!(!tags.contains_key(REDIRECTABLE));
        write_tag(
            &custom.root_path,
            REDIRECTABLE,
            r#"{"replace":true,"values":[{"id":"example:arrow"}]}"#,
        )?;
        let tags = load(
            std::slice::from_ref(&custom),
            std::slice::from_ref(&custom.id),
        )?;
        assert!(!tags.contains_key(REDIRECTABLE));
        Ok(())
    }

    #[test]
    fn cycles_fail_required_tags_and_optional_edges_do_not_displace_required_edges()
    -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let custom = pack(temp.path(), "custom");
        write_tag(
            &custom.root_path,
            "example:a",
            r##"{"values":["#example:b"]}"##,
        )?;
        write_tag(
            &custom.root_path,
            "example:b",
            r##"{"values":["#example:a"]}"##,
        )?;
        write_tag(
            &custom.root_path,
            REDIRECTABLE,
            r##"{"replace":true,"values":["#example:a"]}"##,
        )?;
        let tags = load(
            std::slice::from_ref(&custom),
            std::slice::from_ref(&custom.id),
        )?;
        assert!(!tags.contains_key("example:a"));
        assert!(!tags.contains_key("example:b"));
        assert!(!tags.contains_key(REDIRECTABLE));

        write_tag(
            &custom.root_path,
            "example:a",
            r##"{"values":[{"id":"#example:b","required":false},"arrow"]}"##,
        )?;
        let tags = load(
            std::slice::from_ref(&custom),
            std::slice::from_ref(&custom.id),
        )?;
        assert_eq!(tags["example:a"], HashSet::from([EntityType::ARROW.id]));
        assert_eq!(tags["example:b"], HashSet::from([EntityType::ARROW.id]));
        assert_eq!(tags[REDIRECTABLE], HashSet::from([EntityType::ARROW.id]));
        Ok(())
    }

    #[test]
    fn overriding_a_default_reference_updates_referencing_tags() -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let custom = pack(temp.path(), "custom");
        write_tag(
            &custom.root_path,
            "minecraft:arrows",
            r#"{"replace":true,"values":["snowball"]}"#,
        )?;
        write_tag(
            &custom.root_path,
            REDIRECTABLE,
            r##"{"replace":true,"values":["#minecraft:impact_projectiles"]}"##,
        )?;
        let tags = load(
            std::slice::from_ref(&custom),
            std::slice::from_ref(&custom.id),
        )?;
        assert!(tags[REDIRECTABLE].contains(&EntityType::SNOWBALL.id));
        assert!(!tags[REDIRECTABLE].contains(&EntityType::ARROW.id));
        assert!(!tags[REDIRECTABLE].contains(&EntityType::SPECTRAL_ARROW.id));
        Ok(())
    }

    #[test]
    fn empty_replace_survives_lookup_and_disabled_pack_reload_restores_defaults()
    -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let custom = temp.path().join("datapacks/custom");
        write_tag(&custom, REDIRECTABLE, r#"{"replace":true,"values":[]}"#)?;
        let manager = DatapackManager::new();
        let recipes = RecipeManager::new();
        assert_eq!(
            manager.is_entity_type_tagged(&EntityType::FIREBALL, REDIRECTABLE),
            None
        );
        manager.load_all(temp.path(), &["file/custom".to_string()], &recipes);
        assert_eq!(
            manager.is_entity_type_tagged(&EntityType::FIREBALL, REDIRECTABLE),
            Some(false)
        );
        let mut too_deep = custom.join("data/example/tags/entity_type");
        for _ in 0..=MAX_DIRECTORY_DEPTH {
            too_deep = too_deep.join("nested");
        }
        fs::create_dir_all(too_deep)?;
        manager.load_all(temp.path(), &["file/custom".to_string()], &recipes);
        assert_eq!(
            manager.is_entity_type_tagged(&EntityType::FIREBALL, REDIRECTABLE),
            Some(false)
        );
        manager.load_all(temp.path(), &[], &recipes);
        assert_eq!(
            manager.is_entity_type_tagged(&EntityType::FIREBALL, REDIRECTABLE),
            Some(true)
        );
        let snapshot = manager
            .entity_type_tags
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tags = snapshot.as_ref().ok_or("default tags were not loaded")?;
        let generated = tag::get_latest_map(RegistryKey::EntityType);
        assert_eq!(tags.len(), generated.len());
        for (&id, values) in generated.entries() {
            assert_eq!(
                tags.get(id),
                Some(&values.1.iter().copied().collect::<HashSet<_>>()),
                "{id}"
            );
        }
        Ok(())
    }

    #[test]
    fn oversized_and_malformed_files_do_not_apply_replace() -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let custom = pack(temp.path(), "custom");
        write_tag(
            &custom.root_path,
            REDIRECTABLE,
            r#"{"replace":true,"values":false}"#,
        )?;
        let tags = load(
            std::slice::from_ref(&custom),
            std::slice::from_ref(&custom.id),
        )?;
        assert!(tags[REDIRECTABLE].contains(&EntityType::FIREBALL.id));
        let path = custom
            .root_path
            .join("data/minecraft/tags/entity_type/redirectable_projectile.json");
        File::create(path)?.set_len(MAX_TAG_FILE_BYTES as u64 + 1)?;
        let tags = load(
            std::slice::from_ref(&custom),
            std::slice::from_ref(&custom.id),
        )?;
        assert!(tags[REDIRECTABLE].contains(&EntityType::FIREBALL.id));
        Ok(())
    }
}
