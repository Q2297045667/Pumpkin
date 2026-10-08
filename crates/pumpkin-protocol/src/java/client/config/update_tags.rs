use std::io::Write;

use crate::{ClientPacket, WritingError, ser::NetworkWriteExt};

use crate::codec::var_int::VarInt;
use pumpkin_data::{
    packet::clientbound::config::UPDATE_TAGS,
    tag::{RegistryKey, get_registry_key_tags},
};
use pumpkin_macros::java_packet;
use pumpkin_util::version::JavaMinecraftVersion;

#[java_packet(UPDATE_TAGS)]
pub struct CUpdateTags<'a> {
    pub tags: &'a [pumpkin_data::tag::RegistryKey],
}

impl<'a> CUpdateTags<'a> {
    #[must_use]
    pub const fn new(tags: &'a [RegistryKey]) -> Self {
        Self { tags }
    }
}

impl ClientPacket for CUpdateTags<'_> {
    fn write_packet_data(
        &self,
        write: impl Write,
        version: &JavaMinecraftVersion,
    ) -> Result<(), WritingError> {
        write_tags(self.tags, None, write, *version)
    }
}

pub use dynamic::CUpdateTagsWithEntityTypes;

mod dynamic {
    use super::{
        CUpdateTags, ClientPacket, JavaMinecraftVersion, RegistryKey, WritingError, write_tags,
    };
    use std::io::Write;

    /// Sends tags with the server's current datapack entity type snapshot.
    pub struct CUpdateTagsWithEntityTypes<'a> {
        pub tags: &'a [RegistryKey],
        pub entity_types: &'a std::collections::BTreeMap<String, Vec<u16>>,
    }

    impl crate::packet::MultiVersionJavaPacket for CUpdateTagsWithEntityTypes<'_> {
        fn to_id(version: JavaMinecraftVersion) -> i32 {
            CUpdateTags::to_id(version)
        }
    }

    impl ClientPacket for CUpdateTagsWithEntityTypes<'_> {
        fn write_packet_data(
            &self,
            write: impl Write,
            version: &JavaMinecraftVersion,
        ) -> Result<(), WritingError> {
            write_tags(self.tags, Some(self.entity_types), write, *version)
        }
    }
}

/// Writes registry tags, optionally replacing the entity type registry snapshot.
pub fn write_tags(
    tags: &[RegistryKey],
    entity_types: Option<&std::collections::BTreeMap<String, Vec<u16>>>,
    mut write: impl Write,
    version: JavaMinecraftVersion,
) -> Result<(), WritingError> {
    let valid_keys: Vec<_> = tags
        .iter()
        .copied()
        .filter(|key| key.is_valid_for_version(version))
        .collect();

    write.write_list(&valid_keys, |p, &registry_key| {
        p.write_string(&format!("minecraft:{}", registry_key.identifier_string()))?;
        if registry_key == RegistryKey::EntityType
            && let Some(entity_types) = entity_types
        {
            p.write_var_int(&VarInt(entity_types.len().try_into().map_err(|_| {
                WritingError::Message("Too many entity type tags".to_string())
            })?))?;
            for (name, values) in entity_types {
                p.write_string_bounded(name, u16::MAX as usize)?;
                p.write_list(values, |p, &id| p.write_var_int(&VarInt::from(id)))?;
            }
            return Ok(());
        }
        let Some(values) = get_registry_key_tags(version, registry_key) else {
            // no tags defined for that registry key in this version
            // write an empty list and continue
            p.write_var_int(&VarInt::from(0))?;
            return Ok(());
        };
        p.write_var_int(&values.len().try_into().map_err(|_| {
            WritingError::Message(format!("{} isn't representable as a VarInt", values.len()))
        })?)?;

        for (key, values) in values.entries() {
            // This is technically a `ResourceLocation` but same thing
            p.write_string_bounded(key, u16::MAX as usize)?;
            p.write_list(values.1, |p, &id| p.write_var_int(&VarInt::from(id)))?;
        }

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::{CUpdateTagsWithEntityTypes, RegistryKey};
    use crate::ClientPacket;
    use pumpkin_util::version::JavaMinecraftVersion;
    use std::collections::BTreeMap;

    #[test]
    fn dynamic_entity_tags_use_the_registry_wire_format() -> Result<(), crate::WritingError> {
        let entity_types = BTreeMap::from([("test:pickable".to_string(), vec![0, 127, 128])]);
        let packet = CUpdateTagsWithEntityTypes {
            tags: &[RegistryKey::EntityType],
            entity_types: &entity_types,
        };
        let mut bytes = Vec::new();
        packet.write_packet_data(&mut bytes, &JavaMinecraftVersion::V_26_3)?;
        let expected = b"\x01\x15minecraft:entity_type\x01\x0dtest:pickable\x03\x00\x7f\x80\x01";
        assert_eq!(bytes, expected);
        Ok(())
    }

    #[test]
    fn empty_replacement_is_sent_instead_of_generated_values() -> Result<(), crate::WritingError> {
        let entity_types = BTreeMap::from([("test:pickable".to_string(), Vec::new())]);
        let packet = CUpdateTagsWithEntityTypes {
            tags: &[RegistryKey::EntityType],
            entity_types: &entity_types,
        };
        let mut bytes = Vec::new();
        packet.write_packet_data(&mut bytes, &JavaMinecraftVersion::V_26_3)?;
        assert_eq!(
            bytes,
            b"\x01\x15minecraft:entity_type\x01\x0dtest:pickable\x00"
        );
        Ok(())
    }
}
