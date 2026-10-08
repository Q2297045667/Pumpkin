use std::fs;

use proc_macro2::TokenStream;
use quote::quote;
use serde::Deserialize;

#[derive(Deserialize)]
struct EnderDragonPart {
    dimension: [f32; 2],
    eye_height: f32,
}

pub fn build() -> TokenStream {
    let parts: Vec<EnderDragonPart> =
        serde_json::from_str(&fs::read_to_string("../../assets/ender_dragon_parts.json").unwrap())
            .expect("Failed to parse ender_dragon_parts.json");
    let count = parts.len();
    let dimensions = parts.iter().map(|part| {
        let [width, height] = part.dimension;
        let eye_height = part.eye_height;
        quote! { EntityDimensions::new(#width, #height, #eye_height) }
    });

    quote! {
        use pumpkin_util::math::boundingbox::EntityDimensions;

        pub const ENDER_DRAGON_PART_DIMENSIONS: [EntityDimensions; #count] = [
            #(#dimensions),*
        ];
    }
}
