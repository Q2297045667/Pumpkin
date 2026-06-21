use std::path::Path;

use pumpkin_util::{
    PermissionLvl,
    text::translation::get_translation_text,
    text::{TextComponent, color::NamedColor, hover::HoverEvent},
};

use crate::command::{
    CommandExecutor, CommandResult, CommandSender,
    args::{Arg, ConsumedArgs, simple::SimpleArgConsumer},
    dispatcher::CommandError,
    tree::{
        CommandTree,
        builder::{argument, literal, require},
    },
};

use crate::command::CommandError::InvalidConsumption;

const NAMES: [&str; 1] = ["plugin"];

const DESCRIPTION: &str = "Manage plugins.";

const PLUGIN_NAME: &str = "plugin_name";

struct ListExecutor;

impl CommandExecutor for ListExecutor {
    fn execute<'a>(
        &'a self,
        sender: &'a CommandSender,
        server: &'a crate::server::Server,
        _args: &'a ConsumedArgs<'a>,
    ) -> CommandResult<'a> {
        Box::pin(async move {
            let plugins = server.plugin_manager.active_plugins().await;
            let locale = sender.get_locale(server);

            let message = if plugins.is_empty() {
                TextComponent::custom("pumpkin", "commands.plugin.no_plugins", locale, vec![])
            } else if plugins.len() == 1 {
                TextComponent::custom("pumpkin", "commands.plugin.one_plugin", locale, vec![])
            } else {
                TextComponent::custom(
                    "pumpkin",
                    "commands.plugin.multiple_plugins",
                    locale,
                    vec![TextComponent::text(plugins.len().to_string())],
                )
            };

            let sep_str =
                get_translation_text("pumpkin:commands.plugin.list.separator", locale, vec![]);
            let mut message = message.clone();

            for (i, metadata) in plugins.iter().enumerate() {
                let hover_text = TextComponent::custom(
                    "pumpkin",
                    "commands.plugin.hover_text",
                    locale,
                    vec![
                        TextComponent::text(metadata.version.clone()),
                        TextComponent::text(metadata.authors.join(", ")),
                        TextComponent::text(metadata.description.clone()),
                    ],
                );
                let component = if i == plugins.len() - 1 {
                    TextComponent::text(metadata.name.clone())
                } else {
                    TextComponent::text(format!(
                        "{metadata_name}{sep_str}",
                        metadata_name = metadata.name
                    ))
                }
                .color_named(NamedColor::Green)
                .hover_event(HoverEvent::show_text(hover_text));

                message = message.add_child(component);
            }

            sender.send_message(message).await;

            Ok(plugins.len() as i32)
        })
    }
}

struct LoadExecutor;

impl CommandExecutor for LoadExecutor {
    fn execute<'a>(
        &'a self,
        sender: &'a CommandSender,
        server: &'a crate::server::Server,
        args: &'a ConsumedArgs<'a>,
    ) -> CommandResult<'a> {
        Box::pin(async move {
            let Some(Arg::Simple(plugin_name)) = args.get(PLUGIN_NAME) else {
                return Err(InvalidConsumption(Some(PLUGIN_NAME.into())));
            };
            let plugin_name_owned = plugin_name.to_owned();
            let locale = sender.get_locale(server);

            if server
                .plugin_manager
                .is_plugin_active(plugin_name_owned)
                .await
            {
                let msg = TextComponent::custom(
                    "pumpkin",
                    "commands.plugin.already_loaded",
                    locale,
                    vec![TextComponent::text(plugin_name_owned.to_string())],
                );
                return Err(CommandError::CommandFailed(msg));
            }

            let result = server
                .plugin_manager
                .try_load_plugin(Path::new(&plugin_name_owned))
                .await;

            match result {
                Ok(()) => {
                    let name_str = plugin_name_owned.to_string();
                    sender
                        .send_message(
                            TextComponent::custom(
                                "pumpkin",
                                "commands.plugin.loaded_successfully",
                                locale,
                                vec![TextComponent::text(name_str)],
                            )
                            .color_named(NamedColor::Green),
                        )
                        .await;
                    Ok(1)
                }
                Err(e) => {
                    let msg = TextComponent::custom(
                        "pumpkin",
                        "commands.plugin.failed_to_load",
                        locale,
                        vec![
                            TextComponent::text(plugin_name_owned.to_string()),
                            TextComponent::text(e.to_string()),
                        ],
                    );
                    Err(CommandError::CommandFailed(msg))
                }
            }
        })
    }
}

struct UnloadExecutor;

impl CommandExecutor for UnloadExecutor {
    fn execute<'a>(
        &'a self,
        sender: &'a CommandSender,
        server: &'a crate::server::Server,
        args: &'a ConsumedArgs<'a>,
    ) -> CommandResult<'a> {
        Box::pin(async move {
            let Some(Arg::Simple(plugin_name)) = args.get(PLUGIN_NAME) else {
                return Err(InvalidConsumption(Some(PLUGIN_NAME.into())));
            };
            let plugin_name_owned = plugin_name.to_owned();
            let locale = sender.get_locale(server);

            if !server
                .plugin_manager
                .is_plugin_active(plugin_name_owned)
                .await
            {
                let msg = TextComponent::custom(
                    "pumpkin",
                    "commands.plugin.not_loaded",
                    locale,
                    vec![TextComponent::text(plugin_name_owned.to_string())],
                );
                return Err(CommandError::CommandFailed(msg));
            }

            let result = server.plugin_manager.unload_plugin(plugin_name_owned).await;

            match result {
                Ok(()) => {
                    let name_str = plugin_name_owned.to_string();
                    sender
                        .send_message(
                            TextComponent::custom(
                                "pumpkin",
                                "commands.plugin.unloaded_successfully",
                                locale,
                                vec![TextComponent::text(name_str)],
                            )
                            .color_named(NamedColor::Green),
                        )
                        .await;

                    Ok(1)
                }
                Err(e) => {
                    let msg = TextComponent::custom(
                        "pumpkin",
                        "commands.plugin.failed_to_unload",
                        locale,
                        vec![
                            TextComponent::text(plugin_name_owned.to_string()),
                            TextComponent::text(e.to_string()),
                        ],
                    );
                    Err(CommandError::CommandFailed(msg))
                }
            }
        })
    }
}

struct HotReloadExecutor(bool);

impl CommandExecutor for HotReloadExecutor {
    fn execute<'a>(
        &'a self,
        sender: &'a CommandSender,
        server: &'a crate::server::Server,
        _args: &'a ConsumedArgs<'a>,
    ) -> CommandResult<'a> {
        Box::pin(async move {
            let enabled = self.0;
            let locale = sender.get_locale(server);

            if enabled {
                if let Err(e) = server.plugin_manager.start_watcher().await {
                    return Err(CommandError::CommandFailed(TextComponent::custom(
                        "pumpkin",
                        "commands.plugin.failed_to_start_watcher",
                        locale,
                        vec![TextComponent::text(e.to_string())],
                    )));
                }

                sender
                    .send_message(
                        TextComponent::custom(
                            "pumpkin",
                            "commands.plugin.hotreload_enabled",
                            locale,
                            vec![],
                        )
                        .color_named(NamedColor::Green),
                    )
                    .await;
                sender
                    .send_message(
                        TextComponent::custom(
                            "pumpkin",
                            "commands.plugin.hotreload_warning",
                            locale,
                            vec![],
                        )
                        .color_named(NamedColor::Red),
                    )
                    .await;
            } else {
                server.plugin_manager.stop_watcher().await;

                sender
                    .send_message(
                        TextComponent::custom(
                            "pumpkin",
                            "commands.plugin.hotreload_disabled",
                            locale,
                            vec![],
                        )
                        .color_named(NamedColor::Green),
                    )
                    .await;
            }

            Ok(1)
        })
    }
}

pub fn init_command_tree() -> CommandTree {
    CommandTree::new(NAMES, DESCRIPTION).then(
        require(|sender| sender.has_permission_lvl(PermissionLvl::Three))
            .then(
                literal("load")
                    .then(argument(PLUGIN_NAME, SimpleArgConsumer).execute(LoadExecutor)),
            )
            .then(
                literal("unload")
                    .then(argument(PLUGIN_NAME, SimpleArgConsumer).execute(UnloadExecutor)),
            )
            .then(
                literal("hotreload")
                    .then(literal("enable").execute(HotReloadExecutor(true)))
                    .then(literal("disable").execute(HotReloadExecutor(false))),
            )
            .then(literal("list").execute(ListExecutor)),
    )
}
