use pumpkin_util::text::translation::get_translation_text;
use pumpkin_util::text::{TextComponent, color::NamedColor, hover::HoverEvent};

use crate::command::{
    CommandExecutor, CommandResult, CommandSender, args::ConsumedArgs, tree::CommandTree,
};

const NAMES: [&str; 2] = ["pl", "plugins"];

const DESCRIPTION: &str = "List all available plugins.";

struct Executor;

impl CommandExecutor for Executor {
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
                TextComponent::custom("pumpkin", "commands.plugins.no_plugins", locale, vec![])
            } else if plugins.len() == 1 {
                TextComponent::custom("pumpkin", "commands.plugins.one_plugin", locale, vec![])
            } else {
                TextComponent::custom(
                    "pumpkin",
                    "commands.plugins.multiple_plugins",
                    locale,
                    vec![TextComponent::text(plugins.len().to_string())],
                )
            };

            let sep_str =
                get_translation_text("pumpkin:commands.plugins.list.separator", locale, vec![]);
            let mut message = message.clone();

            for (i, metadata) in plugins.clone().into_iter().enumerate() {
                let hover_text = TextComponent::custom(
                    "pumpkin",
                    "commands.plugins.hover_text",
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

pub fn init_command_tree() -> CommandTree {
    CommandTree::new(NAMES, DESCRIPTION).execute(Executor)
}
