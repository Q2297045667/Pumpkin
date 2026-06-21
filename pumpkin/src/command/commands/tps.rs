use crate::command::CommandResult;
use crate::command::{CommandExecutor, CommandSender, args::ConsumedArgs, tree::CommandTree};
use pumpkin_util::text::{TextComponent, color::NamedColor};

const NAMES: [&str; 1] = ["tps"];

const DESCRIPTION: &str = "Displays the server TPS and MSPT.";

struct Executor;

impl CommandExecutor for Executor {
    fn execute<'a>(
        &'a self,
        sender: &'a CommandSender,
        server: &'a crate::server::Server,
        _args: &'a ConsumedArgs<'a>,
    ) -> CommandResult<'a> {
        Box::pin(async move {
            let tps = server.get_tps().min(server.basic_config.tps as f64);
            let mspt = server.get_mspt();
            let locale = sender.get_locale(server);

            let max_tps = server.basic_config.tps as f64;
            let tps_color = if tps >= max_tps * 0.9 {
                NamedColor::Green
            } else if tps >= max_tps * 0.75 {
                NamedColor::Yellow
            } else {
                NamedColor::Red
            };

            let ms_unit = pumpkin_util::text::translation::get_translation_text(
                "pumpkin:commands.tps.ms_unit",
                locale,
                vec![],
            );

            let message =
                TextComponent::custom("pumpkin", "commands.tps.tps_label", locale, vec![])
                    .add_child(TextComponent::text(format!("{tps:.1}")).color_named(tps_color))
                    .add_child(TextComponent::custom(
                        "pumpkin",
                        "commands.tps.mspt_label",
                        locale,
                        vec![],
                    ))
                    .add_child(
                        TextComponent::text(format!("{mspt:.2}{ms_unit}")).color_named(tps_color),
                    );

            sender.send_message(message).await;

            Ok(tps as i32)
        })
    }
}

pub fn init_command_tree() -> CommandTree {
    CommandTree::new(NAMES, DESCRIPTION).execute(Executor)
}
