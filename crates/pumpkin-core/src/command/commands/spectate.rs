use std::sync::Arc;

use pumpkin_data::translation;
use pumpkin_util::permission::{Permission, PermissionDefault, PermissionRegistry};
use pumpkin_util::text::TextComponent;
use pumpkin_util::{GameMode, PermissionLvl};

use crate::command::argument_builder::{ArgumentBuilder, argument, command};
use crate::command::argument_types::entity::EntityArgumentType;
use crate::command::context::command_context::CommandContext;
use crate::command::errors::error_types::CommandErrorType;
use crate::command::node::dispatcher::CommandDispatcher;
use crate::command::node::{CommandExecutor, CommandExecutorResult};
use crate::entity::{EntityBase, player::Player};

const DESCRIPTION: &str = "Allows a player in spectator mode to spectate a given target entity.";
const PERMISSION: &str = "minecraft:command.spectate";

const ERROR_NOT_PLAYER: CommandErrorType<0> = CommandErrorType::new(
    translation::java::PERMISSIONS_REQUIRES_PLAYER,
    translation::java::PERMISSIONS_REQUIRES_PLAYER,
);
const ERROR_NOT_SPECTATOR: CommandErrorType<1> = CommandErrorType::new(
    translation::java::COMMANDS_SPECTATE_NOT_SPECTATOR,
    translation::java::COMMANDS_SPECTATE_NOT_SPECTATOR,
);
const ERROR_SELF: CommandErrorType<0> = CommandErrorType::new(
    translation::java::COMMANDS_SPECTATE_SELF,
    translation::java::COMMANDS_SPECTATE_SELF,
);
const ERROR_CANNOT_SPECTATE: CommandErrorType<1> = CommandErrorType::new(
    translation::java::COMMANDS_SPECTATE_CANNOT_SPECTATE,
    translation::java::COMMANDS_SPECTATE_CANNOT_SPECTATE,
);

fn queue_camera(
    context: &CommandContext,
    player: &Arc<Player>,
    target: Option<Arc<dyn EntityBase>>,
) {
    let message = target.as_ref().map_or_else(
        || {
            TextComponent::translate_cross(
                translation::java::COMMANDS_SPECTATE_SUCCESS_STOPPED,
                translation::java::COMMANDS_SPECTATE_SUCCESS_STOPPED,
                [],
            )
        },
        |target| {
            TextComponent::translate_cross(
                translation::java::COMMANDS_SPECTATE_SUCCESS_STARTED,
                translation::java::COMMANDS_SPECTATE_SUCCESS_STARTED,
                [target.get_display_name()],
            )
        },
    );
    let result = player.request_spectate(target);
    let source = context.source.clone();
    let client = player.client.clone();
    player.spawn_task(async move {
        let accepted = tokio::select! {
            result = result => result,
            () = client.await_close_interrupt() => return,
        };
        if matches!(accepted, Ok(true)) {
            source.send_feedback(message, false);
        }
    });
}

struct StopSpectateExecutor;
impl CommandExecutor for StopSpectateExecutor {
    fn execute(&self, context: &CommandContext) -> CommandExecutorResult {
        let player = context
            .source
            .output
            .as_player()
            .ok_or_else(|| ERROR_NOT_PLAYER.create_without_context())?;
        if player.gamemode.load() != GameMode::Spectator {
            return Err(ERROR_NOT_SPECTATOR.create_without_context(player.get_display_name()));
        }
        queue_camera(context, &player, None);
        Ok(1)
    }
}

struct SpectateTargetExecutor {
    is_self: bool,
}
impl CommandExecutor for SpectateTargetExecutor {
    fn execute(&self, context: &CommandContext) -> CommandExecutorResult {
        let target = EntityArgumentType::get_entity(context, "target")?;
        let player = if self.is_self {
            context
                .source
                .output
                .as_player()
                .ok_or_else(|| ERROR_NOT_PLAYER.create_without_context())?
        } else {
            EntityArgumentType::get_player(context, "player")?
        };
        if std::ptr::eq(target.get_entity(), player.get_entity()) {
            return Err(ERROR_SELF.create_without_context());
        }
        if player.gamemode.load() != GameMode::Spectator {
            return Err(ERROR_NOT_SPECTATOR.create_without_context(player.get_display_name()));
        }
        if target.get_entity().entity_type.client_tracking_range == 0 {
            return Err(ERROR_CANNOT_SPECTATE.create_without_context(target.get_display_name()));
        }
        queue_camera(context, &player, Some(target));
        Ok(1)
    }
}

pub fn register(dispatcher: &mut CommandDispatcher, registry: &PermissionRegistry) {
    registry.register_permission_or_panic(Permission::new(
        PERMISSION,
        DESCRIPTION,
        PermissionDefault::Op(PermissionLvl::Two),
    ));
    dispatcher.register(
        command("spectate", DESCRIPTION)
            .requires(PERMISSION)
            .executes(StopSpectateExecutor)
            .then(
                argument("target", EntityArgumentType::Entity)
                    .executes(SpectateTargetExecutor { is_self: true })
                    .then(
                        argument("player", EntityArgumentType::Player)
                            .executes(SpectateTargetExecutor { is_self: false }),
                    ),
            ),
    );
}
