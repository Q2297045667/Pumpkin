// Don't warn on event sending macros
#![recursion_limit = "512"]
#![expect(unused_labels)]

#[cfg(target_os = "wasi")]
compile_error!("Compiling for WASI targets is not supported!");

use pumpkin_data::packet::CURRENT_MC_VERSION;
use std::{
    backtrace::{Backtrace, BacktraceStatus},
    io::{self},
    panic::PanicHookInfo,
    process::exit,
    sync::{Arc, LazyLock, OnceLock, atomic::Ordering},
    thread::{self, ThreadId},
};
#[cfg(not(unix))]
use tokio::signal::ctrl_c;
#[cfg(unix)]
use tokio::signal::unix::{SignalKind, signal};

use pumpkin::{
    CRASH_REPORT, SERVER_EXIT_CODE, SERVER_IS_STOPPING, SERVER_LOGGING_LOCALE,
    crash::{CrashReport, FullBacktrace},
    data::VanillaData,
    stop_or_exit_server,
};
use pumpkin::{LoggerOption, PumpkinServer, SHOULD_STOP, STOP_INTERRUPT, stop_server};
use pumpkin_i18n::{Locale, get_translation};
use pumpkin_util::text::translation::translation_to_pretty;

use pumpkin_config::{LoadConfiguration, PumpkinConfig};
use pumpkin_util::text::{
    TextComponent,
    color::{Color, NamedColor},
};
use std::time::Instant;
use tracing::{debug, info, warn};

// Setup some tokens to allow us to identify which event is for which socket.

pub mod block;
pub mod command;
pub mod crash;
pub mod data;
pub mod entity;
pub mod error;
pub mod item;
pub mod logging;
pub mod net;
pub mod plugin;
pub mod server;
pub mod world;

pub static LOGGER_IMPL: LazyLock<Arc<OnceLock<LoggerOption>>> =
    LazyLock::new(|| Arc::new(OnceLock::new()));

const CARGO_PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

static MAIN_THREAD: OnceLock<ThreadId> = OnceLock::new();

/// Returns the server logging locale, falling back to [`Locale::EnUs`].
fn server_locale() -> Locale {
    *SERVER_LOGGING_LOCALE.get().unwrap_or(&Locale::EnUs)
}

// WARNING: All rayon calls from the tokio runtime must be non-blocking! This includes things
// like `par_iter`. These should be spawned in the the rayon pool and then passed to the tokio
// runtime with a channel! See `Level::fetch_chunks` as an example!
#[allow(clippy::too_many_lines)]
#[tokio::main]
async fn main() {
    MAIN_THREAD
        .set(thread::current().id())
        .expect("Expected to successfully set the main thread ID");

    // Set the panic handler.
    std::panic::set_hook(Box::new(handle_panic));

    #[cfg(feature = "console-subscriber")]
    console_subscriber::init();
    let time = Instant::now();

    let exec_dir = std::env::current_dir().unwrap();

    let config = PumpkinConfig::load(&exec_dir);

    let vanilla_data = VanillaData::load();

    pumpkin::init_logger(&config.advanced);

    let locale = server_locale();
    info!(
        "{}",
        translation_to_pretty(
            "pumpkin:server.startup.starting_server",
            locale,
            vec![
                TextComponent::text("Pumpkin")
                    .color_named(NamedColor::Gold)
                    .0,
                TextComponent::text(CARGO_PKG_VERSION.to_string())
                    .color_named(NamedColor::Green)
                    .0,
                TextComponent::text(CURRENT_MC_VERSION.protocol_version().to_string())
                    .color_named(NamedColor::DarkBlue)
                    .0,
            ],
        )
    );

    debug!(
        "Build info: FAMILY: \"{}\", OS: \"{}\", ARCH: \"{}\", BUILD: \"{}\"",
        std::env::consts::FAMILY,
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(debug_assertions) {
            "Debug"
        } else {
            "Release"
        }
    );
    print_support_links_and_warning();

    tokio::spawn(async {
        setup_sighandler()
            .await
            .expect("Unable to setup signal handlers");
    });

    let pumpkin_server = PumpkinServer::new(config.basic, config.advanced, vanilla_data).await;
    let plugin_wait_time = pumpkin_server.init_plugins().await;

    let time_elapsed = time.elapsed().saturating_sub(plugin_wait_time);

    let time_msg = get_translation("pumpkin:server.startup.time_ms", locale)
        .replace("%s", &time_elapsed.as_millis().to_string());
    info!(
        "{}",
        translation_to_pretty(
            "pumpkin:server.startup.started",
            locale,
            vec![
                TextComponent::text(time_msg)
                    .color_named(NamedColor::Gold)
                    .0
            ],
        )
    );
    let basic_config = &pumpkin_server.server.basic_config;

    // Build Java Edition info component
    let java_label = get_translation("pumpkin:server.startup.java_edition_label", locale);
    let java_info = if basic_config.java_edition {
        format!(
            "{} {}",
            TextComponent::text(java_label)
                .color_named(NamedColor::Yellow)
                .to_pretty_console(),
            TextComponent::text(format!("{}", basic_config.java_edition_address))
                .color_named(NamedColor::DarkBlue)
                .to_pretty_console()
        )
    } else {
        String::new()
    };

    // Edition separator
    let separator = if basic_config.java_edition && basic_config.bedrock_edition {
        get_translation("pumpkin:server.startup.edition_separator", locale)
    } else {
        String::new()
    };

    // Build Bedrock Edition info component
    let bedrock_label = get_translation("pumpkin:server.startup.bedrock_edition_label", locale);
    let bedrock_info = if basic_config.bedrock_edition {
        format!(
            "{} {}",
            TextComponent::text(bedrock_label)
                .color_named(NamedColor::Gold)
                .to_pretty_console(),
            TextComponent::text(format!("{}", basic_config.bedrock_edition_address))
                .color_named(NamedColor::DarkBlue)
                .to_pretty_console()
        )
    } else {
        String::new()
    };

    let running_template = get_translation("pumpkin:server.startup.running_server", locale);
    let mut running_msg = running_template.replacen("%s", &java_info, 1);
    running_msg = running_msg.replacen("%s", &separator, 1);
    running_msg = running_msg.replacen("%s", &bedrock_info, 1);
    info!("{}", running_msg);

    pumpkin_server.start().await;

    info!(
        "{}",
        TextComponent::text(get_translation(
            "pumpkin:server.shutdown.stopped",
            server_locale(),
        ))
        .color_named(NamedColor::Red)
        .to_pretty_console()
    );

    exit(SERVER_EXIT_CODE.load(Ordering::Acquire));
}
fn print_support_links_and_warning() {
    let locale = server_locale();
    let issues_url = get_translation("pumpkin:server.issues_url", locale);
    let discord_url = get_translation("pumpkin:server.discord_url", locale);
    let discord_label = get_translation("pumpkin:server.discord_label", locale);

    warn!(
        "{}",
        TextComponent::text(get_translation("pumpkin:server.under_development", locale,))
            .color_named(NamedColor::DarkRed)
            .to_pretty_console(),
    );
    let report_msg = get_translation("pumpkin:server.report_issues", locale).replace(
        "%s",
        &TextComponent::text(issues_url)
            .color_named(NamedColor::DarkAqua)
            .to_pretty_console(),
    );
    info!("{}", report_msg);
    let community_msg = get_translation("pumpkin:server.join_community_support", locale)
        .replace(
            "%s",
            &TextComponent::text(discord_label)
                .color_named(NamedColor::DarkBlue)
                .to_pretty_console(),
        )
        .replacen(
            "%s",
            &TextComponent::text(discord_url)
                .color_named(NamedColor::Aqua)
                .to_pretty_console(),
            1,
        );
    info!("{}", community_msg);
}

fn handle_interrupt() {
    let locale = server_locale();
    warn!(
        "{}",
        TextComponent::text(get_translation("pumpkin:server.received_interrupt", locale,))
            .color_named(NamedColor::Red)
            .to_pretty_console()
    );
    stop_or_exit_server();
}

fn handle_panic(panic_info: &PanicHookInfo<'_>) {
    // Generate a crash report.
    let crash_report = {
        // We capture the backtraces here, and not in the
        // crash report, so that the backtrace doesn't show
        // the CrashReport's `new` function.
        let captured_backtrace = Backtrace::capture();
        let full_backtrace = if captured_backtrace.status() == BacktraceStatus::Captured {
            FullBacktrace::Captured
        } else {
            FullBacktrace::ForceCaptured(Backtrace::force_capture())
        };

        CrashReport::new(panic_info, captured_backtrace, full_backtrace)
    };

    let payload = panic_info.payload();
    let locale = server_locale();
    let unknown = get_translation("pumpkin:crash.unknown_payload", locale);

    if is_main_thread() {
        // It's the first panic;
        // We cannot gracefully shut down as the main thread
        // has panicked. However, we can still generate the crash report.

        if let Some(crash_report) = try_set_crash_report(crash_report) {
            crash_report.print_to_console();
            crash_report.save_and_log();

            tracing::error!(
                "{}",
                TextComponent::text(get_translation(
                    "pumpkin:crash.main_thread_aborting",
                    locale,
                ))
                .color(Color::Named(NamedColor::Red))
                .to_pretty_console()
            );
        } else {
            // It's a subsequent panic.
            tracing::error!(
                "{}: {}",
                TextComponent::text(get_translation(
                    "pumpkin:crash.main_thread_panicked_during_shutdown",
                    locale,
                ))
                .color(Color::Named(NamedColor::Red))
                .bold()
                .to_pretty_console(),
                payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or(&unknown)
            );
        }

        exit(1);
    }

    if try_set_crash_report(crash_report).is_some() {
        // It's the first panic; let's stop the server.
        stop_server();
    } else {
        // It's a subsequent panic; let's just alert about it.
        tracing::error!(
            "{}: {}",
            TextComponent::text(get_translation(
                "pumpkin:crash.panic_during_shutdown",
                locale,
            ))
            .color(Color::Named(NamedColor::Red))
            .bold()
            .to_pretty_console(),
            payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or(&unknown)
        );
    }
}

fn is_main_thread() -> bool {
    Some(&thread::current().id()) == MAIN_THREAD.get()
}

/// Returns `Some` if the crash report was successfully set. That
/// means it is the first panic, and it must be logged and saved later.
///
/// Returns `None` otherwise as the panic is subsequent.
fn try_set_crash_report(crash_report: CrashReport) -> Option<&'static CrashReport> {
    if !SERVER_IS_STOPPING.load(Ordering::Acquire) && CRASH_REPORT.set(crash_report).is_ok() {
        CRASH_REPORT.get()
    } else {
        None
    }
}

// Non-UNIX Ctrl-C handling
#[cfg(not(unix))]
async fn setup_sighandler() -> io::Result<()> {
    if ctrl_c().await.is_ok() {
        handle_interrupt();
    }

    Ok(())
}

// Unix signal handling
#[cfg(unix)]
async fn setup_sighandler() -> io::Result<()> {
    if signal(SignalKind::interrupt())?.recv().await.is_some() {
        handle_interrupt();
    }

    if signal(SignalKind::hangup())?.recv().await.is_some() {
        handle_interrupt();
    }

    if signal(SignalKind::terminate())?.recv().await.is_some() {
        handle_interrupt();
    }

    Ok(())
}
