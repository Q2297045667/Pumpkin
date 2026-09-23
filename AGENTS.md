# AGENTS.md

Notes for coding agents working on Pumpkin. Humans are welcome to read it too. [CONTRIBUTING.md](CONTRIBUTING.md) still applies. If a maintainer or the person you're working for tells you something different from this file, do what they say. The one exception is opening pull requests, covered at the end.

## What Pumpkin is

Pumpkin is a Minecraft server written from scratch in Rust. It aims to behave like the official (vanilla) Java Edition server while running faster and on more threads. It isn't a fork of the Java code and it can't load Bukkit or Fabric plugins; it has its own native and Wasm plugin APIs.

Things worth knowing before you start:

- Java protocol support targets one version, currently 26.3 (`CURRENT_MC_VERSION` and `LOWEST_SUPPORTED_MC_VERSION` in `crates/pumpkin-data/src/generated/packet.rs`). Older clients are handled by plugins. Bedrock has a separate implementation that is still in progress.
- Game logic (block, item, entity and AI behaviour) runs synchronously inside the tick. Networking, disk I/O, plugins and events are async on Tokio. CPU-heavy work such as chunk generation and lighting runs on Rayon.
- Chunk loading and generation live in `pumpkin-world` and use a ticket and level system modelled on vanilla. Pumpkin's numbering isn't the same as vanilla's (see the pitfalls below).
- Pumpkin deliberately differs from vanilla in a few places. The most important one is that the player's own movement is client authoritative. A change that is correct against the Java code can still break an assumption like that, so check how Pumpkin already handles the thing you're touching.

## Layout

Crates live in `crates/` and are named after what they do (`pumpkin-world` for chunks and generation, `pumpkin-protocol` for packets, and so on). `crates/pumpkin` is the server itself. `assets/` holds the extracted vanilla data and `tools/pumpkin-codegen` turns it into `pumpkin-data`. The server is GPL-3.0; the plugin crates have their own licenses (see Plugin API below).

## Vanilla parity

A gameplay change is correct when it behaves like the official server, not when it compiles and looks plausible. Reviewers compare PRs against the decompiled vanilla source, so work from that source too, not from memory or the wiki. Both are often wrong about the details that matter here: tick order, rounding, which side effects happen and when.

### Getting the source

1. Get the version list from `https://piston-meta.mojang.com/mc/game/version_manifest_v2.json`, open the JSON for the target version, and download `downloads.server.url`.
2. The download is a bundler. The real server jar is inside it under `META-INF/versions/<version>/`.
3. Decompile the classes you need with [Vineflower](https://github.com/Vineflower/vineflower). The 26.x jars ship without obfuscation, so names are readable as-is. For older versions, apply Mojang's mappings from `downloads.server_mappings.url`.
4. Decompile only the packages you need (Vineflower's `-only=<package>`). Keep all of this outside the repo.

### Porting

- Start from the vanilla entry point and follow the whole call chain, including every override in the class hierarchy and whatever the base class later does with the fields you set. Porting only the top method is the most common way a port ends up "almost right".
- Look for implicit behaviour. A vanilla call can have side effects its Pumpkin counterpart doesn't, such as loading a chunk on a block read. Code that depends on those side effects has to do them explicitly in Pumpkin.
- Check ordering around early returns. If your port returns early (a cancel, a deflect, a dodge), list what the surrounding Pumpkin function already did before that point (stats, cooldowns, events, sounds) and compare it with where vanilla does each one.
- Check what you delete. When you replace old code with a more faithful version, make sure nothing the old code did along the way gets lost. Vanilla often does that side effect in a different method, so matching the new method against vanilla doesn't prove the side effect is still there.
- Check who else calls the code you changed. Shared entity, world and networking code has many callers, and a fix that's right for one of them can change the others.
- Know what the client simulates. The client predicts some state itself, so sending the server's copy back to the same player fights the prediction. Check who vanilla actually sends each update to.
- Translate values, don't copy the numbers. Flags, levels and enums have Pumpkin equivalents, and they don't always have the same numeric values as vanilla. Look up the Pumpkin constant instead of pasting the Java literal.
- Check for open tracking issues before porting a large system. Some areas already have a planned design (mob AI, for example), and a port that works around it won't be accepted.

If you couldn't check part of a port against vanilla, say which part in the PR.

## Game data and constants

Block and item properties, entity dimensions, tags, recipes, loot tables, sounds and similar values come from `pumpkin-data`, which is generated from the JSON in `assets/`. That JSON is produced by the [Extractor](https://github.com/Pumpkin-MC/Extractor), a Fabric mod that dumps data from the running game. Hardcoded copies go stale on the next Minecraft update, which is why reviewers reject them.

- If the value is in `assets/` but not generated yet, extend `tools/pumpkin-codegen` and regenerate.
- If the value isn't in `assets/` at all, it belongs in the Extractor, so prepare a change there as well. The same pull request rules apply to it.
- Values that vanilla hardcodes as `static final` constants in Java code stay as named Rust constants, using the Java name. Don't try to extract those.
- Don't guess engine values either, like a world height or a dimension's limits. Datapacks can change them, so read them from where the game defines them.

Each generator in `tools/pumpkin-codegen/src/` is usually named after its input: `block.rs` reads `assets/blocks.json`, `item.rs` reads `items.json`, `loot_table.rs` reads the loot tables in `assets/datapack/`, and so on. `main.rs` lists them all.

Regenerate with `cargo run --locked -p pumpkin-codegen` (or `-- <output name>` for a single file) and never edit `crates/pumpkin-data/src/generated/` by hand. A codegen run can rewrite generated files you didn't mean to change, so keep only the files your change needs.

If generated data looks wrong at runtime, check the codegen before blaming the JSON. A JSON variant the generator doesn't handle can fall through to a default without any error.

## Where things get wired up

Paths are under `crates/pumpkin/src/`. Forgetting to register a new implementation is a common mistake, so copy the wiring of the closest existing example.

| Adding | Example to follow | Register in |
|:--|:--|:--|
| Block behaviour | `block/blocks/dirt_path.rs` | `block/blocks/mod.rs`, `block/registry.rs` |
| Block entity | `block/entities/campfire.rs` | `block_entity_from_nbt()` in `block/entities/mod.rs` |
| Item behaviour | `item/items/bucket.rs` | `default_registry()` in `item/items/mod.rs` |
| Mob | `entity/mob/bat.rs` | `from_type()` in `entity/type.rs`, spawn rules in `world/natural_spawner.rs` |
| AI goal | `entity/ai/goal/melee_attack.rs` | `add_goal` / `add_target_goal` on `MobEntity` |
| Command | `command/commands/time.rs` | `default_dispatcher()` in `command/commands/mod.rs` |

Things that aren't obvious from the code:

- `BlockBehaviour`, `ItemBehaviour` and `Goal` methods are synchronous.
- `#[pumpkin_block(...)]` generates `BlockMetadata`. Mobs get `EntityBase` and `NBTStorage` through blanket impls, so mob NBT goes in `mob_write_nbt` / `mob_read_nbt`.
- A block entity's `write_internal()` already writes `id`, `x`, `y` and `z`. `write_nbt()` only writes its own fields.

### Packets

Keep Alive is a small example that touches every layer:

1. **Packet ids** come from `assets/packets.json` through codegen. The packet struct uses `#[java_packet(KEEP_ALIVE)]` with the generated constant, never a literal id.
2. **Serverbound packets** (`S...`, client to server) live in `pumpkin-protocol/src/java/server/<phase>/` and implement `ServerPacket::read()`. **Clientbound packets** (`C...`) live in `java/client/<phase>/` and implement `ClientPacket::write_packet_data()`. Both have to be exported from the phase's `mod.rs`.
3. **Dispatch** of play packets happens in `handle_play_packet()` in `net/java/mod.rs`, which matches on `to_id(version)`. Earlier phases (handshake, status, login, configuration) are handled in `net/java/pending.rs`, not in the play dispatcher.
4. **The handler** goes in its own file under `net/java/play/`.

When you change a packet, check the field order, types and length limits against vanilla, and add a round-trip test next to the existing ones.

## Plugin API

This section covers changes to the plugin API inside this repo. How to write a plugin is documented in `pumpkin-plugin-api`, not here.

- **Mind the licenses.** The server is GPL-3.0, but `pumpkin-plugin-api`, `pumpkin-plugin-wit` and `pumpkin-plugin-utils` are MIT OR Apache-2.0. Don't move or copy server code into those crates. Plugin authors depend on them under the permissive license.
- **The WIT is public.** `crates/pumpkin-plugin-wit` is mirrored to its own repository on every push to `master`, and bindings for other languages are generated from it. A breaking change there breaks SDKs outside this repo too. Prefer additive changes, and call out anything breaking in the PR.
- **Change the layers together.** A WIT change also needs the host bindings (`pumpkin-host-bindings`), the host implementation (`plugin/loader/wasm/`), the SDK (`pumpkin-plugin-api`) and sometimes the runtime (`pumpkin-plugin-runtime`). Run `cargo run --locked -p pumpkin-codegen -- wit` afterwards, because WIT data and packet mappings are generated. Breaking changes to the native plugin API bump `PLUGIN_API_VERSION` in `plugin/mod.rs`.
- **Keep events working.** When you port gameplay that a plugin could want to observe or cancel, check whether an event for it already exists in `plugin/api/events/`. If it does, fire it before the effect is applied, and skip the effect when `cancelled()` returns true. A port that skips an existing event silently breaks every plugin that relied on it.

## Code quality

Maintainers and CodeRabbit read every diff line by line, and they read ports next to the Java. The rules below are the things they keep flagging. Fixing them before handover saves a review round.

### Fit in with the code around you

- Copy the closest existing code in the same area: how it's registered, how it implements its traits, how it reaches world and entity APIs, how it handles errors and lays out imports. A second pattern for the same job is harder to maintain than a slightly imperfect one used everywhere.
- Mirror the vanilla shape. Where vanilla has one named method, write one named helper and put it on the Pumpkin type that matches vanilla's class. Don't inline the math. Reviewers put the two side by side, and matching structure is how they check it.
- Names say what the thing is. Keep vanilla's names for ported concepts so they can be searched for in the Java.

### Don't repeat yourself

- Search for an existing helper before writing common math or lookups yourself. If the same logic appears twice, in your diff or already in the repo, pull it into one helper.
- Name magic numbers. If a number appears more than once, or a neighbouring file already has a name for it, use a constant. Constants ported from Java keep the Java name.

### Keep it as small as the task

- Don't add abstractions, config options, feature flags or traits for hypothetical future use. Three similar lines are fine. A helper called from one place usually isn't worth it unless vanilla has the same method.
- Don't add error handling or fallbacks for cases that can't happen. Validate at the boundaries: network input, files on disk, config, plugin calls.
- Handle `Result` and `Option` for real. Clippy denies `unwrap`, `expect`, `panic`, `todo!` and printing to stdout/stderr. Log with `tracing`. If you really need an exception, use a local `#[expect(..., reason = "...")]`.
- Don't block Tokio workers, because it stalls ticks for every player. CPU-heavy work goes to Rayon, and sync locks or `DashMap` guards are never held across `.await`.
- Bound lengths and allocations when parsing anything from the network or disk.
- Add no new clippy warnings. Some files already have warnings; you don't have to fix those, but don't add to them.

### Comments and text

- Comment only what the code can't say: a non-obvious vanilla quirk, why an order matters, which Java method a block mirrors. Keep it to one short line. Don't narrate what the next line does, and don't put doc prose on every item. A doc comment on a public item is one plain sentence.
- Messages that server admins see follow the existing log lines: one short line saying what's wrong. Don't tell admins to flip an ignore or bypass setting to make a warning go away; people will do that instead of fixing the cause.
- Write commit messages and PR text like a person: plain sentences, no filler, no em dashes, no "this PR aims to". Don't add tool trailers to commits. The one place AI use is disclosed is the PR description, see below.

### Commits

- Use conventional commits with a scope, subject line only. Put details in the PR description, not in a commit body.
- Features that need play testing land as ordered commits, one per stage, so each stage can be tested and reverted on its own.
- Fold fixes into the commit they belong to (amend or fixup and rebase before handover). Don't stack "fix review comments" commits on top.

### Read your own diff before handover

Read the diff the way a maintainer would. Check three things:

1. Added lines: each one is checked against vanilla, and there is no duplicated logic, no bare magic numbers and no clunky hand-rolled code.
2. Removed lines, from `git diff master -- <paths> | grep '^-'`: for each one, can you say where that behaviour went? It should be kept elsewhere, dropped on purpose, or flagged as lost.
3. Callers: for every shared function, field or trait method you changed, list the other callers and say what changes for each of them.

## Scope

Change what the task needs and nothing else. If you notice a separate bug, slow code or a missing feature along the way, mention it in the PR description instead of fixing it in the same diff. That keeps reviews short and reverts clean. Don't reformat code you didn't touch, and don't commit unrelated lockfile or generated-file changes. Base your branch on current `master`; a stale base can show bugs that are already fixed.

## Checking your work

CI only runs on the affected packages (`.github/scripts/ci_packages.py`). Locally, run it on the crates you touched:

```bash
cargo fmt --check
cargo clippy -p <crate> --all-targets --all-features
cargo nextest run -p <crate> --no-tests=pass
cargo machete
typos
```

### Running the server yourself

Passing clippy and tests doesn't show that a gameplay change works. Run the server and exercise the change before you call it done:

- **Boot test.** Start the server in a scratch directory (`cargo run -p pumpkin` writes config and world files into the working directory). Wait for the "done" line in the log. Send console commands over stdin to reach the feature. Stop it with `stop`. Then read the log for panics, errors and warnings your change added. Use debug builds: the release profile uses fat LTO and takes many times longer to build.
- **Bots.** Log a headless client in for anything that needs a player: movement, chunk loading, combat, inventories, packets. [Mineflayer](https://github.com/PrismarineJS/mineflayer) is scriptable and good for "walk here, use this item, check the block". [rust-mc-bot](https://github.com/Eoghanmc22/rust-mc-bot) is good for load tests with many players. Bot libraries often lag behind the newest protocol, so check which version the bot supports first.
- **Load and memory.** For performance or memory changes, run the same bot scenario (same seed, bot count, view distance and duration) on `master` and on your branch. Record MSPT or peak memory for both. One run each isn't enough when the numbers are close.
- **Counters over guesses.** When you're chasing a bug, add temporary counters or `tracing` output to prove the cause before changing code. Take the instrumentation out before committing.
- **Keep harnesses out of the repo.** Bot scripts, benchmark scripts and debug patches live outside the repository and don't go into the PR.

Some things still need a real client: rendering, particles, sounds, animations, GUI screens, and anything that depends on client-side prediction. For those, say in your report exactly what a human should do in-game and what they should see, so they can check it before the PR goes up.

### Tests

A test earns its place when it would catch a real regression that the compiler, clippy and a boot test wouldn't. Many agent-written tests don't do that. They add review work and maintenance, and they make a PR look better tested than it is.

Write a test when:

- The code turns bytes into values or back: codecs, packet reading and writing, NBT, parsers. Round trips and malformed input are worth covering there.
- The logic is easy to break quietly and hard to see in-game: scheduling, caching, ordering, off-by-one boundaries.
- You're fixing a bug that a small test can reproduce. Check that the test fails without your fix. A regression test that passes on `master` proves nothing.
- You can compare against real vanilla output, like extracted data or chunk dumps.

Don't write tests that:

- Repeat the implementation. If the expected value in the test is computed with the same formula as the code, the test can't fail when the formula is wrong.
- Check constants, generated data or static tables against a copy of themselves.
- Only cover constructors, getters, setters, `Default`, derive output, or what the standard library and dependencies already guarantee.
- Check that removed code is gone, or cover code paths your change doesn't touch.
- Mock so much that nothing real is left to test.
- Duplicate an existing test with slightly different numbers.

Size new tests like the ones next to them: usually one focused test per behaviour, in the same place the module already keeps its tests. Scratch checks you wrote while working are for you; delete them instead of turning them into test files. When a change can't be tested meaningfully in code, say so and describe the boot, bot or in-game check you did instead. Behaviour that needs a real world belongs in `pumpkin-gametest`.

### World generation and performance

World generation: compare output with vanilla, not only with the previous Pumpkin build. `crates/pumpkin-world` already has tests against vanilla chunk dumps.

Performance: measure before and after on the same machine and put the numbers in the PR. Run the existing benches on both commits before deciding where a slowdown comes from; the cause is often not where it looks.

## Pull requests

A human opens every pull request, not an agent. Maintainers review each PR by hand, and they need a person on the other end who has run the change and can answer questions about it. Your job ends at a pushed branch and a drafted description. Don't open PRs, and don't post comments, reviews or replies on GitHub by yourself. That includes CLI tools, the API and scheduled or automated workflows.

What to hand over:

- The branch, based on current `master`, with commits in conventional commit format with a scope, like `fix(entity): ...` or `feat(block): ...`.
- A description draft that follows `.github/PULL_REQUEST_TEMPLATE.md`. Under Description, say what changed and why. Under Testing, say what you ran, what you checked against vanilla, and what you didn't verify. Keep it short and only about this change.
- A disclosure as the last line of that description:

  ```
  > This description was drafted by an AI agent (<tool and model>) and reviewed by the author.
  ```

  Reviewers read an agent-written description differently. They check its claims against the code instead of taking them on trust, so they need to know. The line stays even after the human edits the draft. The human removes it only if they rewrote the description themselves.
- For anything a player can see or feel in-game (blocks, items, mobs, combat, movement, particles, sounds, GUIs, world generation), a play-test checklist in "do X, expect Y" form, plus what the human should capture.

### Tell the author what they have to do

**This handover is required for every PR.** When you hand the branch over, say this to the author directly in your final message. Don't leave it buried in the description draft. Agents tend to report "done" and the author opens the PR without these steps, so spell them out every time.

1. **Play test it.** For gameplay changes, the author has to join with a real client and go through your checklist before opening the PR. Your boot and bot tests don't replace this, because some things only show up on a real client.
2. **Attach a screenshot or screen recording.** For anything a player can see or feel in-game, a PR without one isn't ready to open and will be sent back. A recording is better for anything that moves or happens over time. Tell the author exactly what to capture. They record it, because only they are running the client.
3. **Open the PR themselves**, with your description draft (disclosure line included) and the capture attached.

If the change has no in-game effect (tooling, codecs, config, refactors), say that too, so the author knows why no capture is needed.

### Before handing over

- Search open PRs. If someone already has a nearly identical implementation, say so instead of preparing another one.
- One bug or one feature per PR. If a feature has clear stages, separate PRs are easier to review than one large one.
- If a PR is opened by an agent anyway, its title ends with `🤖🤖🤖`.
