---
sidebar_position: 4
title: Model Providers
---

# Model Providers

Infinity Code doesn't talk to any LLM API directly. All model inference goes through **model providers**: separate processes that the daemon launches and manages. Each provider exposes one or more models (for example, the Bedrock provider exposes the Claude models available on AWS Bedrock).

At least one provider must be installed before the agent can start.

## Installing a provider

Providers are installed with `infinity provider install`, which `cargo install`s the provider crate and registers it in `~/.infinity/providers.json`:

```bash
infinity provider install bedrock --crate infinity-provider-bedrock
```

- The first argument (`bedrock`) is the **provider id**, a name you choose. Models are tracked as `provider id + model id`, so pick a stable name.
- `--crate` is the crate to install; its binary becomes the provider's command.
- By default the crate is installed from [crates.io](https://crates.io). Pass `--git <url>` or `--path <dir>` to install from a git repository or a local checkout instead. The source is recorded, exactly like `infinity rap install`, so the provider can be updated later.

### The Bedrock provider

`infinity-provider-bedrock` serves the Claude models available on AWS Bedrock. It uses your ambient AWS configuration through the standard SDK credential chain (`AWS_PROFILE`, environment variables, SSO, instance profiles), and requires Bedrock model access to be enabled on the account.

### The OpenRouter and Groq providers

`infinity-provider-openrouter` calls OpenRouter directly.
`infinity-provider-groq` calls Groq directly, with a separate API key and model list.
Both can be installed together. From the repository root:

```bash
infinity provider install openrouter --crate infinity-provider-openrouter --path "$(pwd)/crates/infinity-provider-openrouter"
infinity provider install groq --crate infinity-provider-groq --path "$(pwd)/crates/infinity-provider-groq"
```

These commands install from the local checkout; they do not require the crates to
have been published. Keep the checkout at a stable path for provider updates.

Export configuration in the environment that starts the daemon:

```bash
export OPENROUTER_API_KEY='your-openrouter-key'
export OPENROUTER_MODELS='anthropic/claude-sonnet-4.6'
export GROQ_API_KEY='your-groq-key'
export GROQ_MODELS='openai/gpt-oss-20b,openai/gpt-oss-120b'
infinity daemon restart
```

Configure only the providers you install. Each registered provider must start
successfully, so an unconfigured provider can prevent daemon startup. For a
service-managed daemon, set these variables in the service environment. Avoid
storing keys in source control or `providers.json`.

Model lists are comma-separated, preserve order, and use the backend's exact IDs.
The first valid entry is that provider's default; duplicate IDs are removed.
Both providers fetch model availability and context windows at startup. Invalid
individual entries are logged to stderr and skipped; if none remain, startup
fails with a diagnostic. Catalog/authentication/network failures also fail startup.

| Provider | Key | Model list | Optional timeout |
| --- | --- | --- | --- |
| OpenRouter | `OPENROUTER_API_KEY` | `OPENROUTER_MODELS` | `OPENROUTER_TIMEOUT_SECS` |
| Groq | `GROQ_API_KEY` | `GROQ_MODELS` | `GROQ_TIMEOUT_SECS` |

Timeouts default to 120 seconds and must be positive. They bound response-header
waiting and each SSE event, rather than the whole completion.

OpenRouter models must advertise tool support in its catalog. Groq's catalog does
not advertise tool capability; the adapter maintains a reviewed list of supported
chat models and output limits, while checking availability and context windows
against the live catalog. The current Groq list includes `openai/gpt-oss-20b`,
`openai/gpt-oss-120b`, `openai/gpt-oss-safeguard-20b`, `llama-3.3-70b-versatile`,
`llama-3.1-8b-instant`, `qwen/qwen3.8-27b`, `minimaxai/minimax-m2.7`, and
`meta-models/muse-spark-open`. Account access and model availability may differ.
Speech and embedding models cannot serve the coding agent.

Both adapters support streaming text, reasoning deltas, tool-call/result history,
and token accounting. Complete tool calls are emitted after the stream ends;
incremental tool-call UI deltas are not emitted. Groq input is currently text-only;
the runtime substitutes image placeholders. OpenRouter supports image input when
the selected model advertises it and preserves opaque reasoning details across
turns. Groq does not replay OpenRouter-specific reasoning history.

Backend-native options can be passed through `CompletionRequest.additional_params`.
For Groq GPT-OSS, use `reasoning_effort` and `include_reasoning`; `reasoning_format`
is unsupported. For OpenRouter, use its `reasoning` and `provider` options.
Protocol-owned model, messages, tools, streaming, and token-limit fields cannot be
overridden. GPT-OSS defaults to nonparallel tool calls on Groq.

To diagnose a startup failure, run `infinity-provider-openrouter` or
`infinity-provider-groq` directly. On success it prints the socket path and waits;
stop the diagnostic process before starting Infinity normally.

See the [OpenRouter model catalog](https://openrouter.ai/models),
[Groq models](https://console.groq.com/docs/models), and
[Groq tool support](https://console.groq.com/docs/tool-use/overview) for backend details.

## Configuration

Providers live in `~/.infinity/providers.json`, a JSON object mapping provider id to its configuration:

```json
{
  "bedrock": {
    "command": ["infinity-provider-bedrock"],
    "crate_name": "infinity-provider-bedrock"
  },
  "my-provider": {
    "command": ["/usr/local/bin/my-provider", "--some-flag"]
  }
}
```

- **`command`**: the command (argv) the daemon runs to start the provider. Bare names are looked up on `PATH`; absolute paths are used as-is.
- **`crate_name`**, **`git`**, **`path`** *(optional)*: installation source, recorded by `infinity provider install` and used by `infinity provider update`. A `crate_name` without `git` or `path` is installed from crates.io. Hand-written entries without `crate_name` still work; they just can't be auto-updated.

**Order matters**: providers are registered in the order they appear in the file, and the first model of the first provider is the default model for new sessions. You can edit the file by hand. The daemon reads it at startup, so run `infinity daemon restart` after changes.

## Switching models

The model picker (`/model` or `Ctrl+M` in the TUI) lists every model from every configured provider. Each thread remembers its own selection; if a selected model disappears from the configuration, the thread falls back to the default model with a warning.

## Updating

```bash
infinity provider update
```

re-installs every provider that has a recorded source (`crate_name` plus optional `git`/`path`). Providers are also updated as part of a full

```bash
infinity update
```

which updates the CLI binary, model providers, and RAP servers in one go.

## Troubleshooting

When the CLI launches the daemon in the background, it waits for the daemon to report readiness; if the daemon exits during startup instead, the CLI prints everything the daemon wrote to stdout/stderr, so configuration errors surface directly:

- **"no model providers configured"**: the daemon refuses to start without `~/.infinity/providers.json`. Install a provider (see above).
- **Provider fails to start**: if a configured provider can't be spawned (e.g. its binary was removed) or doesn't come up, the daemon exits with the provider's error. Details are also logged to `~/.infinity/daemon.log`.

## Building your own provider

Any process that implements the provider protocol can serve models. See [Model Providers in the Infinity Runtime docs](/docs/infinity-runtime/model-providers) for the `ModelProvider` trait and a guide to writing a provider crate.
