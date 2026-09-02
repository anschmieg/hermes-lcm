# Configuration and activation

Hermes-LCM is a general Hermes plugin and a context engine. Both identities must be active:

```yaml
plugins:
  enabled:
    - hermes-lcm

context:
  engine: lcm
```

Restart Hermes after changing plugin or context-engine configuration. Verify with `hermes plugins`, then use `lcm_status` after a normal message has bound the session.

## Installation

An existing checkout can install profile-aware plugin and skill links:

```bash
./scripts/install.sh
HERMES_PROFILE=myprofile ./scripts/install.sh
```

The installer exposes both:

- `plugins/hermes-lcm` for plugin loading;
- `skills/hermes-lcm` for normal skill discovery.

It refuses conflicting paths rather than overwriting an existing install.

## High-impact controls

Use `docs/operator-guide.md` as the complete current source. Start with:

- `LCM_CONTEXT_THRESHOLD`: when normal context pressure triggers compaction;
- `LCM_FRESH_TAIL_COUNT`: newest messages kept raw;
- `LCM_LEAF_CHUNK_TOKENS`: maximum raw material per leaf compaction group;
- `LCM_DATABASE_PATH`: profile-local SQLite path when the default is unsuitable;
- `LCM_IGNORE_SESSION_PATTERNS` and `LCM_STATELESS_SESSION_PATTERNS`: storage ownership boundaries;
- summary/embedding provider settings only after confirming credentials, cost, and data handling.

Optional slash commands are disabled by default with `LCM_ENABLE_SLASH_COMMAND=false`. Destructive cleanup apply is separately guarded. Do not enable mutation surfaces merely to diagnose a problem.

## Compression model (auxiliary.compression)

LCM uses Hermes' auxiliary LLM for summarization. Configure in `config.yaml` under `auxiliary.compression`:

```yaml
auxiliary:
  compression:
    provider: custom:tokligence
    model: mistral/ministral-8b-latest
    timeout: 120
    fallback_chain:
      - provider: openai-codex
        model: gpt-5.6-luna
      - provider: custom:tokligence
        model: mistral/ministral-8b-latest
      - provider: openrouter
        model: openrouter/openai/gpt-5.6-luna
```

- Primary: cheap, fast Mistral/Ministral-class model via Tokligence.
- Fallback 1: GPT-5.6-Luna via OpenAI Codex for higher-quality middle recovery.
- Fallback 2: the cheap Tokligence route again when the Codex route is unavailable.
- Final fallback: Luna on OpenRouter as a paid reliability route that avoids free-tier rate limits.

Do not use `openrouter/free` as fallback for compression — it is too rate-limit-prone. Prefer a cheap paid OpenRouter route with enough summarization quality.

Change one tuning variable at a time, then re-check `lcm_status`, context pressure, summary health, latency, and actual answer quality.
