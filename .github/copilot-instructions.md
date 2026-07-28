# Copilot Instructions

This file is maintained by the Agerpoint BOK and installed via `cd ~/agerpoint/bok/agents && make install`. Do not edit in place — edit the source at `agents/github-copilot/copilot-instructions.md` in the `agerpoint/bok` repo and re-run `make install` to propagate.

## Agerpoint conventions

This repo follows the Agerpoint engineering standards. The BOK is cloned to `~/agerpoint/bok/` and this repo is symlinked to `~/agerpoint/<repo>/` before every Copilot session (see `.github/workflows/copilot-setup-steps.yml`). Read these when relevant to your change:

- `~/agerpoint/bok/agents/AGENTS.md` — Agent meta: where to load standards from, plus the verification canary.
- `~/agerpoint/bok/standards/agents.md` — AI agent habits and principles.
- `~/agerpoint/bok/standards/git.md` — Branch naming, commit messages, PR titles, Shortcut integration.
- `~/agerpoint/bok/standards/code-style.md` — Naming, formatting, imports.
- `~/agerpoint/bok/standards/testing.md` — Coverage targets, test structure.
- `~/agerpoint/bok/standards/code-review.md` — Review expectations.
- `~/agerpoint/bok/standards/architecture.md` — Layering, DI, API design.
- `~/agerpoint/bok/standards/documentation.md` — What to document, ADR format, TODO conventions.
- `~/agerpoint/bok/standards/security.md` — Auth, encryption, input validation, checklists.
- Language-specific: `~/agerpoint/bok/standards/<python|typescript|dotnet|kotlin|swift>/` as applicable.
- Platform-specific: `~/agerpoint/bok/standards/<kubernetes|airflow|databricks>/` when working with cluster manifests, Flux, OpenTofu / Terraform, Helm, or pipeline platforms.
- Architecture: `~/agerpoint/bok/architecture/*.md` for system-level context.
- Past decisions: `~/agerpoint/bok/adrs/*.md` for Architecture Decision Records.

If `~/agerpoint/bok/` is missing, the setup-step workflow failed — surface this rather than proceeding with only repo-local context.

## Project context

For repo-specific context (purpose, tech stack, off-limits directories, test commands, deployment constraints, version-bump rules), read the AI entry point at the repo root that matches your tool:

- `AGENTS.md` — cross-tool convention (OpenAI Codex, Cursor, Aider, Zed, Sourcegraph); also the default for agents without a tool-specific file.
- `CLAUDE.md` — Claude Code.
- `GEMINI.md` — Gemini CLI.

If the matching file isn't present at the repo root, fall back to `README.md` and `CONTRIBUTING.md`.

## Shortcut workflow

Any ticket ID of the form `sc-<id>` refers to a Shortcut story. Follow the Ticket-to-PR flow in `~/agerpoint/bok/standards/ticket-to-pr.md` and the `[sc-<id>]` PR-title convention in `~/agerpoint/bok/standards/git.md#shortcut-integration`.
