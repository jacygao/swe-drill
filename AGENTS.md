# Working in this repository

## Skills

Load the skill that matches the job:

- `writing-specs`: before writing or editing anything under `docs/specs/`
- `writing-notes`: before writing or editing a note in `docs/notes/`
- `writing-tests`: before writing or changing a test
- `writing-issues`: before `gh issue create`
- `writing-pull-requests`: before `gh pr create`
- `working-issues`: before dispatching an issue to subagents
- `writing-agents-md` and `writing-rules`: before editing `AGENTS.md`,
  `CLAUDE.md` or `.claude/rules/`

The skills named here come from the `skills` plugin of the `jacygao/skills`
marketplace, which `.claude/settings.json` enables. They hold the method; this
file holds what is specific to SWE-drill, and they read it from here.

## Specs

`docs/specs/swe-drill-spec.md` is the specification. Its § 7, The project, fixes the
language and the `src/` layout.

## Checks

```bash
uv run pytest
```

## Arms

Name an arm `<agent>-<model>`: the harness's short name, then the model's id
as it carries its version, such as `copilot-gpt-6.1-sol` for Copilot CLI on
`gpt-6.1-sol`. A setting that makes another arm of the same pair goes last:
`copilot-fusion-high`. A name with a dot is quoted in `swe-drill.toml`:

```toml
[arms."copilot-gpt-6.1-sol"]
```

Rename an older arm to this form in the change that next runs it, not before.

## Silent failures

- **A senior regrade on litellm 1.106 or later records a reward with no judge
  run.** Each task's `test.sh` installs `litellm>=1.0,<2.0`, and every judge call
  then raises on the missing `mcp` module. Pin `litellm>=1.0,<1.106`.
- **A senior regrade without `ANTHROPIC_API_KEY` still completes.** Each judge
  records `skipped:no_api_key`. Give Harbor the key.

Read the note in `docs/notes/` that covers what you change before changing it.

## Running agents here

- **Never merge.** The owner merges.
