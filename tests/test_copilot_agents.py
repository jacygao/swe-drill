"""
Run CopilotCli and CopilotCliHve through Harbor's own run, in the
stand-in environment, with a stand-in `copilot` that records how it was
started and prints the JSON lines a test gives it.

Harbor's run writes the stream under the agent's environment_logs_dir, which
each test points at its own sandbox directory. The stand-in's --help lists no
--usage-output-file, as an older Copilot CLI's does, unless a test gives it a
help.txt.
"""

import asyncio
import hashlib
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import pytest
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.copilot_cli import CopilotCli as HarborCopilotCli
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig as TrialAgentConfig

from swe_drill.agents.copilot_cli import CopilotCli
from swe_drill.agents import squiz as squiz_agent_module
from swe_drill.agents.hve import CopilotCliHve, full_auto_prompt
from swe_drill.agents.squiz import CopilotCliSquiz
from swe_drill.config import ConfigError
from swe_drill.agents import copilot_cli as copilot_module

TOKEN = "github_pat_from_the_env_file"
INSTRUCTION = "Fix the bug.\n"
COMMIT = "55352bb8e9f0622ff40f5c6afa80363adaf6b07f"
FULL_AUTO = "Run every phase in full auto mode."
# The real install, taken before the box fixture stubs it out.
INSTALL = CopilotCli._install_copilot
AGENT_FILE = f"""---
name: RPI Agent
handoffs:
  - label: "Research"
    agent: rpi-research
    prompt: "Research only."
  - label: "Full Auto"
    agent: rpi-agent
    prompt: "{FULL_AUTO}"
    send: true
---
The agent.
"""


def checkpoint(nano: int, model: str = "gpt-6.1-sol", effort: str = "high") -> dict:
    return {
        "type": "session.usage_checkpoint",
        "data": {
            "totalNanoAiu": nano,
            "promptCacheBreakState": [{"models": {model: {"reasoning_effort": effort}}}],
            # As Copilot CLI 1.0.105 reports a session's tokens, per model.
            "providerModelMetrics": [
                {"modelId": model, "metrics": {"usage": {
                    "inputTokens": 400002, "outputTokens": 4489, "cacheReadTokens": 395564}}},
                {"modelId": "claude-haiku-4.5", "metrics": {"usage": {
                    "inputTokens": 40000, "outputTokens": 11, "cacheReadTokens": 0}}},
            ],
        },
    }


STREAM = [
    {"type": "user.message", "data": {"content": INSTRUCTION}},
    {"type": "model.call_start", "data": {"model": "gpt-6.1-sol"}},
    {"type": "assistant.message", "data": {"content": "Fixing it.", "outputTokens": 12}},
    checkpoint(12_340_000_000),
    {"type": "model.call_start", "data": {"model": "gpt-6.1-sol"}},
    checkpoint(18_439_490_000),
    {"type": "result", "exitCode": 0},
]

# Records its argv, prints the stream, writes usage.json where it is given a
# usage file, and makes a change in the workspace.
COPILOT = """#!{python}
import json, sys
from pathlib import Path
here = Path(__file__).parent
if sys.argv[1:] == ["--help"]:
    help_text = here / "help.txt"
    print(help_text.read_text() if help_text.exists() else "Usage: copilot [options]")
    sys.exit(0)
(here / "copilot.argv").write_text(json.dumps(sys.argv[1:]))
if sys.argv[1:] == ["--version"]:
    print("GitHub Copilot CLI 1.0.105.")
    sys.exit(0)
print((here / "stream.jsonl").read_text(), end="")
usage = [arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--usage-output-file=")]
if usage and (here / "usage.json").exists():
    Path(usage[0]).write_text((here / "usage.json").read_text())
with open("hello.txt", "a") as f:
    f.write("a known change\\n")
sys.exit(int((here / "exit").read_text()) if (here / "exit").exists() else 0)
"""
# macOS has no stdbuf, which Harbor's run pipes the stream through.
STDBUF = '#!/bin/sh\nshift\nexec "$@"\n'
# Saves the archive the test made wherever -o says, and records the URL.
CURL = """#!/bin/sh
here=$(dirname "$0")
while [ $# -gt 1 ]; do
    [ "$1" = -o ] && out=$2
    shift
done
echo "$1" >> "$here/curl.urls"
cp "$here/hve-core.tar.gz" "$out"
"""


def go(coroutine):
    return asyncio.run(coroutine)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def box(tmp_path, local_environment, monkeypatch):
    """swe-drill.toml and its .env on the host, and a workspace, a home and logs in the sandbox."""
    host = tmp_path / "host"
    download = host / "tasks" / "swe-rebench-07-2026" / "owner__repo-1"
    download.mkdir(parents=True)
    (download / "task.toml").write_text("[agent]\ntimeout_sec = 100\n")
    (host / "swe-drill.toml").write_text(
        'swe_rebench = "tasks/swe-rebench-07-2026"\nenvironment = "docker"\nk = 1\nconcurrency = 1\n'
        "\n"
        '[arms.copilot-gpt]\nagent = "swe_drill.agents.copilot_cli:CopilotCli"\nmodel = "gpt-6.1-sol"\n'
        'kwargs = { version = "1.0.105", reasoning_effort = "high" }\n'
    )
    (host / ".env").write_text(f"COPILOT_GITHUB_TOKEN={TOKEN}\nOTHER_KEY=not-for-copilot\n")
    workspace = tmp_path / "sandbox" / "gleam"
    workspace.mkdir(parents=True)
    git(workspace, "init", "-q")
    (workspace / "hello.txt").write_text("hello\n")
    git(workspace, "add", "-A")
    git(workspace, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    home = tmp_path / "sandbox" / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    for name, text in (("copilot", COPILOT.format(python=sys.executable)), ("stdbuf", STDBUF), ("curl", CURL)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    write_stream(bin_dir, STREAM)
    local_environment.task_env_config.workdir = str(workspace)
    local_environment._persistent_env = {"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin"}
    local_environment._resolve_user = lambda user: None if user == "root" else user
    # The install downloads from npm's registry, which no test reaches.
    monkeypatch.setattr(CopilotCli, "_install_copilot", _no_install)
    monkeypatch.setattr(HarborCopilotCli, "ensure_system_dependencies", _no_install)
    return tmp_path


async def _no_install(*args, **kwargs):
    pass


def write_stream(bin_dir: Path, events: list) -> None:
    (bin_dir / "stream.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))


def bin_dir(box: Path) -> Path:
    return box / "sandbox" / "home" / ".local" / "bin"


def logs(box: Path) -> Path:
    return box / "sandbox" / "logs" / "agent"


def agent(box: Path, cls=CopilotCli, **kwargs):
    kwargs.setdefault("version", "1.0.105")
    kwargs.setdefault("model_name", "gpt-6.1-sol")
    return cls(
        arm="copilot-gpt",
        lane="rebench",
        drill_config=str(box / "host" / "swe-drill.toml"),
        logs=str(logs(box)),
        artifacts=str(box / "sandbox" / "logs" / "artifacts"),
        logs_dir=logs(box),
        environment_logs_dir=PurePosixPath(logs(box)),
        **kwargs,
    )


def run(box: Path, env, the_agent=None) -> tuple[CopilotCli, AgentContext, list]:
    """Both phases and Harbor's post-run parse, with each exec's command and env recorded."""
    the_agent = the_agent or agent(box, reasoning_effort="high")
    seen = []
    exec_as_given = env.exec

    async def recording(command, env=None, **kwargs):
        seen.append((command, dict(env or {})))
        return await exec_as_given(command, env=env, **kwargs)

    env.exec = recording
    try:
        logs(box).mkdir(parents=True, exist_ok=True)
        go(the_agent.setup(env))
        context = AgentContext()
        try:
            go(the_agent.run(INSTRUCTION, env, context))
        finally:
            the_agent.populate_context_post_run(context)
    finally:
        del env.exec
    return the_agent, context, seen


def copilot_argv(box: Path) -> list[str]:
    return json.loads((bin_dir(box) / "copilot.argv").read_text())


def record(box: Path) -> dict:
    return json.loads((logs(box) / "suss.json").read_text())


def test_copilot_runs_with_builtin_mcps_web_tools_and_auto_update_off(box, local_environment):
    run(box, local_environment)

    argv = copilot_argv(box)
    for flag in ("--disable-builtin-mcps", "--no-auto-update", "--disable-client-telemetry",
                 "--no-ask-user", "--yolo"):
        assert flag in argv, f"copilot ran without {flag}"
    tools = argv[argv.index("--excluded-tools") + 1 : argv.index("--excluded-tools") + 3]
    assert tools == ["web_fetch", "web_search"]
    assert "--model=gpt-6.1-sol" in argv
    assert argv[argv.index("--effort") + 1] == "high"
    assert f"--prompt={INSTRUCTION}" in argv


# Regression test for #296: the agent's shells inherited the token, so `printenv` put it in the stream.
def test_copilot_keeps_its_token_out_of_the_agents_shells_296(box, local_environment):
    run(box, local_environment)

    argv = copilot_argv(box)
    assert "--secret-env-vars" in argv, "copilot ran without --secret-env-vars, so its shells inherit the token"
    assert argv[argv.index("--secret-env-vars") + 1] == "COPILOT_GITHUB_TOKEN"


def test_copilot_gets_the_env_files_token_alone_whatever_harbors_environment_holds(
    box, local_environment, monkeypatch
):
    for name, value in (("COPILOT_GITHUB_TOKEN", "from-harbor"), ("GH_TOKEN", "gh"), ("GITHUB_TOKEN", "gh2")):
        monkeypatch.setenv(name, value)

    _, _, seen = run(box, local_environment)

    passed = [env for command, env in seen if "copilot --prompt" in command]
    assert len(passed) == 1
    assert passed[0]["COPILOT_GITHUB_TOKEN"] == TOKEN
    assert set(passed[0]) == {"COPILOT_GITHUB_TOKEN", "SWE_DRILL_START_TOKEN"}, (
        "Copilot CLI's command was given a variable beside the arm's token"
    )
    setup = [env for command, env in seen if "copilot --version" in command]
    assert setup and "COPILOT_GITHUB_TOKEN" not in setup[0], "the setup phase was given the token"


@pytest.mark.parametrize("cls", [CopilotCli, CopilotCliHve])
def test_harbors_preflight_takes_every_kwarg_swe_drill_run_passes(box, cls):
    kwargs = {"version": "1.0.105", "reasoning_effort": "high", "arm": "copilot-gpt",
              "lane": "rebench", "drill_config": str(box / "host" / "swe-drill.toml")}
    if cls is CopilotCliHve:
        kwargs["hve_core_commit"] = COMMIT

    AgentFactory.run_preflight(TrialAgentConfig(
        import_path=f"{cls.__module__}:{cls.__name__}", model_name="gpt-6.1-sol", kwargs=kwargs
    ))


def test_an_effort_in_harbors_environment_does_not_reach_copilot(box, local_environment, monkeypatch):
    monkeypatch.setenv("COPILOT_CLI_EFFORT", "low")

    run(box, local_environment, agent(box))

    assert "--effort" not in copilot_argv(box)


def test_a_model_that_names_a_provider_fails_before_a_sandbox_starts(box):
    with pytest.raises(ConfigError, match="names a provider"):
        CopilotCli(
            arm="copilot-gpt", lane="rebench", drill_config=str(box / "host" / "swe-drill.toml"),
            logs_dir=logs(box), model_name="anthropic/claude-sonnet-5.5", version="1.0.105",
        )


def test_an_env_file_with_no_token_fails_before_a_sandbox_starts(box):
    (box / "host" / ".env").write_text("OTHER_KEY=x\n")

    with pytest.raises(ConfigError, match="COPILOT_GITHUB_TOKEN"):
        agent(box)


def test_the_credits_models_and_efforts_of_the_stream_reach_the_record_and_harbors_context(
    box, local_environment
):
    _, context, _ = run(box, local_environment)

    usage = {"ai_credits": 18.4395, "reported_models": ["gpt-6.1-sol"], "reported_efforts": ["high"]}
    assert record(box)["usage"] == usage
    assert context.metadata == usage
    assert (context.n_input_tokens, context.n_cache_tokens) == (440002, 395564)
    assert context.n_output_tokens == 12, "Harbor's own count was replaced"


# As Copilot CLI writes --usage-output-file, for the session STREAM reports.
USAGE = {
    "sessionStartTime": "2026-11-17T00:00:00Z",
    "totalPremiumRequestCost": 0,
    "modelMetrics": {
        "gpt-6.1-sol": {
            "usage": {"inputTokens": 400002, "outputTokens": 4489, "cacheReadTokens": 395564},
            "totalNanoAiu": 18_439_490_000,
        },
    },
}


def test_harbors_token_counts_from_copilots_usage_file_are_kept(box, local_environment):
    (bin_dir(box) / "help.txt").write_text("  --usage-output-file <path>  write usage on exit\n")
    (bin_dir(box) / "usage.json").write_text(json.dumps(USAGE))

    _, context, _ = run(box, local_environment)

    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (400002, 395564, 4489)
    assert context.metadata["ai_credits"] == 18.4395
    assert "copilot_usage" in context.metadata, "SWE-drill's usage replaced Harbor's metadata"


def test_harbors_trajectory_is_kept(box, local_environment):
    run(box, local_environment)

    trajectory = json.loads((logs(box) / "trajectory.json").read_text())
    assert [step["message"] for step in trajectory["steps"]][-1] == "Fixing it."


def test_the_record_names_the_version_copilot_reports_and_the_kwargs_asked_for(box, local_environment):
    run(box, local_environment)

    assert record(box)["agent"]["version"] == "1.0.105"
    assert record(box)["kwargs"] == {"version": "1.0.105", "reasoning_effort": "high"}
    assert record(box)["status"] == "ok"
    assert "a known change" in (logs(box) / "model.patch").read_text()


def test_a_copilot_that_exits_non_zero_is_harness_failed_with_its_change_captured(box, local_environment):
    (bin_dir(box) / "exit").write_text("1")

    with pytest.raises(NonZeroAgentExitCodeError):
        run(box, local_environment)

    assert record(box)["status"] == "harness-failed"
    assert "a known change" in (logs(box) / "model.patch").read_text()


# hve-core


def publish(box: Path, agent_file: str = AGENT_FILE, plugin: bool = True) -> None:
    """Make the archive the stand-in curl serves, laid out as GitHub's archive of a commit."""
    root = box / "archive" / f"hve-core-{COMMIT}"
    (root / ".github" / "agents" / "hve-core").mkdir(parents=True)
    (root / ".github" / "agents" / "hve-core" / "rpi-agent.agent.md").write_text(agent_file)
    if plugin:
        (root / "plugin.json").write_text('{"name": "hve-core"}')
    with tarfile.open(bin_dir(box) / "hve-core.tar.gz", "w:gz") as archive:
        archive.add(root, arcname=root.name)


def hve(box: Path, **kwargs) -> CopilotCliHve:
    return agent(box, CopilotCliHve, hve_core_commit=COMMIT, reasoning_effort="high", **kwargs)


# Fails as curl does on a dropped HTTP/2 stream, which its own --retry never retries, FAILS times first.
FLAKY_CURL = """#!/bin/sh
here=$(dirname "$0")
failed=$(cat "$here/curl.failed" 2>/dev/null || echo 0)
if [ "$failed" -lt FAILS ]; then
    echo $((failed + 1)) > "$here/curl.failed"
    echo "curl: (92) HTTP/2 stream 0 was not closed cleanly: INTERNAL_ERROR (err 2)" >&2
    exit 92
fi
""" + CURL.split("\n", 1)[1]


def npm_copilot(box: Path) -> Path:
    package = box / "npm" / "package"
    package.mkdir(parents=True)
    (package / "copilot").write_text("#!/bin/sh\necho 'GitHub Copilot CLI 1.0.105.'\n")
    (package / "copilot").chmod(0o755)
    with tarfile.open(bin_dir(box) / "hve-core.tar.gz", "w:gz") as archive:
        archive.add(package, arcname="package")
    return package


def flaky_curl(box: Path, monkeypatch, fails: int) -> None:
    (bin_dir(box) / "curl").write_text(FLAKY_CURL.replace("FAILS", str(fails)))
    monkeypatch.setattr(copilot_module, "FETCH_PAUSE_SECONDS", 0)


def test_copilot_cli_installs_through_a_dropped_download_282(box, local_environment, monkeypatch):
    package = npm_copilot(box)
    flaky_curl(box, monkeypatch, fails=2)

    go(INSTALL(agent(box), local_environment))

    assert (bin_dir(box) / "copilot").read_text() == (package / "copilot").read_text()
    assert (bin_dir(box) / "curl.failed").read_text().strip() == "2"


def test_copilot_cli_install_fails_once_every_download_try_fails(box, local_environment, monkeypatch):
    npm_copilot(box)
    flaky_curl(box, monkeypatch, fails=99)

    with pytest.raises(Exception, match="exit 92"):
        go(INSTALL(agent(box), local_environment))

    assert (bin_dir(box) / "curl.failed").read_text().strip() == str(copilot_module.FETCH_TRIES)


def test_copilot_cli_is_installed_from_npms_registry_not_githubs_releases(box, local_environment):
    package = box / "npm" / "package"
    package.mkdir(parents=True)
    (package / "copilot").write_text("#!/bin/sh\necho 'GitHub Copilot CLI 1.0.105.'\n")
    (package / "copilot").chmod(0o755)
    with tarfile.open(bin_dir(box) / "hve-core.tar.gz", "w:gz") as archive:
        archive.add(package, arcname="package")

    go(INSTALL(agent(box), local_environment))

    url = (bin_dir(box) / "curl.urls").read_text()
    assert re.fullmatch(
        r"https://registry\.npmjs\.org/@github/copilot-linux-(x64|arm64)/-/copilot-linux-\1-1\.0\.105\.tgz\n", url
    ), f"Copilot CLI came from {url!r}"
    assert (bin_dir(box) / "copilot").read_text() == (package / "copilot").read_text()


@pytest.mark.parametrize("version", ["latest", "1.0", "v1.0.107"])
def test_a_copilot_version_that_is_not_exact_is_refused_before_a_sandbox_starts(box, version):
    with pytest.raises(ValueError, match="exact version"):
        agent(box, version=version)


def test_a_fusion_tier_is_written_to_copilots_settings_before_it_runs(box, local_environment):
    run(box, local_environment, agent(box, model_name="fusion", fusion_tier="high"))

    settings = box / "sandbox" / "home" / ".copilot" / "settings.json"
    assert json.loads(settings.read_text()) == {"fusionTier": "high"}
    assert record(box)["kwargs"]["fusion_tier"] == "high", "suss.json does not say which tier ran"


def test_a_fusion_tier_never_reaches_copilot_as_a_flag(box, local_environment):
    run(box, local_environment, agent(box, model_name="fusion", fusion_tier="high"))

    assert "--fusion-tier" not in copilot_argv(box), (
        "Harbor's agent passed --fusion-tier, which Copilot CLI 1.0.107 does not take"
    )


@pytest.mark.parametrize(("model", "tier"), [("fusion", "turbo"), ("gpt-6.1-sol", "high")])
def test_a_fusion_tier_copilot_would_not_use_is_refused(box, model, tier):
    with pytest.raises(ConfigError, match="fusion_tier"):
        agent(box, model_name=model, fusion_tier=tier)


def plugin_dir(box: Path) -> Path:
    return box / "sandbox" / "home" / ".local" / "share" / "hve-core" / COMMIT


def test_hve_installs_the_commits_archive_and_runs_the_rpi_agent_on_the_full_auto_prompt(
    box, local_environment
):
    publish(box)

    run(box, local_environment, hve(box))

    assert (bin_dir(box) / "curl.urls").read_text() == (
        f"https://codeload.github.com/microsoft/hve-core/tar.gz/{COMMIT}\n"
    )
    assert (plugin_dir(box) / "plugin.json").exists()
    argv = copilot_argv(box)
    assert argv[argv.index("--plugin-dir") + 1] == str(plugin_dir(box))
    assert argv[argv.index("--agent") + 1] == "hve-core:rpi-agent"
    assert f"--prompt={FULL_AUTO}\n\n{INSTRUCTION}" in argv
    assert "--disable-builtin-mcps" in argv


def test_hve_keeps_rpis_files_in_the_logs_and_out_of_the_patch(box, local_environment):
    publish(box)
    copilot = bin_dir(box) / "copilot"
    copilot.write_text(copilot.read_text().replace(
        "print((here", 'Path(".copilot-tracking/plan.md").write_text("the plan\\n")\nprint((here'
    ))

    run(box, local_environment, hve(box))

    workspace = box / "sandbox" / "gleam"
    assert (workspace / ".copilot-tracking").is_symlink()
    assert (logs(box) / "copilot-tracking" / "plan.md").read_text() == "the plan\n"
    patch = (logs(box) / "model.patch").read_text()
    assert ".copilot-tracking" not in patch
    assert "a known change" in patch


def test_hve_refuses_a_workspace_that_already_has_copilot_tracking(box, local_environment):
    publish(box)
    (box / "sandbox" / "gleam" / ".copilot-tracking").mkdir()

    with pytest.raises(NonZeroAgentExitCodeError, match="already has .copilot-tracking"):
        run(box, local_environment, hve(box))

    assert not (bin_dir(box) / "copilot.argv").read_text().count("--prompt"), "copilot ran anyway"
    assert record(box)["status"] == "harness-failed"


def test_hve_refuses_a_workspace_whose_ignore_rules_keep_copilot_tracking(box, local_environment):
    publish(box)
    workspace = box / "sandbox" / "gleam"
    (workspace / ".gitignore").write_text("!.copilot-tracking\n")

    with pytest.raises(NonZeroAgentExitCodeError, match="ignore rules"):
        run(box, local_environment, hve(box))


def test_hve_install_fails_on_an_archive_with_no_plugin(box, local_environment):
    publish(box, plugin=False)

    with pytest.raises(RuntimeError, match="plugin.json"):
        go(hve(box).setup(local_environment))


@pytest.mark.parametrize("commit", ["main", COMMIT[:12], COMMIT.upper(), 55352])
def test_hve_refuses_a_commit_that_is_not_a_full_commit(box, commit):
    with pytest.raises(ValueError, match="40 lowercase hex"):
        agent(box, CopilotCliHve, hve_core_commit=commit)


def test_the_full_auto_prompt_is_read_from_its_own_handoff():
    assert full_auto_prompt(AGENT_FILE) == FULL_AUTO


@pytest.mark.parametrize(
    "text",
    [
        AGENT_FILE.replace('"Full Auto"', '"Manual"'),
        AGENT_FILE.replace(f'    prompt: "{FULL_AUTO}"\n', ""),
    ],
)
def test_an_agent_file_with_no_full_auto_prompt_is_refused(text):
    with pytest.raises(RuntimeError, match="Full Auto"):
        full_auto_prompt(text)


# squiz


SQUIZ_VERSION = "0.3.3"
# Records each call, writes squiz's base.json where squiz's hooks were loaded,
# as its SessionStart does, prints the stream, makes a change, and leaves
# squiz-host.sh running detached, as squiz's round host outlives the run. A
# resumed session is a fix: it records its prompt, prints fixer.jsonl and makes
# the fixer's change instead.
SQUIZ_COPILOT = """#!PYTHON
import json, os, shutil, subprocess, sys
from pathlib import Path
here = Path(__file__).parent
argv = sys.argv[1:]
if argv == ["--help"]:
    print("Usage: copilot [options]")
    sys.exit(0)
with open(here / "copilot.calls", "a") as calls:
    calls.write(json.dumps(argv) + "\\n")
if "--version" in argv:
    print("GitHub Copilot CLI 1.0.107.")
    sys.exit(0)
if any(arg.startswith("--resume=") for arg in argv):
    (here / "fix.argv").write_text(json.dumps(argv))
    with open(here / "fix.prompts", "a") as prompts:
        prompts.write(json.dumps(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--prompt="))) + "\\n")
    if (here / "fixer.jsonl").exists():
        print((here / "fixer.jsonl").read_text(), end="")
    with open("hello.txt", "a") as f:
        f.write("the fixer's change\\n")
    sys.exit(int((here / "fix.exit").read_text()) if (here / "fix.exit").exists() else 0)
(here / "copilot.argv").write_text(json.dumps(argv))
(here / "copilot.which").write_text(json.dumps({"node": shutil.which("node"), "squiz": shutil.which("squiz")}))
plugin = Path(argv[argv.index("--plugin-dir") + 1]) if "--plugin-dir" in argv else None
if plugin is not None and (plugin / "hooks" / "hooks.json").exists():
    Path(".squiz/local").mkdir(parents=True, exist_ok=True)
    Path(".squiz/local/base.json").write_text('{"commit": "x", "branches": []}')
print((here / "stream.jsonl").read_text(), end="")
with open("hello.txt", "a") as f:
    f.write("a known change\\n")
host = here / "squiz-host.sh"
if host.exists():
    subprocess.Popen(["/bin/sh", str(host)], start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
"""
# A round host that reviews for a moment, then closes the episode.
CLOSING_HOST = """sleep 1
mkdir -p .squiz/local
printf '{"threads": []}' > .squiz/local/threads.json
printf '2 rounds, 1 finding' > .squiz/local/summary.md
printf '{"head": "0e7db84", "exitStatus": 0, "openThreads": []}' > .squiz/local/closed.json
"""
# Serves Node's archive, from either of the hosts Node publishes on, or squiz's, from the test's own, by the URL.
SQUIZ_CURL = """#!/bin/sh
here=$(dirname "$0")
while [ $# -gt 1 ]; do
    [ "$1" = -o ] && out=$2
    shift
done
echo "$1" >> "$here/curl.urls"
case "$1" in
    https://nodejs.org/*|https://unofficial-builds.nodejs.org/*) cp "$here/node.tar.gz" "$out" ;;
    *) cp "$here/squiz.tar.gz" "$out" ;;
esac
"""
SHA256SUM = """#!PYTHON
import hashlib, sys
for name in sys.argv[1:]:
    with open(name, "rb") as f:
        print(hashlib.sha256(f.read()).hexdigest() + "  " + name)
"""
# Runs squiz's entry point, which the stand-in squiz makes a shell script.
FAKE_NODE = """#!/bin/sh
if [ "$1" = --version ]; then echo v24.15.0; exit 0; fi
exec /bin/sh "$@"
"""
# `review` prints review.<n>.out, and review.<n>.err to stderr, and exits review.<n>.exit, 0 where the test
# gave none, for its nth call.
FAKE_SQUIZ_CLI = """root=$(cd "$(dirname "$0")/.." && pwd)
[ "$1" = version ] && echo "squiz VERSION $root/bin/squiz"
if [ "$1" = review ]; then
    here="$HOME/.local/bin"
    n=$(( $(cat "$here/review.count" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$here/review.count"
    pwd > "$here/review.cwd"
    [ -f "$here/review.sleep" ] && sleep "$(cat "$here/review.sleep")"
    cat "$here/review.$n.out" 2>/dev/null
    cat "$here/review.$n.err" >&2 2>/dev/null
    exit "$(cat "$here/review.$n.exit" 2>/dev/null || echo 0)"
fi
"""


def node_arch() -> str:
    return {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]


def publish_squiz(box: Path, monkeypatch, version: str = SQUIZ_VERSION, musl: bool = False) -> None:
    """
    Make the archives of Node and of squiz the stand-in curl serves, and pin the stand-in Node's checksum.
    With `musl`, the sandbox is an image built on musl, as Alpine is, and Node's archive is the one for musl.
    """
    flavor = f"{node_arch()}-musl" if musl else node_arch()
    if musl:
        loader = box / "musl" / f"ld-musl-{platform.machine()}.so.1"
        loader.parent.mkdir()
        loader.touch()
        monkeypatch.setattr(squiz_agent_module, "MUSL_LOADERS", str(loader.parent / "ld-musl-*"))
    archives = box / "squiz-archives"
    node = archives / f"node-v24.15.0-linux-{flavor}"
    (node / "bin").mkdir(parents=True)
    (node / "bin" / "node").write_text(FAKE_NODE)
    (node / "bin" / "node").chmod(0o755)
    with tarfile.open(bin_dir(box) / "node.tar.gz", "w:gz") as archive:
        archive.add(node, arcname=node.name)
    root = archives / f"squiz-{SQUIZ_VERSION}"
    for path, text in (
        (".claude-plugin/plugin.json", f'{{"name": "squiz", "version": "{version}"}}'),
        ("hooks/hooks.json", '{"hooks": {}}'),
        ("bin/squiz", '#!/bin/sh\nexec node "$(dirname "$0")/../src/cli.ts" "$@"\n'),
        ("src/cli.ts", FAKE_SQUIZ_CLI.replace("VERSION", version)),
    ):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    with tarfile.open(bin_dir(box) / "squiz.tar.gz", "w:gz") as archive:
        archive.add(root, arcname=root.name)
    digest = hashlib.sha256((bin_dir(box) / "node.tar.gz").read_bytes()).hexdigest()
    monkeypatch.setattr(squiz_agent_module, "NODE_SHA256", {flavor: digest})
    for name, text in (("copilot", SQUIZ_COPILOT), ("curl", SQUIZ_CURL), ("sha256sum", SHA256SUM)):
        (bin_dir(box) / name).write_text(text.replace("PYTHON", sys.executable))
        (bin_dir(box) / name).chmod(0o755)


def squiz_agent(box: Path, **kwargs) -> CopilotCliSquiz:
    return agent(box, CopilotCliSquiz, squiz_version=SQUIZ_VERSION, reasoning_effort="high", **kwargs)


def squiz_host(box: Path, script: str) -> None:
    (bin_dir(box) / "squiz-host.sh").write_text(script)


def squiz_plugin(box: Path) -> Path:
    return box / "sandbox" / "home" / ".local" / "share" / "squiz" / SQUIZ_VERSION


def reviews(box: Path, *results: tuple[int, str]) -> None:
    """What each `squiz review` in turn exits and prints."""
    for n, (code, text) in enumerate(results, start=1):
        (bin_dir(box) / f"review.{n}.exit").write_text(str(code))
        (bin_dir(box) / f"review.{n}.out").write_text(text)


THREADS = "1 thread is open on 0e7db84.\n\n- hello.txt:2 — the greeting is doubled.\n"


def test_squiz_patch_is_saved_only_once_squiz_closes_after_copilot_exits(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)
    reviews(box, (2, THREADS), (0, "Reviewed clean.\n"))
    # Each review returns once the round host has closed the episode.
    (bin_dir(box) / "review.sleep").write_text("2")

    run(box, local_environment, squiz_agent(box))

    patch = (logs(box) / "model.patch").read_text()
    assert "the fixer's change" in patch, "the patch was saved before the fix changed the workspace"
    assert "a known change" in patch
    assert record(box)["status"] == "ok"
    assert record(box)["squiz"]["review_exits"] == [2, 0]
    assert (bin_dir(box) / "review.cwd").read_text().strip() == str(box / "sandbox" / "gleam")


def next_step(box: Path, env, the_agent) -> None:
    """The next step of a task, as Harbor runs it: the step's logs moved away, and the same agent run again."""
    shutil.move(logs(box), logs(box).with_name("agent-step-1"))
    logs(box).mkdir()
    go(the_agent.run(INSTRUCTION, env, AgentContext()))


def test_a_squiz_review_that_fails_keeps_what_it_printed_to_stderr(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    reviews(box, (1, ""))
    (bin_dir(box) / "review.1.err").write_text("squiz: no review ran: the head could not be committed\n")

    run(box, local_environment, squiz_agent(box))

    assert record(box)["status"] == "review-stopped"
    assert (logs(box) / "squiz-loop" / "1.review.err").read_text() == (
        "squiz: no review ran: the head could not be committed\n"
    )


def test_a_later_step_refused_for_squizs_paths_records_no_review(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)
    reviews(box, (0, "Reviewed clean.\n"))
    the_agent = squiz_agent(box)
    run(box, local_environment, the_agent)
    (box / "sandbox" / "gleam" / ".squiz.json").write_text("{}")

    with pytest.raises(NonZeroAgentExitCodeError, match="already has .squiz.json"):
        next_step(box, local_environment, the_agent)

    assert record(box)["status"] == "harness-failed"
    assert record(box)["squiz"]["review_exits"] == []


def test_squiz_adds_whichever_of_its_exclusions_info_exclude_lacks(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)
    exclude = box / "sandbox" / "gleam" / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("/.squiz/\n")

    run(box, local_environment, squiz_agent(box))

    assert record(box)["status"] == "ok"
    assert exclude.read_text().splitlines().count("/.squiz.json") == 1
    assert exclude.read_text().splitlines().count("/.squiz/") == 1


def test_squiz_reviews_every_step_of_a_task_in_an_episode_of_its_own(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)
    reviews(box, (2, THREADS), (0, "Reviewed clean.\n"), (0, "Reviewed clean.\n"))
    the_agent = squiz_agent(box)
    run(box, local_environment, the_agent)

    next_step(box, local_environment, the_agent)

    workspace = box / "sandbox" / "gleam"
    first = logs(box).with_name("agent-step-1")
    assert json.loads((first / "suss.json").read_text())["squiz"]["review_exits"] == [2, 0]
    assert record(box)["status"] == "ok"
    assert record(box)["squiz"]["review_exits"] == [0]
    assert record(box)["squiz"]["loaded"]
    assert (first / "squiz" / "base.json").exists() and (logs(box) / "squiz" / "base.json").exists()
    assert not (workspace / ".squiz").exists() and not (workspace / ".squiz.json").exists()
    step_patch = (logs(box) / "step.patch").read_text()
    assert ".squiz" not in step_patch and "a known change" in step_patch


def test_a_fix_resumes_the_agents_session_on_squiz_reviews_output_unedited(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)
    reviews(box, (2, THREADS), (2, "Still open.\n"), (0, ""))

    run(box, local_environment, squiz_agent(box))

    prompts = [json.loads(line) for line in (bin_dir(box) / "fix.prompts").read_text().splitlines()]
    # The shell's $(cat) drops the trailing newline, and nothing else.
    assert prompts == [THREADS.rstrip("\n"), "Still open."]
    agent_argv, fix_argv = copilot_argv(box), json.loads((bin_dir(box) / "fix.argv").read_text())
    session = next(arg.split("=", 1)[1] for arg in agent_argv if arg.startswith("--session-id="))
    assert f"--resume={session}" in fix_argv
    for flag in ("--experimental", "--yolo", "--model=gpt-6.1-sol", "--plugin-dir", "--disable-builtin-mcps"):
        assert flag in fix_argv, f"the fix ran without the agent's {flag}"
    assert not any(arg.startswith("--session-id") for arg in fix_argv)
    assert (logs(box) / "squiz-loop" / "1.review.txt").read_text() == THREADS


def test_a_review_closed_at_squizs_bound_is_ok(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    reviews(box, (3, "Closed at the round cap with 1 low thread open.\n"))

    run(box, local_environment, squiz_agent(box))

    assert record(box)["status"] == "ok"
    assert not (bin_dir(box) / "fix.argv").exists(), "a fix ran on a closed episode"


def test_a_review_out_of_time_is_waited_on_again_without_a_fix(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    reviews(box, (4, "Still under review.\n"), (0, ""))

    run(box, local_environment, squiz_agent(box))

    assert record(box)["squiz"]["review_exits"] == [4, 0]
    assert not (bin_dir(box) / "fix.argv").exists()


def test_a_review_status_squiz_never_gives_is_harness_failed(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    reviews(box, (127, ""))

    with pytest.raises(RuntimeError, match="127"):
        run(box, local_environment, squiz_agent(box))

    assert record(box)["status"] == "harness-failed"


def test_a_fix_that_exits_non_zero_is_harness_failed_with_its_change_captured(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    reviews(box, (2, THREADS))
    (bin_dir(box) / "fix.exit").write_text("1")

    with pytest.raises(RuntimeError, match="fix"):
        run(box, local_environment, squiz_agent(box))

    assert record(box)["status"] == "harness-failed"
    assert "the fixer's change" in (logs(box) / "model.patch").read_text()


def test_squiz_threads_summary_and_close_are_kept_under_agent(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)
    # The review returns once the round host has closed the episode.
    (bin_dir(box) / "review.sleep").write_text("2")

    run(box, local_environment, squiz_agent(box))

    kept = logs(box) / "squiz"
    assert (kept / "threads.json").read_text() == '{"threads": []}'
    assert (kept / "summary.md").read_text() == "2 rounds, 1 finding"
    assert (kept / "closed.json").exists()


def test_a_squiz_review_that_never_closes_is_recorded_as_review_timed_out(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    (bin_dir(box) / "review.sleep").write_text("30")
    the_agent = squiz_agent(box)

    async def under_harbors_timeout():
        await the_agent.setup(local_environment)
        await asyncio.wait_for(the_agent.run(INSTRUCTION, local_environment, AgentContext()), timeout=3)

    logs(box).mkdir(parents=True, exist_ok=True)
    with pytest.raises(TimeoutError):
        go(under_harbors_timeout())

    assert record(box)["status"] == "review-timed-out"
    assert record(box)["squiz"]["review_exits"] == []
    assert "a known change" in (logs(box) / "model.patch").read_text()


def test_a_squiz_review_that_fails_is_recorded_as_stopped(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    reviews(box, (1, ""))

    run(box, local_environment, squiz_agent(box))

    assert record(box)["status"] == "review-stopped"
    assert record(box)["squiz"]["review_exits"] == [1]


def test_squiz_is_configured_for_the_local_venue_with_copilot_reviewing_and_no_fixer(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)

    run(box, local_environment, squiz_agent(box))

    settings = json.loads((logs(box) / "squiz.json").read_text())
    assert settings == {
        "venue": "local",
        "reviewer": "copilot",
        "copilotTokenFile": str(box / "sandbox" / "home" / ".local" / "share" / "swe-drill" / "copilot-token"),
    }


def test_squizs_reviewer_runs_on_the_arms_reviewer_model_and_effort(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)

    run(box, local_environment, squiz_agent(box, reviewer_model="claude-sonnet-5.5", reviewer_effort="high"))

    settings = json.loads((logs(box) / "squiz.json").read_text())
    assert (settings["model"], settings["thinking"]) == ("claude-sonnet-5.5", "high")


@pytest.mark.parametrize("kwargs", [{"reviewer_effort": "huge"}, {"reviewer_model": ""}, {"reviewer_model": 5}])
def test_a_reviewer_setting_squiz_would_refuse_is_refused_before_a_sandbox_starts(box, kwargs):
    with pytest.raises(ValueError, match=r"reviewer_\w+ must be"):
        squiz_agent(box, **kwargs)


def test_a_squiz_sessions_model_and_effort_are_read_from_its_stored_events():
    # A session's stored events.jsonl, as squiz's reviewer leaves it, has no model.call_start.
    stored = "".join(json.dumps(event) + "\n" for event in (
        {"type": "session.start", "data": {"selectedModel": ""}},
        {"type": "session.model_change", "data": {"newModel": "claude-sonnet-5.5", "reasoningEffort": "high"}},
        {"type": "session.usage_record", "data": {"usage": {"model": "claude-sonnet-5.5", "outputTokens": 131}}},
    ))

    events = copilot_module.read_events(stored)

    assert events["reported_models"] == ["claude-sonnet-5.5"]
    assert events["reported_efforts"] == ["high"]
    assert re.search(copilot_module.EVENT_LINES.strip("'"), '"type": "session.usage_record"'), "the grep drops the record"
    assert re.search(copilot_module.EVENT_LINES.strip("'"), '"type": "session.model_change"'), "the grep drops the change"


def test_squizs_reviewer_gets_the_token_from_a_private_file_removed_after_the_run(
    box, local_environment, monkeypatch
):
    publish_squiz(box, monkeypatch)
    # Copilot CLI gives its hooks no COPILOT_GITHUB_TOKEN, so the round host reads the file .squiz.json names.
    squiz_host(box, (
        "file=$(sed -n 's/.*\"copilotTokenFile\": \"\\([^\"]*\\)\".*/\\1/p' .squiz.json)\n"
        f"cat \"$file\" > {bin_dir(box)}/token.seen\n"
        f"stat -c %a \"$file\" 2>/dev/null > {bin_dir(box)}/token.mode || stat -f %Lp \"$file\" > {bin_dir(box)}/token.mode\n"
    ) + CLOSING_HOST)

    reviews(box, (2, THREADS), (0, ""))

    run(box, local_environment, squiz_agent(box))

    token_file = box / "sandbox" / "home" / ".local" / "share" / "swe-drill" / "copilot-token"
    assert (bin_dir(box) / "token.seen").read_text() == TOKEN, "squiz's round host could not read the token"
    assert (bin_dir(box) / "token.mode").read_text().strip() == "600"
    assert not token_file.exists(), "the token file outlived the run"
    assert TOKEN not in (logs(box) / "model.patch").read_text()


def test_squizs_files_are_never_in_the_patch(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    # squiz's own .squiz/.gitignore is left out, so only SWE-drill's exclude keeps .squiz/ out.
    squiz_host(box, CLOSING_HOST + "printf '{}' > .squiz/local/state.json\n")
    reviews(box, (2, THREADS), (0, ""))

    run(box, local_environment, squiz_agent(box))

    patch = (logs(box) / "model.patch").read_text()
    assert ".squiz" not in patch, f"squiz's files reached the patch:\n{patch}"
    assert "the fixer's change" in patch


def test_squiz_refuses_a_workspace_that_already_has_squiz_files(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    (box / "sandbox" / "gleam" / ".squiz.json").write_text("{}")

    with pytest.raises(NonZeroAgentExitCodeError, match="already has .squiz"):
        run(box, local_environment, squiz_agent(box))

    assert not (bin_dir(box) / "copilot.argv").exists(), "copilot ran anyway"
    assert record(box)["status"] == "harness-failed"


def test_squiz_refuses_a_workspace_whose_ignore_rules_keep_squiz_files(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    (box / "sandbox" / "gleam" / ".gitignore").write_text("!.squiz.json\n")

    with pytest.raises(NonZeroAgentExitCodeError, match="ignore rules"):
        run(box, local_environment, squiz_agent(box))


def test_every_copilot_command_squiz_runs_passes_experimental_and_the_record_says_so(
    box, local_environment, monkeypatch
):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)

    run(box, local_environment, squiz_agent(box))

    calls = [json.loads(line) for line in (bin_dir(box) / "copilot.calls").read_text().splitlines()]
    assert len(calls) == 2, "expected the version check and the run"
    for argv in calls:
        assert "--experimental" in argv, f"copilot ran without --experimental: {argv}"
    argv = copilot_argv(box)
    assert argv[argv.index("--plugin-dir") + 1] == str(squiz_plugin(box))
    assert record(box)["squiz"]["copilot_experimental"] is True
    assert record(box)["squiz"]["loaded"] is True


def test_squizs_install_check_runs_copilot_with_experimental(box, local_environment):
    package = box / "npm" / "package"
    package.mkdir(parents=True)
    (package / "copilot").write_text(f'#!/bin/sh\necho "$@" > {bin_dir(box)}/installed.argv\n')
    (package / "copilot").chmod(0o755)
    with tarfile.open(bin_dir(box) / "hve-core.tar.gz", "w:gz") as archive:
        archive.add(package, arcname="package")

    go(INSTALL(squiz_agent(box), local_environment))

    assert (bin_dir(box) / "installed.argv").read_text().split() == ["--experimental", "--version"]


def test_a_run_whose_squiz_never_loaded_fails(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)
    the_agent = squiz_agent(box)
    go(the_agent.setup(local_environment))
    (squiz_plugin(box) / "hooks" / "hooks.json").unlink()
    logs(box).mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="base.json"):
        go(the_agent.run(INSTRUCTION, local_environment, AgentContext()))

    assert record(box)["status"] == "harness-failed"
    assert record(box)["squiz"]["loaded"] is False


def test_squiz_installs_node_and_squiz_at_their_pinned_versions(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    asked = []

    async def recording(self, environment, dependencies):
        asked.extend(dependencies)

    monkeypatch.setattr(HarborCopilotCli, "ensure_system_dependencies", recording)
    the_agent = squiz_agent(box)

    go(the_agent.setup(local_environment))

    assert (bin_dir(box) / "curl.urls").read_text().splitlines() == [
        f"https://nodejs.org/dist/v24.15.0/node-v24.15.0-linux-{node_arch()}.tar.gz",
        f"https://codeload.github.com/jacygao/squiz/tar.gz/refs/tags/v{SQUIZ_VERSION}",
    ]
    assert "ps" in asked, "squiz needs ps, which a slim image lacks"
    assert the_agent.squiz_installed == SQUIZ_VERSION
    shim = (squiz_plugin(box) / "bin" / "squiz").read_text()
    node = box / "sandbox" / "home" / ".local" / "share" / "swe-drill" / "node-v24.15.0" / "bin" / "node"
    assert str(node) in shim, "squiz would run on whatever node is on PATH"
    assert (bin_dir(box) / "squiz").resolve() == (squiz_plugin(box) / "bin" / "squiz").resolve()


# nodejs.org builds no Node for musl, which Alpine images are built on.
def test_squiz_installs_node_built_for_musl_on_an_image_built_on_musl(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch, musl=True)
    asked = []

    async def recording(self, environment, dependencies):
        asked.extend(dependencies)

    monkeypatch.setattr(HarborCopilotCli, "ensure_system_dependencies", recording)
    the_agent = squiz_agent(box)

    go(the_agent.setup(local_environment))

    assert (bin_dir(box) / "curl.urls").read_text().splitlines()[0] == (
        "https://unofficial-builds.nodejs.org/download/release/v24.15.0/"
        f"node-v24.15.0-linux-{node_arch()}-musl.tar.gz"
    )
    assert "node_libraries" in asked, "Node's musl build links libstdc++ and libgcc, which Alpine leaves out"
    assert "procps" in asked, "squiz runs ps -p, which Alpine's busybox ps lacks"
    assert the_agent.squiz_installed == SQUIZ_VERSION
    node = box / "sandbox" / "home" / ".local" / "share" / "swe-drill" / "node-v24.15.0" / "bin" / "node"
    assert str(node) in (squiz_plugin(box) / "bin" / "squiz").read_text()


def test_squizs_node_stays_off_the_agents_path(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    squiz_host(box, CLOSING_HOST)

    run(box, local_environment, squiz_agent(box))

    which = json.loads((bin_dir(box) / "copilot.which").read_text())
    assert which["node"] is None or "swe-drill" not in which["node"], "the task's node was replaced"
    assert which["squiz"] == str(bin_dir(box) / "squiz"), "a resumed session cannot run squiz reply by name"


def test_squiz_install_fails_on_a_node_archive_with_another_checksum(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    monkeypatch.setattr(squiz_agent_module, "NODE_SHA256", {node_arch(): "0" * 64})

    with pytest.raises(Exception, match="checksum"):
        go(squiz_agent(box).setup(local_environment))


def test_squiz_install_fails_when_the_archive_is_another_squiz_version(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch, version="0.3.0")

    with pytest.raises(RuntimeError, match="0.3.0"):
        go(squiz_agent(box).setup(local_environment))


# 0.3.2 drops a round's findings when its state lock is slow to take, and earlier versions' reviewers get no token.
@pytest.mark.parametrize("version", ["0.3.2", "0.3.1", "0.2.9", "0.1.0"])
def test_a_squiz_before_0_3_3_is_refused(box, version):
    with pytest.raises(ValueError, match="0.3.3"):
        agent(box, CopilotCliSquiz, squiz_version=version)


@pytest.mark.parametrize("version", ["latest", "0.3", "v0.3.1", 3])
def test_a_squiz_version_that_is_not_exact_is_refused_before_a_sandbox_starts(box, version):
    with pytest.raises(ValueError, match="exact version"):
        agent(box, CopilotCliSquiz, squiz_version=version)


def session_events(nano: int, model: str) -> str:
    return "".join(json.dumps(event) + "\n" for event in (
        {"type": "session.start", "data": {}},
        {"type": "model.call_start", "data": {"model": model}},
        checkpoint(nano // 2, model),
        checkpoint(nano, model),
    ))


def test_the_trials_credits_include_squizs_reviewer_sessions_and_the_fixes(box, local_environment, monkeypatch):
    publish_squiz(box, monkeypatch)
    files = {
        # A reviewer's COPILOT_HOME is its round's session directory.
        ".squiz/local/rounds/1/session/session-state/r1/events.jsonl": session_events(3_000_000_000, "gpt-6-astra"),
        ".squiz/local/rounds/2/session/session-state/r2/events.jsonl": session_events(1_500_000_000, "gpt-6-astra"),
    }
    script = "".join(
        f"mkdir -p {shlex.quote(str(Path(path).parent))}\nprintf %s {shlex.quote(text)} > {shlex.quote(path)}\n"
        for path, text in files.items()
    )
    squiz_host(box, script + CLOSING_HOST)
    reviews(box, (2, THREADS), (2, THREADS), (0, ""))
    # The resumed session's checkpoints count on from the agent's 18.43949.
    write_stream(bin_dir(box), STREAM)
    (bin_dir(box) / "fixer.jsonl").write_text(session_events(22_439_490_000, "gpt-6.1-sol"))

    _, context, _ = run(box, local_environment, squiz_agent(box))

    usage = record(box)["usage"]
    assert usage["ai_credits"] == pytest.approx(22.4395 + 4.5)
    assert usage["squiz"] == {
        "reviewer": {"ai_credits": 4.5, "reported_models": ["gpt-6-astra"], "sessions": 2},
        "fixer": {"ai_credits": 4.0, "reported_models": ["gpt-6.1-sol"], "turns": 2},
    }
    assert usage["reported_models"] == ["gpt-6.1-sol"]
    assert context.metadata["ai_credits"] == usage["ai_credits"], "Harbor's result leaves out the review"
    assert (logs(box) / "squiz-loop" / "2.fixer.jsonl").exists()


def test_a_squiz_session_that_reported_no_credits_leaves_the_trials_credits_unknown(
    box, local_environment, monkeypatch
):
    publish_squiz(box, monkeypatch)
    # A reviewer stopped before its first usage checkpoint.
    events = json.dumps({"type": "model.call_start", "data": {"model": "gpt-6-astra"}}) + "\n"
    path = ".squiz/local/rounds/1/session/session-state/r1/events.jsonl"
    squiz_host(box, f"mkdir -p {shlex.quote(str(Path(path).parent))}\nprintf %s {shlex.quote(events)} > {path}\n"
               + CLOSING_HOST)

    _, context, _ = run(box, local_environment, squiz_agent(box))

    usage = record(box)["usage"]
    assert usage["squiz"]["reviewer"]["ai_credits"] is None, "an unknown cost was counted as zero"
    assert usage["ai_credits"] is None, "a partial sum was recorded as the whole run's credits"
    assert context.metadata["ai_credits"] is None
