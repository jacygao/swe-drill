import json
import re
import shlex
import uuid
from typing import Any

from harbor.agents.installed.base import NonZeroAgentExitCodeError, PackageSpec
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from swe_drill import sandbox
from swe_drill.agents import RECORD_NAME
from swe_drill.agents import copilot_cli
from swe_drill.agents.copilot_cli import CopilotCli, fetch

NODE_VERSION = "24.15.0"
# The sha256 of Node's archive of NODE_VERSION for each build, as the SHASUMS256.txt beside it gives it.
# nodejs.org builds only for glibc, so the builds for musl are its unofficial-builds site's.
NODE_SHA256 = {
    "x64": "44836872d9aec49f1e6b52a9a922872db9a2b02d235a616a5681b6a85fec8d89",
    "arm64": "73afc234d558c24919875f51c2d1ea002a2ada4ea6f83601a383869fefa64eed",
    "x64-musl": "f55af5bd489c5347b113ca6594cae00a54b30ba57ac5875324311bfc6f4762e3",
    "arm64-musl": "31e98aa960a067da91edffd5d93bc46657b5d2a8029612c359f5f2ac0060152a",
}
NODE_SITE = "https://nodejs.org/dist"
NODE_MUSL_SITE = "https://unofficial-builds.nodejs.org/download/release"
# An image whose C library is musl, as Alpine's is, has its dynamic loader here.
MUSL_LOADERS = "/lib/ld-musl-*"
EXACT_VERSION = re.compile(r"\d+\.\d+\.\d+")
# The first squiz that keeps a round's findings when its state lock is slow to take, as on a loaded machine.
# Its reviewer also signs in from `copilotTokenFile`, since Copilot CLI gives its hooks no token.
LEAST_VERSION = (0, 3, 3)
SETTINGS = {"venue": "local", "reviewer": "copilot"}
# The values squiz's `thinking` takes.
REVIEWER_EFFORTS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
# `squiz review`'s exit statuses.
CLEAN, OPEN, FAILED, CLOSED_AT_BOUND, OUT_OF_TIME = 0, 2, 1, 3, 4


class CopilotCliSquiz(CopilotCli):
    """
    CopilotCli with squiz, at the arm's `squiz_version`, reviewing the
    agent's change in its local venue with Copilot CLI as reviewer, and the
    agent's own session answering what it finds.

    squiz only reviews. Once `copilot -p` has exited, the run waits on
    `squiz review`, and while threads are open resumes the agent's session
    with what it printed, as a session squiz wakes would be. So nothing is
    stopped or the patch captured until the episode closes.

    squiz keeps its settings in the workspace's .squiz.json and its state in
    .squiz/, which .git/info/exclude keeps out of the patch. Both are the
    agent's own, so each step of a task reviews its own change in an episode
    of its own, and its state is copied to that step's logs once the run ends.

    Copilot CLI gives its hooks no COPILOT_GITHUB_TOKEN, so squiz's reviewer
    signs in from a token file outside the workspace, which exists only while
    the agent runs.
    """

    KWARGS = (*CopilotCli.KWARGS, "squiz_version", "reviewer_model", "reviewer_effort")
    REQUIRED_KWARGS = ("version", "squiz_version")
    OWNED_PATHS = (".squiz.json", ".squiz")
    # squiz's hooks and its wake run only with Copilot CLI's experimental features on.
    GLOBAL_FLAGS = ("--experimental",)
    AFTER_AGENT_TIMED_OUT = "review-timed-out"
    # squiz tells a process running from one gone by `ps`, which a slim image lacks.
    SYSTEM_PACKAGES = {
        **CopilotCli.SYSTEM_PACKAGES,
        "ps": PackageSpec(
            commands=("ps",),
            packages={"apt-get": ("procps",), "dnf": ("procps-ng",), "yum": ("procps-ng",), "apk": ("procps",)},
        ),
        # Node's musl build links the C++ runtime, which glibc images carry and Alpine leaves out.
        "node_libraries": PackageSpec(commands=(), packages={"apk": ("libstdc++", "libgcc")}, always_install=True),
        # Alpine's `ps` is busybox's, which has no -p, so the check for "ps" finds one squiz cannot use.
        "procps": PackageSpec(commands=(), packages={"apk": ("procps",)}, always_install=True),
    }

    def __init__(self, *, squiz_version: Any, reviewer_model: Any = None, reviewer_effort: Any = None, **kwargs: Any):
        """
        Raises ValueError for a `squiz_version` that is not exact, as 0.3.3 is,
        or is before 0.3.3, and for a `reviewer_model` or `reviewer_effort`
        squiz would refuse.

        The reviewer runs on Copilot's own default model and squiz's default
        effort where the arm names none.
        """
        if reviewer_model is not None and (not isinstance(reviewer_model, str) or not reviewer_model.strip()):
            raise ValueError(f"reviewer_model must be a Copilot CLI model name, not {reviewer_model!r}")
        if reviewer_effort is not None and reviewer_effort not in REVIEWER_EFFORTS:
            raise ValueError(f"reviewer_effort must be one of {', '.join(REVIEWER_EFFORTS)}, not {reviewer_effort!r}")
        if not isinstance(squiz_version, str) or not EXACT_VERSION.fullmatch(squiz_version):
            raise ValueError(f"squiz_version must be an exact version, as 0.3.3 is, not {squiz_version!r}")
        if tuple(int(part) for part in squiz_version.split(".")) < LEAST_VERSION:
            raise ValueError(
                f"squiz_version {squiz_version} is before 0.3.3, the first squiz that keeps"
                " a round's findings when its state lock is slow to take"
            )
        super().__init__(
            squiz_version=squiz_version, reviewer_model=reviewer_model, reviewer_effort=reviewer_effort, **kwargs
        )
        self.reviewer = {
            key: value for key, value in (("model", reviewer_model), ("thinking", reviewer_effort)) if value is not None
        }
        self.squiz_version = squiz_version
        self.squiz_installed: str | None = None
        self._plugin: str | None = None
        self._token_file: str | None = None
        self._session: str | None = None
        self._review_exits: list[int] = []

    async def install(self, environment: BaseEnvironment) -> None:
        """
        Install Copilot CLI as CopilotCli does, then Node NODE_VERSION and
        squiz at its version, each under $HOME/.local/share/.

        squiz's own `bin/squiz` runs whatever `node` is on PATH, and its hooks
        inherit the agent's PATH, where `node` is the task's. So it is
        replaced by one that runs this Node, and linked into $HOME/.local/bin,
        where the resumed session runs `squiz reply` by name. Raises RuntimeError when
        the installed squiz reports another version.
        """
        await super().install(environment)
        await self.ensure_system_dependencies(environment, ("curl", "tar", "ps"))
        home = ((await self.exec_as_agent(environment, 'printf %s "$HOME"')).stdout or "").strip()
        node = f"{home}/.local/share/swe-drill/node-v{NODE_VERSION}"
        plugin = f"{home}/.local/share/squiz/{self.squiz_version}"
        musl = (await environment.exec(f"ls {MUSL_LOADERS} >/dev/null 2>&1")).return_code == 0
        if musl:
            await self.ensure_system_dependencies(environment, ("node_libraries", "procps"))
        site = NODE_MUSL_SITE if musl else NODE_SITE
        suffix = "-musl" if musl else ""
        checksums = " ".join(f"{build}) want={digest};;" for build, digest in NODE_SHA256.items())
        await self.exec_as_agent(
            environment,
            f"set -e; {fetch()}fail() {{ echo \"$1\"; exit 1; }}; "
            'case "$(uname -m)" in x86_64|amd64) arch=x64;; aarch64|arm64) arch=arm64;; '
            '*) fail "Node publishes no build for $(uname -m)";; esac; '
            f'build="$arch{suffix}"; '
            f'case "$build" in {checksums} *) fail "no checksum is pinned for Node on $build";; esac; '
            "dir=$(mktemp -d); trap 'rm -rf \"$dir\"' EXIT; cd \"$dir\"; "
            "fetch node.tar.gz"
            f' "{site}/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-$build.tar.gz"; '
            "sum=$(sha256sum node.tar.gz | cut -d ' ' -f 1); "
            f'[ "$sum" = "$want" ] || fail "the archive of Node {NODE_VERSION} has checksum $sum, not $want"; '
            "mkdir unpacked; tar -xzf node.tar.gz -C unpacked; "
            f"mkdir -p {shlex.quote(node.rsplit('/', 1)[0])}; rm -rf {shlex.quote(node)}; "
            f'mv "unpacked/node-v{NODE_VERSION}-linux-$build" {shlex.quote(node)}',
        )
        source = f"squiz-{self.squiz_version}"
        archive = f"https://codeload.github.com/jacygao/squiz/tar.gz/refs/tags/v{self.squiz_version}"
        shim = f'#!/bin/sh\nexec {shlex.quote(node + "/bin/node")} {shlex.quote(plugin + "/src/cli.ts")} "$@"\n'
        installed = await self.exec_as_agent(
            environment,
            f"set -e; {fetch()}dir=$(mktemp -d); trap 'rm -rf \"$dir\"' EXIT; cd \"$dir\"; "
            f"fetch squiz.tar.gz {shlex.quote(archive)}; "
            "mkdir unpacked; tar -xzf squiz.tar.gz -C unpacked; "
            f"{{ test -f unpacked/{source}/hooks/hooks.json && test -f unpacked/{source}/src/cli.ts; }}"
            f" || {{ echo 'the archive of squiz {self.squiz_version} holds no {source}/hooks/hooks.json"
            " and src/cli.ts'; exit 1; }; "
            f"mkdir -p {shlex.quote(plugin.rsplit('/', 1)[0])}; rm -rf {shlex.quote(plugin)}; "
            f"mv unpacked/{source} {shlex.quote(plugin)}; "
            f"printf %s {shlex.quote(shim)} > {shlex.quote(plugin + '/bin/squiz')}; "
            f"chmod 755 {shlex.quote(plugin + '/bin/squiz')}; "
            'mkdir -p "$HOME/.local/bin"; '
            f'ln -sfn {shlex.quote(plugin + "/bin/squiz")} "$HOME/.local/bin/squiz"; '
            '"$HOME/.local/bin/squiz" version',
        )
        # `squiz version` prints `squiz <version> <path>`, the version its plugin.json names.
        words = (installed.stdout or "").split()
        reported = words[1] if len(words) > 1 and words[0] == "squiz" else (installed.stdout or "").strip()
        if reported != self.squiz_version:
            raise RuntimeError(f"the archive of squiz {self.squiz_version} installed squiz {reported}")
        self.squiz_installed = reported
        self._plugin = plugin
        self._token_file = f"{home}/.local/share/swe-drill/copilot-token"

    def build_cli_flags(self) -> str:
        """The agent's flags, which name the session every fix resumes."""
        session = f" --session-id={self._session}" if self._session is not None else ""
        return f"{self._shared_flags()}{session}"

    def _shared_flags(self) -> str:
        if self._plugin is None:
            raise RuntimeError("CopilotCliSquiz ran before install placed squiz")
        return f"{super().build_cli_flags()} --plugin-dir {shlex.quote(self._plugin)}"

    async def _before_agent(self, environment: BaseEnvironment) -> None:
        """
        Write the token file, as the agent's user and readable by it alone,
        then the workspace's .squiz.json, and keep it and .squiz/ out of git.

        Raises NonZeroAgentExitCodeError, so the trial is graded with no change,
        when the workspace's own ignore rules keep either in the patch.
        """
        if self._token_file is None:
            raise RuntimeError("CopilotCliSquiz ran before install placed squiz")
        self._session = str(uuid.uuid4())
        self._review_exits = []
        token_file = shlex.quote(self._token_file)
        # The token reaches the file from the environment, so no command line holds it.
        written = await self.exec_as_agent(
            environment,
            f"umask 077 && mkdir -p \"$(dirname {token_file})\" && rm -f {token_file}"
            f' && printf %s "${self.TOKEN}" > {token_file}',
        )
        if written.return_code != 0:
            raise NonZeroAgentExitCodeError(f"cannot write the token file {self._token_file}")
        settings = shlex.quote(json.dumps({**SETTINGS, **self.reviewer, "copilotTokenFile": self._token_file}))
        result = await environment.exec(
            "fail() { echo \"$1\"; exit 1; }; "
            "exclude=$(git rev-parse --git-path info/exclude)"
            " || fail 'git cannot find the workspace'\"'\"'s info/exclude'; "
            'mkdir -p "${exclude%/*}" || fail "cannot make ${exclude%/*}"; '
            # The newline first, in case the file does not end in one. Once, though every step runs this.
            "for line in /.squiz.json /.squiz/; do grep -qxF \"$line\" \"$exclude\" 2>/dev/null"
            " || printf '\\n%s\\n' \"$line\" >> \"$exclude\" || fail \"cannot write $exclude\"; done; "
            # The repository's own .gitignore outranks info/exclude, and can unignore either.
            "{ git check-ignore -q .squiz.json && git check-ignore -q .squiz/; }"
            " || fail 'the workspace'\"'\"'s ignore rules keep .squiz.json or .squiz/, so squiz'\"'\"'s files"
            " would be in the patch'; "
            f"printf %s {settings} > .squiz.json || fail 'cannot write .squiz.json'",
            cwd=self._prepared[0],
        )
        if result.return_code != 0:
            raise NonZeroAgentExitCodeError((result.stdout or "").strip())

    async def _after_agent(self, environment: BaseEnvironment) -> str:
        """
        Run `squiz review` until the episode closes, resuming the agent's
        session on what it printed each time it leaves threads open.

        The trial is `ok` when the episode closes, clean or at squiz's bound,
        and `review-stopped` when squiz fails. A review that runs out of time
        is waited on again, since its round goes on, so only Harbor's agent
        timeout ends a review that never closes.

        Raises RuntimeError when the run left no base.json, which squiz's
        SessionStart hook writes, since then squiz never loaded and nothing
        will review the change; when `squiz review` exits with a status it
        never gives; and when a fix exits non-zero, as the agent's own run
        would fail.
        """
        local = self._local()
        loaded = await environment.exec(f"test -f {shlex.quote(local + '/base.json')}")
        if loaded.return_code != 0:
            raise RuntimeError(
                f"squiz never loaded: Copilot CLI exited and left no {local}/base.json,"
                " which squiz's SessionStart hook writes"
            )
        loop = shlex.quote(self._loop())
        await environment.exec(f"rm -rf {loop} && mkdir -p {loop}")
        while True:
            turn = len(self._review_exits) + 1
            review = shlex.quote(f"{self._loop()}/{turn}.review.txt")
            # Where squiz says why a review failed.
            errors = shlex.quote(f"{self._loop()}/{turn}.review.err")
            code = await self._exit_of(
                environment, f'export PATH="$HOME/.local/bin:$PATH"; squiz review > {review} 2> {errors}'
            )
            self._review_exits.append(code)
            if code in (CLEAN, CLOSED_AT_BOUND):
                return "ok"
            if code == FAILED:
                return "review-stopped"
            if code == OUT_OF_TIME:
                continue
            if code != OPEN:
                raise RuntimeError(f"squiz review exited {code}, a status it never gives")
            await self._fix(environment, review, shlex.quote(f"{self._loop()}/{turn}.fixer.jsonl"))

    async def _fix(self, environment: BaseEnvironment, review: str, stream: str) -> None:
        """
        Resume the agent's session with `squiz review`'s output, unedited, as
        its prompt, and with the agent's own flags.

        Copilot CLI's `--resume` fails on a session it cannot find, so a fix
        never runs in a fresh one.
        """
        model = f" --model={shlex.quote(self._parse_model_name(self.model_name)[1])}" if self.model_name else ""
        code = await self._exit_of(
            environment,
            'export PATH="$HOME/.local/bin:$PATH"; '
            f'copilot --prompt="$(cat {review})" --resume={self._session} --yolo'
            f" --disable-mcp-server github-mcp-server{model} {self._shared_flags()}"
            f" --output-format=json > {stream} 2>&1 </dev/null",
        )
        if code != 0:
            raise RuntimeError(f"the fix resuming session {self._session} exited {code}")

    async def _exit_of(self, environment: BaseEnvironment, command: str) -> int:
        """
        Run `command` in the workspace as the agent, and return its exit status.

        Harbor raises on any status but 0, so the shell prints the status and exits 0.
        """
        result = await self.exec_as_agent(environment, f"{command}; printf '\\n%s' $?", cwd=self._prepared[0])
        return int((result.stdout or "").rsplit("\n", 1)[-1])

    async def _keep_owned(self, environment: BaseEnvironment) -> None:
        """
        Copy squiz's settings, and its state less its objects, to the logs,
        before the adapter removes them.

        The token file goes first, since a cancel during the copy, at the
        cleanup time limit, would otherwise leave it.
        """
        await self._remove_token(environment)
        kept = shlex.quote(self._kept())
        local = shlex.quote(self._local())
        settings = shlex.quote(f"{self._prepared[0]}/.squiz.json")
        await sandbox.run(
            environment,
            f"rm -rf {kept} && {{ [ ! -d {local} ] || {{ cp -R {local} {kept} && rm -rf {kept}/objects; }}; }}"
            f" && {{ [ ! -f {settings} ] || cp {settings} {shlex.quote(self._kept() + '.json')}; }}",
        )

    async def _record(self, environment: BaseEnvironment, status: str, run_seconds: float) -> None:
        """
        Remove the token file, then record the trial as CopilotCli does.

        The run's reviews are cleared once recorded, so a later step's record
        holds its own alone, even where it is refused before it reviews.
        """
        await self._remove_token(environment)
        await super()._record(environment, status, run_seconds)
        self._review_exits = []

    async def _remove_token(self, environment: BaseEnvironment) -> None:
        if self._token_file is not None:
            await environment.exec(f"rm -f {shlex.quote(self._token_file)}")

    async def _usage(self, environment: BaseEnvironment) -> dict[str, Any]:
        """
        CopilotCli's usage, with `ai_credits` the whole run's: the agent's
        first run, each reviewer session's, and each fix's, which `squiz`
        breaks out. It is None where any of them is unknown, so no partial
        sum reads as the whole.

        A resumed session's credits count on from where it left off, so the
        fixes cost the last one's largest checkpoint less the agent's own.
        """
        usage = await super()._usage(environment)
        reviewer = await _sessions_usage(
            environment, f"find {shlex.quote(self._kept())} -path '*/session/session-state/*/events.jsonl' 2>/dev/null"
        )
        listed = await environment.exec(f"ls {shlex.quote(self._loop())} 2>/dev/null")
        turns = sorted(
            (name for name in (listed.stdout or "").split() if name.endswith(".fixer.jsonl")),
            key=lambda name: int(name.split(".", 1)[0]),
        )
        fixer: dict[str, Any] = {"ai_credits": 0.0, "reported_models": [], "turns": len(turns)}
        if turns:
            models: set[str] = set()
            last = None
            for name in turns:
                path = shlex.quote(f"{self._loop()}/{name}")
                events = copilot_cli.read_events(
                    (await environment.exec(f"grep -E {copilot_cli.EVENT_LINES} {path}")).stdout or ""
                )
                models.update(events["reported_models"])
                last = events["credits"]
            agent = usage["ai_credits"]
            fixer["ai_credits"] = None if last is None or agent is None else round(last - agent, 4)
            fixer["reported_models"] = sorted(models)
        parts = [usage["ai_credits"], reviewer["ai_credits"], fixer["ai_credits"]]
        usage["ai_credits"] = None if None in parts else round(sum(parts), 4)
        usage["squiz"] = {"reviewer": reviewer, "fixer": fixer}
        return usage

    async def _details(self, environment: BaseEnvironment) -> dict[str, Any]:
        """
        Which squiz ran, whether it loaded, and each `squiz review`'s exit
        status, which says how its episode ended.
        """
        loaded = await environment.exec(f"test -f {shlex.quote(self._kept() + '/base.json')}")
        return {
            "squiz": {
                "version": self.squiz_installed,
                "copilot_experimental": "--experimental" in self.build_cli_flags().split(),
                "loaded": loaded.return_code == 0,
                "review_exits": list(self._review_exits),
            }
        }

    def populate_context_post_run(self, context: AgentContext) -> None:
        """As CopilotCli does, with the whole run's credits suss.json records in place of the agent's alone."""
        super().populate_context_post_run(context)
        record = self.logs_dir / RECORD_NAME
        if not record.exists():
            return
        usage = json.loads(record.read_text()).get("usage") or {}
        context.metadata = {**(context.metadata or {}), "ai_credits": usage.get("ai_credits"), "squiz": usage.get("squiz")}

    def _local(self) -> str:
        return f"{self._prepared[0]}/.squiz/local"

    def _kept(self) -> str:
        return f"{self.sandbox_logs.rstrip('/')}/squiz"

    def _loop(self) -> str:
        """
        Where each `squiz review`'s output and errors and each fix's stream are
        kept, as `<n>.review.txt`, `<n>.review.err` and `<n>.fixer.jsonl`.
        """
        return f"{self.sandbox_logs.rstrip('/')}/squiz-loop"


async def _sessions_usage(environment: BaseEnvironment, listing: str) -> dict[str, Any]:
    """
    The credits and models of every Copilot CLI session whose events.jsonl
    `listing` prints. The credits are None where any session reported none,
    since its cost is unknown, not nothing.
    """
    paths = sorted(((await environment.exec(listing)).stdout or "").split())
    credits: float | None = 0.0
    models: set[str] = set()
    for path in paths:
        result = await environment.exec(f"grep -E {copilot_cli.EVENT_LINES} {shlex.quote(path)}")
        events = copilot_cli.read_events(result.stdout or "")
        credits = None if credits is None or events["credits"] is None else credits + events["credits"]
        models.update(events["reported_models"])
    return {
        "ai_credits": None if credits is None else round(credits, 4),
        "reported_models": sorted(models),
        "sessions": len(paths),
    }
