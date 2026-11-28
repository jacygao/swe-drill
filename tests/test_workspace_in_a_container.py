"""
Find the workspace in a real container, whose image's WORKDIR is neither /app
nor a directory of this machine.

Only `docker exec` with no `-w` shows where the image starts a command, which
is where Harbor's Docker environment runs one for a task that sets no workdir.
"""

import asyncio
import subprocess

import pytest

from conftest import require_docker
from swe_drill.workspace import WorkspaceError, find_workspace, make_workspace, read_base, read_start_tree

# node:24 has git and bash. /app and /testbed exist too, so looking for either
# is caught.
REPOSITORY = """\
FROM node:24
RUN mkdir -p /gleam/src /app /testbed && cd /gleam \\
 && git init -q && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m base
WORKDIR /gleam
"""
INSIDE_IT = "FROM swe-drill-test/workdir:gleam\nWORKDIR /gleam/src\n"
# As a SlopCodeBench image: /app empty, owned by a user other than the one commands run as.
EMPTY_AND_SOMEONE_ELSES = "FROM node:24\nRUN mkdir /app && chown node:node /app\nWORKDIR /app\n"


def build(tag: str, dockerfile: str) -> str:
    built = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input=dockerfile,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, f"cannot build {tag}: {built.stderr}"
    return tag


@pytest.fixture(scope="module")
def images() -> dict[str, str]:
    require_docker()
    return {
        "gleam": build("swe-drill-test/workdir:gleam", REPOSITORY),
        "gleam/src": build("swe-drill-test/workdir:gleam-src", INSIDE_IT),
        "app": build("swe-drill-test/workdir:app", EMPTY_AND_SOMEONE_ELSES),
    }


@pytest.fixture
def container_image(request, images) -> str:
    return images[request.param]


@pytest.mark.parametrize("container_image", ["gleam"], indirect=True)
def test_the_workspace_is_the_images_workdir(container_environment):
    assert asyncio.run(find_workspace(container_environment)) == "/gleam"


@pytest.mark.parametrize("container_image", ["gleam/src"], indirect=True)
def test_a_workdir_inside_the_repository_fails_naming_it(container_environment):
    with pytest.raises(WorkspaceError) as raised:
        asyncio.run(find_workspace(container_environment))
    assert "/gleam/src" in str(raised.value), "the error does not name the directory"


@pytest.mark.parametrize("container_image", ["app"], indirect=True)
def test_an_empty_workdir_another_user_owns_becomes_a_repository_git_will_use(container_environment):
    workspace = asyncio.run(make_workspace(container_environment))

    assert workspace == "/app"
    assert asyncio.run(read_base(container_environment, workspace))
    assert asyncio.run(read_start_tree(container_environment, workspace))
    # The agent's own commands run as the same user, and use git there too.
    status = asyncio.run(container_environment.exec("git status --porcelain && git log --oneline", cwd="/app"))
    assert status.return_code == 0, status.stdout
    # As squiz commits its head, with no system or global config to mark the directory safe.
    isolated = asyncio.run(
        container_environment.exec(
            "GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 git status --porcelain 2>&1", cwd="/app"
        )
    )
    assert isolated.return_code == 0, isolated.stdout
