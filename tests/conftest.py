"""Shared fixtures: one isolated platform root and one end-to-end demo run.

The demo is the expensive part (it really trains subprocess jobs), so it is run
once per session and reused. ``client``/``controller`` depend on the demo, which
guarantees the session database is reset (``--fresh``) before any other test
touches it. Database-only tests use the function-scoped ``repo`` fixture.
"""

from __future__ import annotations

import pytest

from algo_agent import bootstrap
from algo_agent.cli.client import ControlClient
from algo_agent.config import load_settings
from algo_agent.storage.repository import Repository

METHODS = ["baseline_mean", "ridge", "gbr"]
EXTRAS = {"methods_hint": METHODS, "pause_between_iterations": True}


@pytest.fixture(scope="session")
def root(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("algo_agent_root")
    settings = load_settings(str(path))
    bootstrap.init_platform(settings)
    return str(path)


@pytest.fixture(scope="session")
def settings(root):
    return load_settings(root)


@pytest.fixture(scope="session")
def demo_report(settings):
    """Drive one complete offline iteration loop once for the whole session."""
    from algo_agent import demo

    client = ControlClient(settings, force_mode="embedded", extras=dict(EXTRAS))
    return demo.run_demo(settings, client=client, task_id="demo_task", fresh=True,
                         json_mode=True)


@pytest.fixture(scope="session")
def client(settings, demo_report):
    return ControlClient(settings, force_mode="embedded", extras=dict(EXTRAS))


@pytest.fixture(scope="session")
def controller(client):
    return client._controller()


@pytest.fixture(scope="session")
def demo_repo(controller):
    return controller.repo


@pytest.fixture()
def repo(settings):
    return Repository(settings.db_path)
