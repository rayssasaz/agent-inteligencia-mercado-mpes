# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runnability tests for the recipe."""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch


def _reload_app_modules(env_overrides: dict[str, str]) -> object:
    """Reload app.config and app.agent cleanly with specific environment variables."""
    for mod in list(sys.modules):
        if mod == "app" or mod.startswith("app."):
            sys.modules.pop(mod, None)

    with patch.dict(os.environ, env_overrides, clear=False):
        with patch("google.auth.default", return_value=(MagicMock(), "test-project")):
            import app.config
            import app.agent

            importlib.reload(app.config)
            importlib.reload(app.agent)
            return app.agent


def test_agent_runnability() -> None:
    """Verify agent.py imports and defines the expected globals."""
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
    os.environ.setdefault("MODEL_NAME", "gemini-3.5-flash")

    with patch("google.auth.default", return_value=(MagicMock(), "test-project")):
        import app.agent

    assert app.agent.root_agent is not None
    assert app.agent.app is not None


def test_ai_studio_mode_uses_web_search_only() -> None:
    """Verify AI Studio mode (GOOGLE_API_KEY, no datastores) loads with web search."""
    agent_mod = _reload_app_modules(
        {
            "GOOGLE_API_KEY": "test-api-key",
            "GOOGLE_GENAI_USE_VERTEXAI": "False",
            "MODEL_NAME": "gemini-3.5-flash",
            "DATA_STORE_DATASEBRAE": "",
            "DATA_STORE_PORTFOLIO": "",
            "DATA_STORE_SHAREPOINT": "",
            "DATA_STORE_ONEDRIVE": "",
        }
    )
    tool_names = [t.agent.name for t in agent_mod.SEBRAE_RESEARCH_TOOLS]
    assert tool_names == ["agente_web"]


def test_vertex_mode_with_datastores_loads_all_subagents() -> None:
    """Verify Vertex AI mode with configured datastores registers all 5 subagents."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GOOGLE_API_KEY", None)
        agent_mod = _reload_app_modules(
            {
                "GOOGLE_GENAI_USE_VERTEXAI": "True",
                "GOOGLE_CLOUD_PROJECT": "test-project",
                "MODEL_NAME": "gemini-3.5-flash",
                "DATA_STORE_DATASEBRAE": "projects/123/locations/global/collections/default_collection/dataStores/ds1",
                "DATA_STORE_PORTFOLIO": "projects/123/locations/global/collections/default_collection/dataStores/ds2",
                "DATA_STORE_SHAREPOINT": "projects/123/locations/global/collections/default_collection/dataStores/ds3",
                "DATA_STORE_ONEDRIVE": "projects/123/locations/global/collections/default_collection/dataStores/ds4",
            }
        )
    tool_names = [t.agent.name for t in agent_mod.SEBRAE_RESEARCH_TOOLS]
    assert tool_names == [
        "agente_datasebrae",
        "agente_portfolio",
        "agente_sharepoint",
        "agente_onedrive",
        "agente_web",
    ]
