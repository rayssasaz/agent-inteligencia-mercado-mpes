# Copyright 2025 Google LLC
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

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file in the app directory
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Authentication Configuration:
# By default, uses AI Studio with GOOGLE_API_KEY from .env file.
# To use Vertex AI instead, set GOOGLE_GENAI_USE_VERTEXAI=TRUE in your .env
# and ensure you have Google Cloud credentials configured.

if os.getenv("GOOGLE_API_KEY"):
    # AI Studio mode (default): set explicitly so a stale env value cannot
    # silently override the detected auth mode.
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
else:
    # Vertex AI mode: Fall back to Google Cloud credentials
    import google.auth

    _, project_id = google.auth.default()
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id or "")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    # Vertex AI mode: always set explicitly (computed, not user-configurable).
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


def _clean_env(var_name: str) -> str | None:
    """Return environment variable value or None if unset, empty, or a placeholder."""
    val = (os.getenv(var_name) or "").strip()
    if not val or val.startswith("<") or "YOUR_" in val:
        return None
    return val


@dataclass
class ResearchConfiguration:
    """Configuration for research-related models, auth mode, and datastores.

    Attributes:
        critic_model (str): Model for evaluation tasks.
        worker_model (str): Model for working/generation tasks.
        max_search_iterations (int): Maximum search iterations allowed.
        use_vertex_ai (bool): Whether Vertex AI mode is active.
        datasebrae_data_store_id (str | None): Optional Vertex AI Search datastore ID.
        portfolio_data_store_id (str | None): Optional Vertex AI Search datastore ID.
        sharepoint_data_store_id (str | None): Optional Vertex AI Search datastore ID.
        onedrive_data_store_id (str | None): Optional Vertex AI Search datastore ID.
    """

    critic_model: str = os.getenv("MODEL_NAME")  # type: ignore[assignment]
    worker_model: str = os.getenv("MODEL_NAME")  # type: ignore[assignment]
    max_search_iterations: int = 3 # before: 5
    use_vertex_ai: bool = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"

    datasebrae_data_store_id: str | None = _clean_env("DATA_STORE_DATASEBRAE")
    portfolio_data_store_id: str | None = _clean_env("DATA_STORE_PORTFOLIO")
    sharepoint_data_store_id: str | None = _clean_env("DATA_STORE_SHAREPOINT")
    onedrive_data_store_id: str | None = _clean_env("DATA_STORE_ONEDRIVE")


config = ResearchConfiguration()
