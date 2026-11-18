import os

import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

from app.agent import root_agent
from app.config import config

# Configurações do ambiente GCP (lidas de variáveis de ambiente / app/.env)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "")
DEPLOY_LOCATION = os.getenv("DEPLOY_LOCATION", "us-west1")
STAGING_BUCKET = os.getenv("STAGING_BUCKET", "")

if not PROJECT_ID or not STAGING_BUCKET:
    raise RuntimeError(
        "Defina GOOGLE_CLOUD_PROJECT e STAGING_BUCKET no ambiente antes de executar o deploy."
    )

vertexai.init(
    project=PROJECT_ID,
    location=DEPLOY_LOCATION,
    staging_bucket=STAGING_BUCKET,
)

adk_app = AdkApp(
    agent=root_agent,
    enable_tracing=True,
)

REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,adk]>=1.158.0",
    "google-adk>=2.9.0",
    "cloudpickle",
    "pydantic",
]

deploy_env_vars = {
    "GOOGLE_GENAI_USE_VERTEXAI": "TRUE",
    "GOOGLE_CLOUD_LOCATION": os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    **{
        key: val
        for key, val in {
            "MODEL_NAME": config.worker_model,
            "DATA_STORE_DATASEBRAE": config.datasebrae_data_store_id,
            "DATA_STORE_PORTFOLIO": config.portfolio_data_store_id,
            "DATA_STORE_SHAREPOINT": config.sharepoint_data_store_id,
            "DATA_STORE_ONEDRIVE": config.onedrive_data_store_id,
        }.items()
        if val
    },
}

remote_agent = agent_engines.create(
    agent_engine=adk_app,
    display_name="deep-search-inteligencia-mpes-sebrae teste local",
    description="Agente de planejamento e pesquisa profunda para Inteligência de Mercado e MPEs (Sebrae-SP).",
    requirements=REQUIREMENTS,
    extra_packages=["./app"],
    env_vars=deploy_env_vars,
)

print("✅ Deploy concluído!")
print("Resource Name:", remote_agent.resource_name)
