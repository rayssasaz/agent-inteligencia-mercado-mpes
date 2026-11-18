# Deep Search — Inteligência de Mercado e Apoio a MPEs (Sebrae-SP)

Agente multiagente de pesquisa profunda construído com **Google Agent Development Kit (ADK)** e **Gemini**, personalizado para gerar **Dossiês de Inteligência de Mercado e Apoio a Micro e Pequenas Empresas (MEI, ME e EPP)** no contexto do Sebrae-SP.

> **Nota sobre repositório público:** Este projeto foi estruturado para portfólio e demonstração técnica. **Nenhum dado interno, credencial, projeto GCP ou ID de datastore corporativo está exposto no código.** As conexões com bases corporativas via Vertex AI Search são opcionais e encapsuladas em variáveis de ambiente, permitindo rodar o agente integralmente em modo público com uma chave do Google AI Studio e busca web.

---

## Principais Funcionalidades

- **Planejamento Human-in-the-Loop (HITL):** O agente elabora um plano de pesquisa estruturado com metas `[RESEARCH]` e `[DELIVERABLE]` e só inicia a execução após aprovação explícita do usuário.
- **Arquitetura Multipesquisa por Especialistas:** Orquestra subagentes especialistas por fonte (`AgentTool`):
  - **`agente_web`** (`google_search`): Sempre ativo; coleta tendências de mercado, cenário econômico, regulação e notícias regionais.
  - **Subagentes Corporativos (opcionais, via `VertexAiSearchTool`):** `agente_datasebrae`, `agente_portfolio`, `agente_sharepoint` e `agente_onedrive` — ativados automaticamente apenas quando o modo Vertex AI e os respectivos `DATA_STORE_*` estão configurados.
- **Loop Iterativo de Avaliação e Refinamento:** Um avaliador crítico (`research_evaluator`) audita os achados com schema estruturado (`Feedback`), aponta lacunas por fonte (`target_source`) e aciona buscas complementares até atingir o critério de aprovação ou o limite `max_search_iterations`.
- **Relatório com Citações Inline:** Callbacks (`collect_research_sources_callback` e `citation_replacement_callback`) capturam metadados de grounding e convertem tags `<cite source="src-N"/>` em links Markdown rastreáveis.

---

## Fluxo da Arquitetura

```
Tema do usuário
  -> interactive_planner_agent (LlmAgent raiz - HITL)
       aciona plan_generator (AgentTool) -> plano de 5 metas
       refina com o usuário até aprovação explícita
       | (aprovado)
       v
  research_pipeline (SequentialAgent)
       1. section_planner               -> esboço do dossiê (state: report_sections)
       2. section_researcher            -> primeira rodada multipesquisa (state: section_research_findings)
       3. iterative_refinement_loop (LoopAgent, max = 3)
            research_evaluator          -> nota pass/fail + follow_up_queries por target_source
            escalation_checker          -> encerra o loop quando grade == "pass"
            enhanced_search_executor    -> executa buscas complementares e consolida achados
       4. report_composer_with_citations -> dossiê final em português com citações inline
```

---

## Como Rodar Localmente

### Pré-requisitos

- Python `>=3.11, <3.13`
- [uv](https://docs.astral.sh/uv/) para gerenciamento de dependências
- (Opcional) [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) caso utilize Vertex AI e datastores corporativos

### 1. Instalar dependências

```bash
uv sync
```

### 2. Configurar variáveis de ambiente (`app/.env`)

Copie o arquivo [`.env.example`](.env.example) para `app/.env` e escolha um dos modos abaixo:

#### Modo A — Público / Portfólio (AI Studio + Busca Web)
Ideal para testar localmente sem precisar de projeto GCP ou datastores:

```env
GOOGLE_API_KEY=SUA_CHAVE_DO_AI_STUDIO
GOOGLE_GENAI_USE_VERTEXAI=False
MODEL_NAME=gemini-3.5-flash
```

#### Modo B — Corporativo (Vertex AI + Datastores)
Para conectar aos datastores privados via Application Default Credentials (ADC):

1. Autentique sua conta no terminal:
   ```bash
   gcloud auth application-default login
   ```
2. Em `app/.env`, **remova ou deixe em branco** `GOOGLE_API_KEY` e preencha:
   ```env
   MODEL_NAME=gemini-3.5-flash
   GOOGLE_CLOUD_PROJECT=<SEU_PROJECT_ID_GCP>
   GOOGLE_CLOUD_LOCATION=global

   DATA_STORE_DATASEBRAE=projects/<PROJECT_NUMBER>/locations/global/collections/default_collection/dataStores/<ID>
   DATA_STORE_PORTFOLIO=projects/<PROJECT_NUMBER>/locations/global/collections/default_collection/dataStores/<ID>
   DATA_STORE_SHAREPOINT=projects/<PROJECT_NUMBER>/locations/global/collections/default_collection/dataStores/<ID>
   DATA_STORE_ONEDRIVE=projects/<PROJECT_NUMBER>/locations/global/collections/default_collection/dataStores/<ID>
   ```

### 3. Executar a interface ADK Web

```bash
uv run adk web --port 8501
```

Acesse `http://localhost:8501` no navegador e selecione o agente **`app`**.

---

## Testes e Qualidade

Execute os testes automatizados (validam o carregamento do agente nos modos AI Studio e Vertex AI com datastores):

```bash
uv run pytest tests/test_runnability.py
```

Para lint e checagem de tipos:

```bash
make lint
```

---

## Deploy no Vertex AI Agent Engine (Opcional)

O script [`app/deploy.py`](app/deploy.py) empacota `./app` para o **Vertex AI Agent Engine** lendo `GOOGLE_CLOUD_PROJECT`, `DEPLOY_LOCATION`, `STAGING_BUCKET` e as variáveis `DATA_STORE_*` diretamente do ambiente.

---

## Estrutura do Projeto

- `app/agent.py`: Definição completa do grafo multiagente, subagentes especialistas, loop de refinamento e callbacks de citação.
- `app/config.py`: Dataclass `ResearchConfiguration`, detecção automática de modo (AI Studio vs. Vertex AI) e leitura segura das variáveis de ambiente.
- `app/__init__.py`: Bootstrap de variáveis de ambiente (`app/.env`) antes da importação de `root_agent`.
- `app/deploy.py`: Script de deploy para Vertex AI Agent Engine.
- `tests/test_runnability.py`: Testes de importação e configuração dos modos AI Studio e Vertex AI.
- `.env.example`: Modelo documentado de variáveis de ambiente com placeholders.
