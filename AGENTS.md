# Deep Search — Inteligência de Mercado e Apoio a MPEs (Sebrae-SP)

## Intent

Agente multiagente de pesquisa profunda construído sobre **Google ADK + Gemini**, adaptado para gerar **Dossiês de Inteligência de Mercado e Apoio a MPEs (MEI, ME, EPP)** no contexto do Sebrae-SP.

O projeto opera em **modo dual** sem alterar a arquitetura multipesquisa:
- **Modo Público / Portfólio (AI Studio):** usa `GOOGLE_API_KEY` e executa pesquisas via `agente_web` (`google_search`), sem expor ou exigir credenciais/datastores corporativos.
- **Modo Corporativo (Vertex AI + ADC):** quando `GOOGLE_API_KEY` não está definida e `DATA_STORE_*` estão configurados, ativa os subagentes `VertexAiSearchTool` (`agente_datasebrae`, `agente_portfolio`, `agente_sharepoint`, `agente_onedrive`) em conjunto com `agente_web`.

## Security & Public Repository Rules

- **Nunca** inserir IDs de projeto GCP, buckets, credenciais ou `data_store_id` reais no código-fonte, testes ou documentação.
- Todo acesso a datastores e configurações de deploy deve vir de variáveis de ambiente lidas via `app/config.py` e documentadas apenas com placeholders em `.env.example`.
- `.env` e `app/.env` estão protegidos no `.gitignore` e nunca devem ser versionados.

## End-to-end flow

```
user topic
  -> interactive_planner_agent (root LlmAgent)          # Phase 1: Plan & Refine (HITL)
       calls plan_generator (AgentTool) -> 5-goal plan in Portuguese
       refines with user; waits for EXPLICIT approval
       |  (approved)
       v
  research_pipeline (SequentialAgent)                    # Phase 2: Autonomous research
       1. section_planner            -> markdown outline           (state: report_sections)
       2. section_researcher         -> first-pass multi-source    (state: section_research_findings)
       3. iterative_refinement_loop (LoopAgent, max = config.max_search_iterations = 3)
            research_evaluator       -> grade pass/fail + follow_up_queries with target_source (state: research_evaluation)
            escalation_checker       -> stop loop when grade == "pass"
            enhanced_search_executor -> run follow-ups by target_source, merge findings
       4. report_composer_with_citations -> dossiê with <cite source="src-N"/> tags
            citation_replacement_callback rewrites tags -> Markdown links (state: final_report_with_citations)
```

## Key files

1. **`app/agent.py`** — Toda a orquestração multiagente:
   - `interactive_planner_agent` (root `LlmAgent`, `output_key="research_plan"`) e `plan_generator` (`AgentTool`).
   - `_build_datastore_agent()`: instancia subagentes `VertexAiSearchTool` (`agente_datasebrae`, `agente_portfolio`, `agente_sharepoint`, `agente_onedrive`) somente quando `config.use_vertex_ai` é `True` e o respectivo `data_store_id` está preenchido.
   - `agente_web`: subagente `google_search`, sempre presente em `SEBRAE_RESEARCH_TOOLS`.
   - `research_pipeline` (`SequentialAgent`): `section_planner` → `section_researcher` → `iterative_refinement_loop` (`research_evaluator`, `EscalationChecker`, `enhanced_search_executor`) → `report_composer` (`report_composer_with_citations`).
   - `Feedback` / `SearchQuery`: schema Pydantic estruturado do avaliador com `target_source` (`datasebrae`, `portfolio`, `sharepoint`, `onedrive`, `web`).
   - `collect_research_sources_callback` / `citation_replacement_callback`: coleta metadados web (`grounding_metadata`) e internos (`retrieved_contexts`) e resolve citações inline.
2. **`app/config.py`** — Bootstrap de `app/.env`, detecção de auth (AI Studio vs. Vertex AI), `_clean_env()` (ignora placeholders/vazios) e `ResearchConfiguration` (`worker_model`, `critic_model`, `max_search_iterations = 3`, `use_vertex_ai`, `*_data_store_id`).
3. **`app/__init__.py`** — Carrega `app/.env`, resolve defaults Vertex AI quando `GOOGLE_API_KEY` está ausente e importa `root_agent`.
4. **`app/deploy.py`** — Deploy no Vertex AI Agent Engine lendo `GOOGLE_CLOUD_PROJECT`, `DEPLOY_LOCATION`, `STAGING_BUCKET` e `DATA_STORE_*` do ambiente.
5. **`tests/test_runnability.py`** — Testes de importação e validação dos modos AI Studio (somente `agente_web`) e Vertex AI com datastores (5 subagentes).

## Running & Testing

- **Instalar dependências:** `uv sync`
- **Testar via ADK Web:** `uv run adk web --port 8501` (equivalente a `make playground`)
- **Rodar testes:** `uv run pytest tests/test_runnability.py`
- **Lint / Typecheck:** `make lint` (`codespell`, `ruff check`, `ruff format --check`, `mypy`)
- **Observação:** Não há pasta `frontend/` neste repositório; a execução local padrão é via `adk web` ou `adk api_server`.
