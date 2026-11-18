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

import datetime
import logging
import re
from collections.abc import AsyncGenerator
from typing import Literal
from urllib.parse import urlparse

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps.app import App
from google.adk.events import Event, EventActions
from google.adk.planners import BuiltInPlanner
from google.adk.tools import VertexAiSearchTool, google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from .config import config


# --- Structured Output Models ---
class SearchQuery(BaseModel):
    """Modelo representando uma consulta direcionada para uma fonte específica."""

    target_source: Literal[
        "datasebrae", "portfolio", "sharepoint", "onedrive", "web"
    ] = Field(
        description=(
            "Fonte alvo da pesquisa: 'datasebrae' (estatísticas/pesquisas MPEs), "
            "'portfolio' (cursos/consultorias/soluções Sebrae), "
            "'sharepoint' (guias/estudos internos Sebrae-SP), "
            "'onedrive' (documentos complementares) ou "
            "'web' (tendências externas e notícias recentes)."
        )
    )
    search_query: str = Field(
        description="Consulta altamente específica e direcionada em português."
    )


class Feedback(BaseModel):
    """Modelo de avaliação de qualidade da pesquisa do dossiê Sebrae-SP."""

    grade: Literal["pass", "fail"] = Field(
        description=(
            "Resultado da avaliação. 'pass' se a pesquisa cobrir dados externos, "
            "dados internos Sebrae e recomendações reais do portfólio; "
            "'fail' se houver lacunas relevantes."
        )
    )
    comment: str = Field(
        description=(
            "Explicação detalhada em português destacando pontos fortes e lacunas "
            "(ex: falta de recorte territorial SP, ausência de produtos do portfólio ou falta de indicadores)."
        )
    )
    follow_up_queries: list[SearchQuery] | None = Field(
        default=None,
        description=(
            "Lista de consultas específicas (com target_source definido) para corrigir as lacunas. "
            "Deve ser null ou vazia se grade for 'pass'."
        ),
    )


# --- Helpers & Callbacks ---
def _normalize_uri(uri: str) -> str:
    """Converte URIs gs:// para links https://storage.cloud.google.com/ quando aplicável."""
    if not uri:
        return ""
    if uri.startswith("gs://"):
        return "https://storage.cloud.google.com/" + uri[len("gs://") :]
    return uri


def collect_research_sources_callback(
    callback_context: CallbackContext,
) -> None:
    """Coleta fontes web (google_search) e corporativas (VertexAiSearchTool) dos eventos da sessão."""
    session = callback_context._invocation_context.session
    url_to_short_id = callback_context.state.get("url_to_short_id", {})
    sources = callback_context.state.get("sources", {})
    id_counter = len(url_to_short_id) + 1

    for event in session.events:
        if not (
            event.grounding_metadata
            and event.grounding_metadata.grounding_chunks
        ):
            continue

        chunks_info: dict[int, str] = {}
        for idx, chunk in enumerate(event.grounding_metadata.grounding_chunks):
            url = ""
            title = ""
            domain = ""

            # 1. Fontes Web (google_search)
            if getattr(chunk, "web", None) and chunk.web.uri:
                url = chunk.web.uri
                domain = chunk.web.domain or urlparse(url).netloc or "Web"
                title = (
                    chunk.web.title
                    if chunk.web.title and chunk.web.title != chunk.web.domain
                    else domain
                )

            # 2. Fontes Internas Vertex AI Search (DataSebrae, Portfólio, SharePoint, OneDrive)
            elif getattr(chunk, "retrieved_context", None):
                rc = chunk.retrieved_context
                raw_uri = getattr(rc, "uri", "") or ""
                title = getattr(rc, "title", "") or "Documento Interno Sebrae"
                url = _normalize_uri(raw_uri) or f"#sebrae-doc-{id_counter}"
                domain = urlparse(url).netloc or "Sebrae Interno"

            if not url:
                continue

            if url not in url_to_short_id:
                short_id = f"src-{id_counter}"
                url_to_short_id[url] = short_id
                sources[short_id] = {
                    "short_id": short_id,
                    "title": title,
                    "url": url,
                    "domain": domain,
                    "supported_claims": [],
                }
                id_counter += 1
            chunks_info[idx] = url_to_short_id[url]

        if event.grounding_metadata.grounding_supports:
            for support in event.grounding_metadata.grounding_supports:
                confidence_scores = support.confidence_scores or []
                chunk_indices = support.grounding_chunk_indices or []
                for i, chunk_idx in enumerate(chunk_indices):
                    if chunk_idx in chunks_info:
                        short_id = chunks_info[chunk_idx]
                        confidence = (
                            confidence_scores[i]
                            if i < len(confidence_scores)
                            else 0.5
                        )
                        text_segment = (
                            support.segment.text if support.segment else ""
                        )
                        sources[short_id]["supported_claims"].append(
                            {
                                "text_segment": text_segment,
                                "confidence": confidence,
                            }
                        )

    callback_context.state["url_to_short_id"] = url_to_short_id
    callback_context.state["sources"] = sources


def citation_replacement_callback(
    callback_context: CallbackContext,
) -> genai_types.Content:
    """Substitui tags <cite source="src-N"/> por links Markdown no relatório final."""
    final_report = callback_context.state.get("final_cited_report", "")
    sources = callback_context.state.get("sources", {})

    def tag_replacer(match: re.Match) -> str:
        short_id = match.group(1)
        if not (source_info := sources.get(short_id)):
            logging.warning(
                f"Invalid citation tag found and removed: {match.group(0)}"
            )
            return ""
        display_text = source_info.get(
            "title", source_info.get("domain", short_id)
        )
        url = source_info.get("url", "")
        if url.startswith("#"):
            return f" ({display_text})"
        return f" [{display_text}]({url})"

    processed_report = re.sub(
        r'<cite\s+source\s*=\s*["\']?\s*(src-\d+)\s*["\']?\s*/>',
        tag_replacer,
        final_report,
    )
    processed_report = re.sub(r"\s+([.,;:])", r"\1", processed_report)
    callback_context.state["final_report_with_citations"] = processed_report
    return genai_types.Content(parts=[genai_types.Part(text=processed_report)])


# --- Custom Agent for Loop Control ---
class EscalationChecker(BaseAgent):
    """Verifica a avaliação da pesquisa e interrompe o loop caso grade == 'pass'."""

    def __init__(self, name: str):
        super().__init__(name=name)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        evaluation_result = ctx.session.state.get("research_evaluation")
        if evaluation_result and evaluation_result.get("grade") == "pass":
            logging.info(
                f"[{self.name}] Research evaluation passed. Escalating to stop loop."
            )
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logging.info(
                f"[{self.name}] Research evaluation failed or not found. Loop will continue."
            )
            yield Event(author=self.name)


# --- DATASTORES CONFIG (SEBRAE-SP) ---
def _build_datastore_agent(
    *,
    name: str,
    description: str,
    instruction: str,
    data_store_id: str | None,
) -> LlmAgent | None:
    """Create a Vertex AI Search specialist subagent only when Vertex AI and datastore ID are active."""
    if not (config.use_vertex_ai and data_store_id):
        return None
    tool = VertexAiSearchTool(data_store_id=data_store_id)
    return LlmAgent(
        name=name,
        model=config.worker_model,
        description=description,
        instruction=instruction,
        tools=[tool],
        after_agent_callback=collect_research_sources_callback,
    )


# --- SUBAGENTES ESPECIALISTAS POR FONTE (Encapsulados via AgentTool) ---
agente_datasebrae = _build_datastore_agent(
    name="agente_datasebrae",
    description="Consulta estatísticas oficiais, pesquisas setoriais, perfil empresarial e indicadores de MPEs no DataSebrae.",
    instruction="""
    Você é especialista na base DataSebrae.
    Busque dados estatísticos, pesquisas setoriais, taxas de sobrevivência, faturamento, maturidade digital e indicadores de MEI, ME e EPP.
    Retorne um resumo objetivo em português contendo os dados numéricos exatos encontrados, além do título e URI/link de cada documento consultado.
    """,
    data_store_id=config.datasebrae_data_store_id,
)

agente_portfolio = _build_datastore_agent(
    name="agente_portfolio",
    description="Consulta cursos, consultorias, oficinas, programas e produtos oficiais do Portfólio Sebrae.",
    instruction="""
    Você é especialista no Portfólio de Soluções do Sebrae.
    Busque produtos, cursos, consultorias, programas (ex: Sebraetec, ALI, Empretec, trilhas e oficinas) aderentes ao setor, porte (MEI/ME/EPP) e dores informadas.
    Retorne o nome exato da solução, público-alvo, objetivo/benefício prático e a referência do documento consultado.
    """,
    data_store_id=config.portfolio_data_store_id,
)

agente_sharepoint = _build_datastore_agent(
    name="agente_sharepoint",
    description="Consulta documentos internos, estudos territoriais, manuais, guias e PDFs no SharePoint do Sebrae-SP.",
    instruction="""
    Você é especialista na base documental interna do SharePoint do Sebrae-SP.
    Busque estudos regionais/territoriais, diretrizes de atendimento, notas técnicas, cartilhas setoriais e boas práticas internas.
    Retorne os achados relevantes em português preservando títulos de arquivos e referências.
    """,
    data_store_id=config.sharepoint_data_store_id,
)

agente_onedrive = _build_datastore_agent(
    name="agente_onedrive",
    description="Consulta arquivos e materiais de trabalho complementares no OneDrive conectado.",
    instruction="""
    Busque materiais complementares, análises prévias e documentos de apoio no OneDrive quando solicitado.
    Retorne um resumo fiel em português com os títulos dos arquivos encontrados.
    Nunca exponha dados pessoais sensíveis (CPF, telefone, e-mail pessoal) em conformidade com a LGPD.
    """,
    data_store_id=config.onedrive_data_store_id,
)

agente_web = LlmAgent(
    name="agente_web",
    model=config.worker_model,
    description="Realiza buscas na web aberta sobre tendências de mercado, notícias locais/regionais, regulação e cenário econômico.",
    instruction="""
    Você é um pesquisador de mercado externo.
    Use `google_search` para levantar tendências recentes do setor, notícias econômicas do município/região em São Paulo, mudanças regulatórias e comportamento do consumidor.
    Priorize fontes confiáveis (IBGE, Fundação Seade, associações setoriais, imprensa econômica e órgãos públicos).
    Retorne os fatos sintetizados em português.
    """,
    tools=[google_search],
    after_agent_callback=collect_research_sources_callback,
)

_ACTIVE_DATASTORE_AGENTS = [
    agent
    for agent in (
        agente_datasebrae,
        agente_portfolio,
        agente_sharepoint,
        agente_onedrive,
    )
    if agent is not None
]

SEBRAE_RESEARCH_TOOLS = [
    AgentTool(agent) for agent in _ACTIVE_DATASTORE_AGENTS
] + [AgentTool(agente_web)]


# --- AGENT DEFINITIONS ---
plan_generator = LlmAgent(
    model=config.worker_model,
    name="plan_generator",
    description="Gera ou refina o plano de pesquisa orientado à ação para dossiês de Inteligência de Mercado e MPEs do Sebrae-SP.",
    instruction=f"""
    Você é um estrategista de Inteligência de Mercado do Sebrae-SP. Seu trabalho é criar um PLANO DE PESQUISA de alto nível em **português**, e não responder ao tema diretamente.
    Se já houver um PLANO DE PESQUISA no estado da sessão, refine-o com base no feedback do usuário.

    PLANO DE PESQUISA ATUAL (SE HOUVER):
    {{ research_plan? }}

    **CONTEXTO PADRÃO SEBRAE-SP:**
    - Caso o usuário não especifique estado, assuma o estado de **São Paulo (SP)**.
    - Caso não especifique porte, considere **MPEs em geral (MEI, ME e EPP)**.

    **INSTRUÇÃO GERAL: CLASSIFICAÇÃO DE TAREFAS (OBRIGATÓRIO MANTER AS TAGS EXATAS EM INGLÊS)**
    Cada tópico (bullet point) DEVE começar com uma das tags abaixo (o sistema depende dessas tags exatas):
    - **`[RESEARCH]`**: Metas de coleta de informações, investigação de dados externos e consulta às bases internas do Sebrae (DataSebrae, SharePoint, Portfólio).
    - **`[DELIVERABLE]`**: Metas de síntese e criação de artefatos estruturados ao final da pesquisa (ex: tabelas comparativas, matrizes de dores x soluções Sebrae, planos de ação).

    **REGRA INICIAL (5 METAS `[RESEARCH]` COBRINDO OS EIXOS SEBRAE):**
    Sua saída inicial DEVE começar com uma lista de 5 metas `[RESEARCH]` orientadas à ação (iniciando com verbos como "Analisar", "Mapear", "Investigar", "Identificar", "Levantar"), cobrindo:
    1. Cenário atual, tendências de mercado e fatores competitivos do setor (busca externa/web).
    2. Perfil empresarial, indicadores setoriais e estatísticas de MPEs (MEI/ME/EPP) na base DataSebrae.
    3. Recorte territorial/regional em SP, estudos internos e diretrizes aplicáveis no SharePoint do Sebrae-SP.
    4. Principais desafios operacionais, financeiros, comerciais ou digitais enfrentados pelas MPEs desse setor/porte.
    5. Mapeamento de cursos, consultorias e programas aderentes no Portfólio Sebrae para mitigar as dores identificadas.

    - **Entregáveis Implícitos Proativos (`[DELIVERABLE][IMPLIED]`):** Logo após as 5 metas `[RESEARCH]`, adicione pelo menos 1 ou 2 entregáveis práticos prefixados com `[DELIVERABLE][IMPLIED]` — por exemplo:
      - `[DELIVERABLE][IMPLIED] Construir uma tabela Markdown relacionando as principais dores do setor às soluções recomendadas do Portfólio Sebrae.`

    **REGRA DE REFINAMENTO:**
    - Ao incorporar ajustes do usuário em itens existentes, adicione `[MODIFIED]` (ex: `[RESEARCH][MODIFIED]`).
    - Ao adicionar novos itens pedidos pelo usuário, use `[RESEARCH][NEW]` ou `[DELIVERABLE][NEW]`.
    - Se um ajuste implicar um novo quadro/síntese útil ao consultor, adicione com `[DELIVERABLE][IMPLIED]`.
    - Mantenha a ordem original dos itens existentes.

    **USO RESTRITO DE FERRAMENTA:**
    Crie o plano *sem pesquisar*. Use `google_search` APENAS se o usuário mencionar uma sigla, CNAE ou programa desconhecido que precise ser desambiguado para redigir o plano.
    Data atual: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    tools=[google_search],
)


section_planner = LlmAgent(
    model=config.worker_model,
    name="section_planner",
    description="Converte o plano aprovado em uma estrutura Markdown de seções para o Dossiê Sebrae-SP.",
    instruction="""
    Você é um arquiteto de relatórios executivos do Sebrae-SP. Usando o tema e o plano presente em `research_plan`, estruture o esboço em Markdown do dossiê final em **português**.
    Nota: Ignore as tags de controle (`[MODIFIED]`, `[NEW]`, `[IMPLIED]`, `[RESEARCH]`, `[DELIVERABLE]`) ao escrever os títulos.

    Crie uma estrutura clara contendo de 5 a 6 seções sem sobreposição, seguindo preferencialmente esta lógica consultiva:
    # 1. Resumo Executivo
    Síntese dos principais achados e direcionamento rápido para o consultor Sebrae-SP.
    # 2. Panorama Setorial e Tendências de Mercado
    Cenário atual, comportamento da demanda, inovações e movimentações do setor.
    # 3. Perfil das MPEs e Recorte Territorial (SP)
    Dados do DataSebrae, particularidades regionais/municipais em São Paulo e perfil por porte (MEI, ME e EPP).
    # 4. Diagnóstico de Dores, Riscos e Oportunidades
    Gargalos críticos de gestão e alavancas de crescimento para os pequenos negócios analisados.
    # 5. Recomendações de Atendimento e Portfólio Sebrae
    Plano prático de intervenção e tabela relacionando dores identificadas a soluções reais do Portfólio Sebrae.

    Adapte subtítulos conforme as especificidades do `research_plan`.
    NÃO inclua seção de "Referências" ou "Fontes" ao final; todas as citações serão inseridas inline no texto.
    """,
    output_key="report_sections",
)


section_researcher = LlmAgent(
    model=config.worker_model,
    name="section_researcher",
    description="Executa a primeira rodada completa de pesquisa combinando bases internas do Sebrae-SP e busca web.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    Você é um pesquisador sênior de Inteligência de Mercado do Sebrae-SP. Sua missão é executar o plano em `research_plan` com fidelidade absoluta, escrevendo tudo em **português**.

    Você receberá uma lista de metas prefixadas com `[RESEARCH]` ou `[DELIVERABLE]`. Siga rigorosamente estas duas fases sequenciais:

    ---

    **Fase 1: Coleta de Informações (Metas `[RESEARCH]`)**

    *   Processe **TODAS** as metas `[RESEARCH]` antes de iniciar a Fase 2.
    *   **Roteamento Inteligente de Ferramentas (CRÍTICO — não use apenas busca web!):**
        *   Para indicadores de MPEs, pesquisas setoriais, taxas de sobrevivência e perfil MEI/ME/EPP -> acione **`agente_datasebrae`**.
        *   Para cursos, consultorias, trilhas, programas (Sebraetec, ALI, Empretec etc.) e ofertas Sebrae -> acione **`agente_portfolio`**.
        *   Para estudos regionais, manuais internos, diretrizes e documentos territoriais de SP -> acione **`agente_sharepoint`** (e `agente_onedrive` se pertinente).
        *   Para notícias recentes, tendências de consumo, dados macroeconômicos externos (IBGE/Seade/associações) e mercado local -> acione **`agente_web`**.
    *   Para cada meta `[RESEARCH]`:
        *   Formule de 2 a 4 consultas específicas e chame as ferramentas mais adequadas à natureza daquela meta.
        *   Sintetize os achados em um resumo rico em dados, preservando números, nomes exatos de soluções do Sebrae e referências às fontes retornadas.
        *   Respeite a LGPD: utilize apenas dados agregados/institucionais, nunca expondo dados pessoais de clientes.

    ---

    **Fase 2: Síntese e Criação de Entregáveis (Metas `[DELIVERABLE]`)**

    *   Inicie somente após concluir todas as metas `[RESEARCH]`.
    *   Para cada meta `[DELIVERABLE]`:
        *   Produza exatamente o artefato solicitado (ex: tabela Markdown comparativa de Dores x Soluções do Portfólio Sebrae, matriz de oportunidades, checklist para consultores).
        *   Baseie-se exclusivamente nos dados coletados na Fase 1, sem inventar nomes de programas ou estatísticas.

    ---

    **Saída Final:** Apresente de forma organizada todos os resumos detalhados das metas `[RESEARCH]` seguidos de todos os artefatos gerados nas metas `[DELIVERABLE]`.
    """,
    tools=SEBRAE_RESEARCH_TOOLS,
    output_key="section_research_findings",
    after_agent_callback=collect_research_sources_callback,
)


_DATASTORE_EVAL_NOTE = (
    ""
    if _ACTIVE_DATASTORE_AGENTS
    else "\n    **MODO SOMENTE WEB ATIVO:** Nenhum datastore interno está habilitado neste ambiente; avalie a qualidade usando exclusivamente fontes web (`target_source: \"web\"`) e o recorte territorial/MPEs.\n"
)

research_evaluator = LlmAgent(
    model=config.critic_model,
    name="research_evaluator",
    description="Avalia criticamente a pesquisa sob a ótica do Sebrae-SP e gera consultas complementares direcionadas.",
    instruction=f"""
    Você é um analista sênior de Qualidade e Inteligência Estratégica do Sebrae-SP avaliando os achados em `section_research_findings`.
{_DATASTORE_EVAL_NOTE}
    **CRITÉRIOS DE AVALIAÇÃO SEBRAE-SP:**
    1. **Cruzamento Externo + Interno:** A pesquisa trouxe tanto tendências externas de mercado quanto dados internos (`DataSebrae` / `SharePoint`)? Se tiver apenas informações genéricas da internet, reprove (`fail`).
    2. **Recorte Territorial e Porte:** Há contextualização para o município/região de São Paulo solicitado e recorte para pequenos negócios (MEI, ME, EPP)?
    3. **Aderência ao Portfólio Sebrae:** As recomendações citam soluções reais verificadas no `Portfólio Sebrae` (cursos, consultorias, programas específicos) em vez de conselhos genéricos?
    4. **Profundidade e Rastreabilidade:** Os dados quantitativos e afirmações possuem lastro claro?

    **REGRAS DE DECISÃO:**
    - Assuma que o tema pedido pelo usuário é válido; avalie apenas a completude e profundidade da pesquisa.
    - Se encontrar lacunas importantes nos critérios acima, atribua `grade: "fail"`, explique claramente o que faltou em `comment` (em português) e gere de 3 a 5 `follow_up_queries` altamente específicas, preenchendo corretamente o campo `target_source` (`"datasebrae"`, `"portfolio"`, `"sharepoint"`, `"onedrive"` ou `"web"`) para direcionar a correção à base certa.
    - Se a pesquisa já estiver sólida, contextualizada para SP e com soluções reais do Sebrae, atribua `grade: "pass"`.

    Data atual: {datetime.datetime.now().strftime("%Y-%m-%d")}
    Sua resposta DEVE ser exclusivamente um objeto JSON válido seguindo o schema `Feedback`.
    """,
    output_schema=Feedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    output_key="research_evaluation",
)


enhanced_search_executor = LlmAgent(
    model=config.worker_model,
    name="enhanced_search_executor",
    description="Executa buscas complementares nas bases indicadas pelo avaliador e consolida os novos achados.",
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(include_thoughts=True)
    ),
    instruction="""
    Você é um pesquisador especialista executando uma rodada de refinamento porque a avaliação anterior recebeu `fail`.

    1. Analise `research_evaluation` para entender as lacunas apontadas em `comment` e a lista `follow_up_queries`.
    2. Para CADA item em `follow_up_queries`, acione a ferramenta correspondente ao `target_source`:
       - `target_source == "datasebrae"` -> chame `agente_datasebrae`
       - `target_source == "portfolio"` -> chame `agente_portfolio`
       - `target_source == "sharepoint"` -> chame `agente_sharepoint`
       - `target_source == "onedrive"` -> chame `agente_onedrive`
       - `target_source == "web"` -> chame `agente_web`
    3. Sintetize as novas evidências em **português** e INTEGRE-AS ao conteúdo já existente em `section_research_findings`, atualizando também tabelas/entregáveis se novas soluções do Portfólio ou indicadores tiverem sido encontrados.
    4. Sua saída DEVE ser a versão completa, consolidada e aprimorada de `section_research_findings` (nunca descarte os achados válidos da rodada anterior).
    """,
    tools=SEBRAE_RESEARCH_TOOLS,
    output_key="section_research_findings",
    after_agent_callback=collect_research_sources_callback,
)


report_composer = LlmAgent(
    model=config.critic_model,
    name="report_composer_with_citations",
    include_contents="none",
    description="Redige o Dossiê de Inteligência de Mercado e Apoio a MPEs final em português com citações inline.",
    instruction="""
    Transforme os insumos coletados em um **Dossiê de Inteligência de Mercado e Apoio a MPEs do Sebrae-SP** altamente profissional, consultivo, acionável e rigorosamente citado, inteiramente em **português**.

    ---
    ### DADOS DE ENTRADA
    *   Plano de Pesquisa: `{research_plan}`
    *   Achados da Pesquisa: `{section_research_findings}`
    *   Fontes Catalogadas: `{sources}`
    *   Estrutura do Dossiê: `{report_sections}`

    ---
    ### DIRETRIZES EDITORIAIS SEBRAE-SP
    1. **Público-alvo:** Consultores, agentes de atendimento e analistas de negócios do Sebrae-SP que usarão este dossiê para orientar micro e pequenas empresas (MEI, ME, EPP).
    2. **Clareza e Ação:** Destaque indicadores-chave, oportunidades práticas no território e inclua tabelas Markdown claras (especialmente o cruzamento entre **Dor/Necessidade da MPE** e **Solução Recomendada do Portfólio Sebrae**).
    3. **Conformidade e LGPD:** Apresente apenas dados agregados, setoriais e institucionais. Jamais exponha nomes, CPFs, e-mails ou telefones de clientes individuais.
    4. **Honestidade Epistêmica:** Se algum dado específico de um município pequeno não estiver disponível nas bases e você usar o recorte regional ou estadual de SP como referência, explicite isso claramente para o consultor.

    ---
    ### SISTEMA OBRIGATÓRIO DE CITAÇÕES
    Para citar uma fonte presente em `{sources}`, insira a tag especial imediatamente após a afirmação suportada:

    **Formato único permitido:** `<cite source="src-ID_NUMBER" />` (exemplo: `<cite source="src-1" />`)

    ---
    ### Instruções Finais
    - Siga estritamente a organização de seções definida em **Estrutura do Dossiê** (`{report_sections}`).
    - Utilize APENAS tags `<cite source="src-ID_NUMBER" />` correspondentes a IDs existentes em `{sources}`.
    - NÃO crie uma seção final de "Referências" ou "Fontes"; todas as citações são resolvidas inline automaticamente.
    """,
    output_key="final_cited_report",
    after_agent_callback=citation_replacement_callback,
)


research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Executa o plano aprovado: estrutura o dossiê, pesquisa em bases internas Sebrae e web, valida qualidade em loop e redige o relatório citado.",
    sub_agents=[
        section_planner,
        section_researcher,
        LoopAgent(
            name="iterative_refinement_loop",
            max_iterations=config.max_search_iterations,
            sub_agents=[
                research_evaluator,
                EscalationChecker(name="escalation_checker"),
                enhanced_search_executor,
            ],
        ),
        report_composer,
    ],
)


interactive_planner_agent = LlmAgent(
    name="interactive_planner_agent",
    model=config.worker_model,
    description="Assistente principal de planejamento de dossiês de Inteligência de Mercado do Sebrae-SP.",
    instruction=f"""
    Você é o Assistente de Planejamento de Inteligência de Mercado e Apoio a MPEs do **Sebrae-SP**.
    Comunique-se sempre em **português**, de forma cordial, objetiva e consultiva.

    **REGRA DE TRIAGEM E PLANEJAMENTO:**
    - Nunca responda à pesquisa diretamente por conta própria; seu papel é planejar, refinar com o consultor e delegar.
    - Se o pedido inicial do usuário for extremamente vago (por exemplo, sem indicar nem o **setor/segmento** nem a **região/município**), faça uma única pergunta rápida pedindo o **Setor/CNAE**, o **Município/Região de SP** e o **Porte foco (MEI, ME, EPP ou geral)**.
    - Assim que o usuário informar pelo menos o setor ou tema (mesmo que omita porte ou cidade, caso em que o plano assumirá MPEs no estado de SP), chame IMEDIATAMENTE a ferramenta `plan_generator` para gerar a proposta de plano e apresente-a ao usuário.

    **SEU FLUXO DE TRABALHO:**
    1. **Planejar:** Acione `plan_generator` para estruturar o plano e apresente-o claramente ao usuário, perguntando se deseja ajustar algum recorte (região, porte, foco em alguma dor específica) ou se aprova o início da pesquisa.
    2. **Refinar:** Caso o usuário peça alterações, chame `plan_generator` novamente para atualizar o plano até que esteja satisfatório.
    3. **Executar:** Somente após aprovação EXPLÍCITA do usuário (ex: "pode rodar", "aprovado", "está ótimo", "seguir"), delegue imediatamente para o subagente `research_pipeline`.

    Data atual: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    sub_agents=[research_pipeline],
    tools=[AgentTool(plan_generator)],
    output_key="research_plan",
)

root_agent = interactive_planner_agent
app = App(root_agent=root_agent, name="app")
