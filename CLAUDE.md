# Synapse Colosseum — Especificação Completa

## Arquitetura

PWA mobile-first. Frontend: monolito `index.html` (HTML + CSS + JS inline). Backend: Python (FastAPI presumido) servindo `/api/*`. Gateway LLM: OpenRouter (`https://openrouter.ai/api/v1`). Comunicação em tempo real via SSE (Server-Sent Events).

### Stack atual

- Frontend: HTML/CSS/JS vanilla, sem framework, sem bundler
- Backend: Python, endpoints REST + SSE streaming
- LLM Gateway: OpenRouter (API compatível com OpenAI)
- Persistência: sessões em memória/disco (JSON), localStorage no client

### Endpoints existentes

```
POST /api/debate/start          → cria sessão
POST /api/debate/{id}/next      → dispara próxima rodada
GET  /api/debate/{id}/stream    → SSE com eventos da rodada
POST /api/debate/{id}/stop      → para rodada
POST /api/debate/{id}/finalize  → gera relatório final
POST /api/debate/{id}/agenda    → gera agenda pré-debate
POST /api/debate/{id}/research  → pesquisa pré-sessão (Sonar)
POST /api/debate/{id}/generate-personas → personas via LLM
POST /api/debate/{id}/peer-review      → avaliação cruzada
POST /api/debate/{id}/live-summary     → resumo parcial
POST /api/debate/{id}/rerun/{model}    → re-gerar resposta
POST /api/debate/{id}/abort/{model}    → cancelar modelo
POST /api/debate/{id}/upload           → anexar documento
GET  /api/debate/{id}/export           → markdown
GET  /api/debate/{id}/export/json      → JSON
GET  /api/debate/{id}/export/csv       → CSV
POST /api/debate/{id}/podcast          → gera podcast
GET  /api/debate/{id}/summary          → resumo evolutivo
POST /api/debate/fork                  → fork de ponto
GET  /api/models                       → lista modelos configurados
GET  /api/sessions                     → sessões salvas
POST /api/sessions/{id}/resume         → retomar sessão
POST /api/sessions/{id}/clone          → clonar sessão
DELETE /api/sessions/{id}              → excluir
GET  /api/sessions/compare?ids=        → comparar sessões
POST /api/analyze-prompt               → análise do prompt
GET  /api/metrics                      → métricas por provedor
GET  /api/logs                         → log de erros
POST /api/logs/clear                   → limpar logs
GET  /api/personas                     → catálogo de personas
```

### Eventos SSE (stream)

```
round_start          → { round, order[], mode, max_tokens }
token                → { round, model_id, model_name, content }
message_complete     → { round, model_id, model_name, content, cost, truncated, latency_ms, usage, cached, injection_warning[], persona, is_error }
round_end            → { round, round_cost, compression_cost, total_cost }
compression_start    → { round }
compression_done     → { round, summary_length }
compression_failed   → { round }
scoring_start        → { round }
round_scores         → { round, scores: {name: {total, justificativa}}, avg_scores }
round_xray           → { round, consenso, divergencia, modelo_destaque, model_scores }
round_narrative      → { round, narrative, cost }
survival_scoring_start → { round }
survival_result      → { round, scores, eliminated[], eliminated_ids[], survivors[], scoring_cost }
survival_winner      → { winner }
budget_warning       → { percent, total_cost, max_budget, message? }
round_complete       → fim da rodada (client chama /next manualmente)
done                 → fim do debate
error                → { message }
```

### Modos de debate

- `debate` — competitivo padrão
- `brainstorm` — cooperativo
- `devil_advocate` — posições opostas forçadas
- `survival` — eliminatório 50% por rodada (juiz pontua, metade eliminada)
- `delphi` — previsão/convergência
- `cross_exam` — interrogatório cruzado
- `investment_committee` — alocação de recursos
- `compression_ladder` — compressão iterativa

-----

## OpenRouter API — Referência Técnica

### Autenticação e headers

```
Authorization: Bearer {OPENROUTER_API_KEY}
Content-Type: application/json
HTTP-Referer: {site_url}    (opcional, recomendado)
X-Title: Synapse Colosseum  (opcional)
```

### max_tokens vs max_completion_tokens

OpenRouter aceita ambos e converte automaticamente. `max_tokens` é deprecated na spec mas funciona. Diferença da API direta da OpenAI: lá, enviar `max_tokens` para modelos o-series retorna erro; via OpenRouter, não. **Decisão: usar `max_tokens` para compatibilidade universal.**

### Compatibilidade de parâmetros por família

|Família               |temperature|reasoning|response_format|Notas                       |
|----------------------|-----------|---------|---------------|----------------------------|
|OpenAI o3/o4/GPT-5+   |❌ ERRO     |✅        |✅              |NÃO enviar temperature/top_p|
|Claude Opus/Sonnet 4.6|✅          |✅        |✅              |Suporte completo            |
|Gemini 2.5/3.1        |✅          |✅        |✅              |Suporte completo            |
|DeepSeek R1/V3.2      |✅          |✅        |❌              |Sem response_format         |
|Llama 4               |✅          |❌        |✅              |Sem reasoning               |
|Grok 3                |✅          |❌        |✅              |Sem reasoning               |
|Mistral/Qwen          |✅          |Varia    |✅              |Checar supported_parameters |

### Request templates por família

```javascript
// OpenAI Reasoning (o3-mini, o4-mini, GPT-5.4, GPT-5.4 Mini/Pro)
// ❌ NÃO incluir temperature, top_p, presence_penalty, frequency_penalty
{ model: "openai/o4-mini", max_tokens: 100000, reasoning: { effort: "medium" } }

// Claude, Gemini, DeepSeek (com reasoning)
{ model: "anthropic/claude-sonnet-4.6", max_tokens: 8192, temperature: 0.7, reasoning: { effort: "high" } }

// Llama, Mistral, Qwen (sem reasoning)
{ model: "meta-llama/llama-4-maverick", max_tokens: 4096, temperature: 0.7, top_p: 0.95 }
```

### Reasoning — comportamento detalhado

- Parâmetro `reasoning` ignorado silenciosamente em modelos sem suporte (GPT-4o, Llama 4)
- Claude: aceita `reasoning.max_tokens` (preferencial) e `reasoning.effort`
  - Effort ratios: xhigh=0.95, high=0.80, medium=0.50, low=0.20, minimal=0.10
  - **Claude 4.6 ignora `reasoning.effort`** → usar `reasoning.max_tokens`
- Gemini: mapeia `reasoning.effort` → `thinkingLevel` (minimal/low/medium/high; xhigh→high)
- Reasoning tokens = output tokens para cobrança
- Response: `usage.completion_tokens_details.reasoning_tokens`

### CORS e Rate Limits

- CORS: `Access-Control-Allow-Origin: *`
- 20 RPM (todos os tiers), 50 req/dia (free), 1000 req/dia (paid)
- DDoS via Cloudflare (limite por IP além do por API key)
- Timeout: ~300s observado

### Streaming

- Todos os modelos suportam `stream: true` (SSE OpenAI-compatible)
- Formato: `data: {json}\n\n`
- Keep-alive durante reasoning: `: OPENROUTER PROCESSING`
- OpenAI o3/o4: reasoning tokens NÃO visíveis (apenas summaries)
- Claude: thinking tokens visíveis via `thinking_delta`

### finish_reason

5 valores normalizados: `stop`, `length`, `tool_calls`, `content_filter`, `error`
Original do provider em `native_finish_reason`.

### /api/v1/models

- Sem autenticação, 349 modelos (~500KB), sem paginação
- Identificar reasoning: checar `"reasoning"` em `supported_parameters` ou `pricing.internal_reasoning`

### Segurança

- Client-side: não proibido nos TOS mas desaconselhado
- Spending limits por key (credit limit, guardrails diário/semanal/mensal)
- Keys com escopo: model allowlist, provider allowlist, budget limits
- OAuth PKCE recomendado para produção user-facing

-----

## Tabela de Preços (Março 2026)

Preços em $/1M tokens. Latência em segundos.

|ID OpenRouter                       |Nome                 |Provider  |Input|Output|Reasoning?|Context|MaxCompletion|Latência|
|------------------------------------|---------------------|----------|-----|------|----------|-------|-------------|--------|
|qwen/qwen3.5-flash-02-23            |Qwen 3.5 Flash       |Qwen      |0.07 |0.26  |Sim       |1M     |65536        |~0      |
|meta-llama/llama-4-scout            |Llama 4 Scout        |Meta      |0.08 |0.30  |Não       |327K   |16384        |~0      |
|deepseek/deepseek-v3.2              |DeepSeek V3.2        |DeepSeek  |0.26 |0.38  |Sim       |164K   |—            |1.5     |
|google/gemini-2.5-flash-lite        |Gemini 2.5 Flash Lite|Google    |0.10 |0.40  |Sim       |1M     |65535        |~0      |
|x-ai/grok-4-fast                    |Grok 4 Fast          |xAI       |0.20 |0.50  |Sim       |2M     |30000        |0.4     |
|x-ai/grok-3-mini                    |Grok 3 Mini          |xAI       |0.30 |0.50  |Sim       |131K   |—            |~0      |
|meta-llama/llama-4-maverick         |Llama 4 Maverick     |Meta      |0.15 |0.60  |Não       |1M     |16384        |~1      |
|openai/gpt-4o-mini                  |GPT-4o Mini          |OpenAI    |0.15 |0.60  |Não       |128K   |16384        |~0      |
|mistralai/mistral-small-2603        |Mistral Small 4      |Mistral   |0.15 |0.60  |Sim       |262K   |—            |~0      |
|perplexity/sonar                    |Sonar                |Perplexity|1.00 |1.00  |Não       |127K   |—            |~1      |
|minimax/minimax-m2.5                |MiniMax M2.5         |MiniMax   |0.20 |1.17  |Sim       |197K   |65536        |~0      |
|minimax/minimax-m2.7                |MiniMax M2.7         |MiniMax   |0.30 |1.20  |Sim       |205K   |131072       |~1      |
|openai/gpt-5.4-nano                 |GPT-5.4 Nano         |OpenAI    |0.20 |1.25  |Sim       |400K   |128000       |~0      |
|google/gemini-3.1-flash-lite-preview|Gemini 3.1 Flash Lite|Google    |0.25 |1.50  |Sim       |1M     |65536        |~0      |
|qwen/qwen3.5-plus-02-15             |Qwen 3.5 Plus        |Qwen      |0.26 |1.56  |Sim       |1M     |65536        |~1      |
|mistralai/devstral-2512             |Devstral 2           |Mistral   |0.40 |2.00  |Não       |262K   |—            |~1      |
|qwen/qwen3.5-122b-a10b              |Qwen 3.5 122B        |Qwen      |0.26 |2.08  |Sim       |262K   |65536        |1.0     |
|deepseek/deepseek-r1                |DeepSeek R1          |DeepSeek  |0.70 |2.50  |Sim       |64K    |16000        |1.5     |
|google/gemini-2.5-flash             |Gemini 2.5 Flash     |Google    |0.30 |2.50  |Sim       |1M     |65535        |0.7     |
|openai/o4-mini-high                 |o4-mini-high         |OpenAI    |1.10 |4.40  |Sim       |200K   |100000       |~6      |
|openai/o4-mini                      |o4-mini              |OpenAI    |1.10 |4.40  |Sim       |200K   |100000       |6.0     |
|openai/o3-mini-high                 |o3-mini-high         |OpenAI    |1.10 |4.40  |Sim       |200K   |100000       |~6      |
|openai/o3-mini                      |o3-mini              |OpenAI    |1.10 |4.40  |Sim       |200K   |100000       |5.8     |
|openai/gpt-5.4-mini                 |GPT-5.4 Mini         |OpenAI    |0.75 |4.50  |Sim       |400K   |128000       |~1      |
|anthropic/claude-haiku-4.5          |Claude Haiku 4.5     |Anthropic |1.00 |5.00  |Sim       |200K   |64000        |0.7     |
|x-ai/grok-4.20-beta                 |Grok 4.20 Beta       |xAI       |2.00 |6.00  |Sim       |2M     |—            |~1      |
|openai/o3                           |o3                   |OpenAI    |2.00 |8.00  |Sim       |200K   |100000       |~5      |
|perplexity/sonar-reasoning-pro      |Sonar Reasoning Pro  |Perplexity|2.00 |8.00  |Sim       |128K   |—            |~2      |
|google/gemini-2.5-pro               |Gemini 2.5 Pro       |Google    |1.25 |10.00 |Sim       |1M     |65536        |~2      |
|openai/gpt-4o                       |GPT-4o               |OpenAI    |2.50 |10.00 |Não       |128K   |16384        |0.5     |
|google/gemini-3.1-pro-preview       |Gemini 3.1 Pro       |Google    |2.00 |12.00 |Sim       |1M     |65536        |3.5     |
|anthropic/claude-sonnet-4.6         |Claude Sonnet 4.6    |Anthropic |3.00 |15.00 |Sim       |1M     |128000       |1.1     |
|openai/gpt-5.4                      |GPT-5.4              |OpenAI    |2.50 |15.00 |Sim       |1.05M  |128000       |~2      |
|x-ai/grok-3                         |Grok 3               |xAI       |3.00 |15.00 |Não       |131K   |131072       |0.7     |
|perplexity/sonar-pro                |Sonar Pro            |Perplexity|3.00 |15.00 |Não       |200K   |8000         |~1      |
|anthropic/claude-opus-4.6           |Claude Opus 4.6      |Anthropic |5.00 |25.00 |Sim       |1M     |128000       |2.3     |
|openai/gpt-5.4-pro                  |GPT-5.4 Pro          |OpenAI    |30.00|180.00|Sim       |1.05M  |128000       |~5      |

### Alertas de nomenclatura

- Não existe “o4” full — apenas o4-mini e o4-mini-high
- Não existe “GPT-5” simples — OpenAI usa GPT-5.4, GPT-5.4 Mini, etc.
- Reasoning tokens inclusos no preço de output, nunca cobrados separadamente

### Modelo de compressão recomendado

Gemini 2.0 Flash Lite: $0.00088 por chamada (10K input + 600 output). Projeção: 100 partidas/mês = $0.62, 1000/mês = $6.16.

-----

## Sistema de Roteamento Multi-Modelo (Planner)

### Princípio

O app não escolhe todos os modelos. Escolhe o menor conjunto que maximize contraste útil, profundidade adequada e custo sob controle. Unidade de planejamento = Rodada (fase com objetivo: explorar, sintetizar, criticar, arbitrar). Sem orçamento → cap de 3 modelos, 3-4 rodadas, evitar premium desnecessário.

### Sinais extraídos do prompt (schema do planner)

```
task_type:          quick_answer | brainstorm | debate | decision_memo | longform_synthesis | deep_research | code_generation | code_review | troubleshooting | multimodal_analysis | creative_writing | adversarial_review
topic_domain:       texto livre (área temática)
depth_required:     shallow | standard | deep | exhaustive
factuality_risk:    low | medium | high | critical
novelty_need:       low | medium | high
adversarial_need:   low | medium | high
deliverable_form:   answer | bullets | memo | matrix | report | code | plan | comparison
output_length:      short | medium | long | very_long
need_for_sources:   none | helpful | required
coding_intensity:   none | light | medium | high
multimodal_input:   none | image | audio | video | mixed
latency_tolerance:  low | medium | high
budget_provided:    boolean
budget_total_usd:   number | null
risk_of_error:      low | medium | high | critical
```

### Regras de roteamento por cenário

|scenario_id        |Quando usar                          |Variedade|Validação|Especialista|Novidade|mix_bias                |Nota                              |
|-------------------|-------------------------------------|---------|---------|------------|--------|------------------------|----------------------------------|
|quick_answer       |Pergunta objetiva, resposta curta    |low      |low      |low         |low     |specific_generalist     |1-2 modelos bastam                |
|brainstorm         |Ideias, exploração, naming           |high     |medium   |low         |high    |diverse_panel           |Variedade > consistência          |
|debate             |Prós x contras, teses opostas        |high     |high     |medium      |medium  |diverse_panel_plus_judge|Contraste melhora resultado       |
|decision_memo      |Recomendação acionável com trade-offs|medium   |high     |medium      |medium  |hybrid_anchor           |Âncora forte + auditor            |
|longform_synthesis |Síntese longa, explicação robusta    |medium   |medium   |high        |low     |hybrid_anchor           |Pouca variedade, mais consistência|
|deep_research      |Factual/volátil, busca, fontes       |medium   |critical |high        |low     |research_then_reason    |Pesquisa + síntese + auditoria    |
|code_generation    |Gerar código, arquitetura, testes    |low      |medium   |medium      |low     |specialist_then_judge   |Especialista primeiro             |
|code_review        |Revisão de PR, bug hunt, segurança   |medium   |high     |medium      |medium  |specialist_then_judge   |Especialista + crítico            |
|troubleshooting    |Diagnóstico técnico, causa-raiz      |medium   |high     |medium      |low     |reasoner_anchor         |Raciocínio > variedade            |
|multimodal_analysis|Imagem/áudio/vídeo central           |medium   |medium   |medium      |low     |multimodal_then_reason  |Multimodal lidera                 |
|creative_writing   |Texto criativo, storytelling         |medium   |low      |high        |high    |diverse_panel           |Variedade > precisão factual      |
|adversarial_review |Red team, crítica forte              |high     |high     |medium      |low     |challenger_heavy        |Oposição é o objetivo             |

### Pools de modelos por papel e orçamento

**FREE:**

- general: Qwen3 Next 80B A3B Instruct (free) | backups: Arcee Trinity Large, Nous Hermes 3 405B
- reasoning: Qwen3 Next 80B A3B Instruct (free) | backup: Arcee Trinity Large
- coding: Qwen3 Coder 480B A35B (free) | backup: Nous Hermes 3 405B
- multimodal: NVIDIA Nemotron Nano 12B 2 VL (free) | backup: Qwen3 Next 80B

**LEAN:**

- general: Ministral 3 14B 2512 | backups: Grok 4 Fast, Solar Pro 3
- reasoning: ERNIE 4.5 21B A3B Thinking | backups: Grok 4.1 Fast Reasoning, Qwen3 Next 80B Thinking
- coding: Qwen3 Coder 30B A3B Instruct | backups: KAT-Coder-Pro V1, Devstral Small 1.1
- multimodal: Qwen3 VL 8B Instruct | backup: Voxtral Small 24B
- research: Tongyi DeepResearch 30B A3B | backup: Relace Search

**BALANCED:**

- general: Moonshot V1 128K | backups: Mistral Large 3 2512, Amazon Nova Pro 1.0
- reasoning: DeepSeek R1 0528 | backups: Qwen3 235B A22B Thinking 2507, Qwen3 Next 80B Thinking
- coding: KAT-Coder-Pro V1 | backups: Qwen3 Coder Next, GPT-5.1-Codex
- multimodal: Qwen3 VL 30B A3B Thinking | backups: MiMo-V2-Omni, Qwen3 VL 235B
- research: Sonar | backups: Tongyi DeepResearch, Relace Search

**EXPANDED:**

- general: MiMo-V2-Pro | backups: Qwen3 Max, Amazon Nova Pro 1.0
- reasoning: Qwen3 235B A22B Thinking 2507 | backups: DeepSeek R1 0528, o3
- coding: GPT-5.1-Codex | backups: KAT-Coder-Pro V1, GPT-5.1-Codex-Max
- multimodal: MiMo-V2-Omni | backups: Qwen3 VL 235B, Qwen3 VL 30B
- research: Sonar Pro | backups: Sonar Deep Research, Tongyi DeepResearch

**PREMIUM:**

- general: Claude Sonnet 4.6 | backups: Gemini 3.1 Pro, GPT-5.4
- reasoning: o3 Pro | backups: Claude Opus 4.6, o1-pro
- coding: GPT-5.1-Codex-Max | backups: GPT-5.1-Codex, Claude Sonnet 4.6
- multimodal: MiMo-V2-Omni | backups: Qwen3 VL 235B, GLM 4.5V
- research: Sonar Deep Research | backups: Sonar Reasoning Pro, Sonar Pro

### Execution Templates — rodadas por cenário × budget

Formato: rodadas/modelos. Cada rodada tem: objetivo, nº modelos, tokens input/output.

|Cenário            |default_no_budget|lean |balanced|expanded|premium|
|-------------------|-----------------|-----|--------|--------|-------|
|quick_answer       |2r/2m            |1r/1m|2r/2m   |2r/3m   |2r/3m  |
|brainstorm         |3r/3m            |2r/2m|3r/4m   |3r/5m   |3r/5m  |
|debate             |3r/3m            |2r/2m|3r/3m   |3r/4m   |4r/4m  |
|decision_memo      |3r/3m            |2r/2m|3r/3m   |3r/4m   |4r/4m  |
|longform_synthesis |3r/2m            |2r/2m|3r/3m   |3r/3m   |4r/4m  |
|deep_research      |3r/3m            |3r/3m|4r/4m   |4r/5m   |4r/5m  |
|code_generation    |3r/2m            |2r/2m|3r/3m   |3r/3m   |4r/4m  |
|code_review        |3r/3m            |2r/2m|3r/3m   |3r/4m   |4r/4m  |
|troubleshooting    |3r/2m            |2r/2m|3r/3m   |3r/4m   |4r/4m  |
|multimodal_analysis|3r/2m            |2r/2m|3r/3m   |3r/4m   |4r/4m  |
|creative_writing   |3r/3m            |2r/2m|3r/4m   |3r/4m   |3r/5m  |
|adversarial_review |3r/3m            |2r/2m|3r/3m   |4r/4m   |4r/4m  |

### Padrão de rodadas típico (exemplo: debate/balanced)

```
R1: construir lados opostos → 2 modelos × 1800 input / 650 output
R2: testar colisão          → 1 modelo  × 1400 input / 500 output
R3: julgar e integrar       → 1 modelo  × 1000 input / 350 output
Papéis: reasoner + challenger + judge
Pools: reasoning/balanced + general/balanced
```

### Budget Shifts

|Shift        |Ajuste                                                |Vantagem                         |Desvantagem          |
|-------------|------------------------------------------------------|---------------------------------|---------------------|
|sem orçamento|cap 3 modelos, balanced/expanded seletivo             |Boa qualidade sem custo excessivo|Pode perder juiz top |
|-30%         |fundir rodadas, remover challenger, manter 1 validador|Mantém esqueleto                 |Menos contraste      |
|-15%         |reduzir tokens exploratórios e diversidade            |Corta gordura                    |Menos ângulos        |
|na meta      |usar balanced                                         |Melhor custo-benefício           |—                    |
|+20%         |adicionar 1 challenger ou auditoria                   |Mais robustez                    |Mais latência        |
|+50%         |subir 1 papel para premium                            |Melhor arbitragem                |Retorno marginal cai |
|+100%        |apenas tarefas críticas                               |Robustez máxima                  |Overkill para maioria|

### Schema de saída do planner (JSON)

```json
{
  "archetype": "debate",
  "confidence": 0.85,
  "extracted_signals": { /* PromptSignals */ },
  "budget_mode": "balanced",
  "mix_bias": "diverse_panel_plus_judge",
  "selected_model_roles": ["general", "reasoner", "judge"],
  "selected_models": [
    { "name": "DeepSeek R1", "role": "reasoner", "round_ids": [1, 2] },
    { "name": "Moonshot V1", "role": "general", "round_ids": [1] },
    { "name": "Claude Sonnet 4.6", "role": "judge", "round_ids": [3] }
  ],
  "round_plan": [
    { "round_number": 1, "goal": "construir lados opostos", "models": ["DeepSeek R1", "Moonshot V1"], "input_tokens_per_model": 1800, "output_tokens_per_model": 650 }
  ],
  "estimated_total_cost_usd": 0.045,
  "budget_comparison_options": [
    { "delta_pct": -30, "advantages": ["Mais barato"], "disadvantages": ["Menos contraste"] }
  ],
  "warnings": ["tema volátil, recomenda-se pesquisa"],
  "user_facing_rationale": "3 rodadas com reasoner, challenger e juiz para maximizar contraste."
}
```

### Exemplos de teste

|Cenário        |Prompt                                                          |Budget|Modo             |Plano esperado                                     |
|---------------|----------------------------------------------------------------|------|-----------------|---------------------------------------------------|
|brainstorm     |30 ideias de posicionamento para clínica premium de saúde mental|—     |default_no_budget|3r/3m, diversidade alta, generalistas + crítico    |
|debate         |Lançar produto B2B agora ou esperar 6 meses                     |$8    |balanced         |3r/3m, reasoner + challenger + juiz                |
|deep_research  |Estado da arte de agentes médicos para triagem em saúde mental  |$20   |expanded         |4r/5m, research + reasoner + sintetizador + auditor|
|code_generation|Pipeline Python para classificar prompts, com testes e fallback |$6    |balanced         |3r/3m, coder + validator + judge                   |
|multimodal     |Analisar imagem de dashboard e identificar gargalos de receita  |$5    |balanced         |3r/3m, multimodal + reasoner + validator           |

-----

## Checklist de Implementação

- [ ] Cache local do /api/v1/models (349 modelos, ~500KB) — renovar a cada 1h
- [ ] stream: true para toda chamada (UX + evitar timeout em reasoning)
- [ ] Spending limits em todas as API keys
- [ ] Não enviar temperature/top_p para modelos OpenAI reasoning
- [ ] Usar reasoning.max_tokens (não effort) para Claude 4.6
- [ ] Detectar truncamento via finish_reason: “length”
- [ ] Monitorar usage.completion_tokens_details.reasoning_tokens
- [ ] Compressão entre rodadas via Gemini 2.0 Flash Lite ($0.00088/chamada)
- [ ] OAuth PKCE para produção (não expor API key no browser)