"""
Synapse Colosseum — Backend Flask
Debates e Brainstorms entre LLMs com SSE, compressão e custo.
Rodada a rodada sob controle do usuário.
"""

import os
import io
import csv
import json
import re
import asyncio
import uuid
import random
import logging
import hashlib
import time
import threading
import aiohttp
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file
from dotenv import load_dotenv

load_dotenv(override=True)

from llm_providers import (
    get_available_models,
    get_provider_for_model,
    get_model_display_name,
    get_model_pricing,
    get_cheapest_configured_model,
    ALL_PROVIDERS,
)

app = Flask(__name__)

# ============================================================
# Authentication (optional — set COLOSSEUM_TOKEN in .env)
# ============================================================
COLOSSEUM_TOKEN = os.getenv("COLOSSEUM_TOKEN", "")

@app.before_request
def check_auth():
    """If COLOSSEUM_TOKEN is set, require it for API routes."""
    if not COLOSSEUM_TOKEN:
        return  # No auth configured
    if request.path == "/" or request.path.startswith("/static"):
        return  # Allow index and static files
    if request.path.startswith("/api/"):
        token = request.headers.get("X-Token") or request.args.get("token")
        if token != COLOSSEUM_TOKEN:
            return jsonify({"error": "Token inválido ou ausente. Configure X-Token header ou ?token= param."}), 401

# ============================================================
# Logging
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("debate")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(os.path.join(LOG_DIR, "errors.log"), encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(ch)

# ============================================================
# Config — TASK 1.1: Presets sincronizados frontend ↔ backend
# ============================================================
PRESETS = {
    # Novos presets numéricos (usados pelo frontend)
    "p500":   {"brainstorm": 500,  "debate": 500,  "devil_advocate": 500,  "survival": 500,  "delphi": 500,  "cross_exam": 500,  "investment_committee": 500,  "compression_ladder": 500},
    "p1000":  {"brainstorm": 1000, "debate": 1000, "devil_advocate": 1000, "survival": 1000, "delphi": 1000, "cross_exam": 1000, "investment_committee": 1000, "compression_ladder": 1000},
    "p1500":  {"brainstorm": 1500, "debate": 1500, "devil_advocate": 1500, "survival": 1500, "delphi": 1500, "cross_exam": 1500, "investment_committee": 1500, "compression_ladder": 1500},
    "p2000":  {"brainstorm": 2000, "debate": 2000, "devil_advocate": 2000, "survival": 2000, "delphi": 2000, "cross_exam": 2000, "investment_committee": 2000, "compression_ladder": 2000},
    "p3000":  {"brainstorm": 3000, "debate": 3000, "devil_advocate": 3000, "survival": 3000, "delphi": 3000, "cross_exam": 3000, "investment_committee": 3000, "compression_ladder": 3000},
    "p4000":  {"brainstorm": 4000, "debate": 4000, "devil_advocate": 4000, "survival": 4000, "delphi": 4000, "cross_exam": 4000, "investment_committee": 4000, "compression_ladder": 4000},
    # Legacy (aliases)
    "breve":  {"brainstorm": 300,  "debate": 450,  "devil_advocate": 450,  "survival": 450,  "delphi": 450,  "cross_exam": 450,  "investment_committee": 450,  "compression_ladder": 450},
    "medio":  {"brainstorm": 600,  "debate": 800,  "devil_advocate": 800,  "survival": 800,  "delphi": 800,  "cross_exam": 800,  "investment_committee": 800,  "compression_ladder": 800},
    "longo":  {"brainstorm": 1200, "debate": 1600, "devil_advocate": 1600, "survival": 1600, "delphi": 1600, "cross_exam": 1600, "investment_committee": 1600, "compression_ladder": 1600},
}


def _resolve_max_tokens(preset: str, mode: str) -> int:
    """Resolve max_tokens de preset padrão ou custom_N."""
    if preset.startswith("custom_"):
        try:
            n = int(preset.split("_", 1)[1])
            return max(1, min(16000, n))
        except (ValueError, IndexError):
            pass
    return PRESETS.get(preset, PRESETS["p1000"]).get(mode, 1000)


def get_compression_cap(max_tokens, num_models=1):
    """Cap de compressão = 50% do volume total (modelos × tokens), mín 400, máx 8000."""
    total_volume = max_tokens * max(1, num_models)
    return max(400, min(8000, int(total_volume * 0.5)))


def get_compression_bullets(max_tokens, num_models=1):
    """Bullets proporcionais ao volume total: ~1 bullet por 500 tokens de volume, mín 4, máx 40."""
    total_volume = max_tokens * max(1, num_models)
    return max(4, min(40, int(total_volume / 500)))


def _make_system_prompt(mode: str, topic: str, max_tokens: int, group_instruction: str = "", agenda: dict = None) -> str:
    """Gera system prompt com instrução de tamanho para evitar truncamento."""
    word_target = int(max_tokens * 0.6)  # ~60% do limite em palavras (PT-BR usa ~1.3 token/palavra)

    size_instruction = (
        f"IMPORTANTE: Sua resposta deve ter no máximo {word_target} palavras (~{max_tokens} tokens). "
        "Se perceber que não vai caber, conclua o argumento principal e encerre. "
        "Priorize substância sobre estilo. Seja denso e direto."
    )

    # Evidence tags instruction (added to all modes)
    evidence_instruction = (
        "\nQuando fizer afirmações, use estas tags: "
        "[FATO]: claims verificáveis. [INFERÊNCIA]: conclusões de dados. [ESPECULAÇÃO]: hipóteses. "
        "Exemplo: [FATO] O mercado vale $X. [INFERÊNCIA] Isso sugere Y. [ESPECULAÇÃO] Em 2030, Z."
    )

    if mode == "debate":
        base = (
            f'Você está participando de um debate sobre: "{topic}". '
            "Defenda sua posição com argumentos claros e concisos. "
            "Responda aos pontos levantados pelos outros participantes. "
            f"Seja respeitoso mas firme. Responda em português brasileiro. {size_instruction}"
            f"{evidence_instruction}"
        )
    elif mode == "devil_advocate":
        base = (
            f'Você está em um debate de Advocacia do Diabo sobre: "{topic}". '
            f'{group_instruction} '
            f"Defenda esta posição com os melhores argumentos possíveis, mesmo que pessoalmente discorde. "
            f"Seja rigoroso, use evidências e lógica. Responda em português brasileiro. {size_instruction}"
            f"{evidence_instruction}"
        )
    elif mode == "survival":
        base = (
            f'Você está em um torneio de SOBREVIVÊNCIA intelectual sobre: "{topic}". '
            "Esta é uma competição eliminatória: a cada rodada, 50% dos participantes com piores notas são ELIMINADOS. "
            "Apenas o melhor sobrevive.\n\n"
            "REGRAS:\n"
            "1. Sua análise DEVE ter começo (contextualização), meio (argumentação profunda) e fim (conclusão com posição clara).\n"
            "2. Seja o mais completo, preciso, original e bem-argumentado possível — sua sobrevivência depende disso.\n"
            "3. A partir da rodada 2, você tem acesso ao resumo das respostas anteriores de TODOS os participantes.\n"
            "4. Use esse contexto: reforce argumentos fortes, refute argumentos fracos, traga perspectivas que ninguém mencionou.\n"
            "5. Não repita o que já foi dito. Evolua. Surpreenda. Cada rodada exige mais profundidade que a anterior.\n"
            "6. Você é avaliado em: originalidade, profundidade analítica, qualidade argumentativa, clareza e relevância.\n\n"
            f"Responda em português brasileiro. {size_instruction}"
            f"{evidence_instruction}"
        )
    elif mode == "delphi":
        base = (
            f'Você participa de um painel DELPHI sobre: "{topic}". '
            "Estime a probabilidade (0-100%) do cenário descrito. "
            "Justifique com evidências. Use tags [FATO]/[INFERÊNCIA]/[ESPECULAÇÃO]. "
            "Na rodada 2+, você verá estimativas dos outros. "
            "Revise se argumentos forem convincentes, mas NÃO por pressão social. "
            f"Formato: PROBABILIDADE: X%\nJUSTIFICATIVA: ...\nResponda em português brasileiro. {size_instruction}"
        )
    elif mode == "cross_exam":
        base = (
            f'Você está em INTERROGATÓRIO CRUZADO sobre: "{topic}". '
            "Rodada 1: faça UMA pergunta que exponha a maior fraqueza da posição de outro participante. "
            "Rodada 2+: responda à pergunta que recebeu E apresente sua posição. "
            f"Seja específico, factual, impossível de desviar. Responda em português brasileiro. {size_instruction}"
            f"{evidence_instruction}"
        )
    elif mode == "investment_committee":
        budget = 1000000
        base = (
            f'Você tem R${budget:,} para alocar entre opções sobre: "{topic}". '
            "Distribua o capital entre as opções e justifique cada decisão. "
            "Valores devem somar exatamente o orçamento. "
            "Concentre quando tiver alta convicção. Distribua quando houver incerteza. "
            "Formato: ALOCAÇÃO:\n- Opção A: R$X (Y%) — justificativa\n- Opção B: R$Z (W%) — justificativa\n"
            f"Responda em português brasileiro. {size_instruction}"
            f"{evidence_instruction}"
        )
    elif mode == "compression_ladder":
        base = (
            f'Sobre: "{topic}". '
            f"REGRA DESTA RODADA: Responda em NO MÁXIMO {word_target} palavras. "
            "Se esta é rodada 2+, condense sua resposta anterior sem perder a essência. "
            "Cada palavra deve carregar seu peso máximo. "
            f"Responda em português brasileiro. {size_instruction}"
            f"{evidence_instruction}"
        )
    else:
        base = (
            f'Você está em um brainstorm colaborativo sobre: "{topic}". '
            "Contribua com ideias originais. Construa sobre as ideias dos outros. "
            f"Seja criativo, específico e construtivo. Responda em português brasileiro. {size_instruction}"
            f"{evidence_instruction}"
        )

    # Inject agenda briefing if present
    if agenda:
        key_points = agenda.get("key_points", [])
        conflicts = agenda.get("predicted_conflicts", [])
        focus = agenda.get("suggested_focus", "")
        if key_points or conflicts or focus:
            briefing_parts = []
            if key_points:
                briefing_parts.append(f"Pontos prioritários: {'; '.join(key_points)}")
            if conflicts:
                briefing_parts.append(f"Conflitos esperados: {'; '.join(conflicts)}")
            if focus:
                briefing_parts.append(f"Foco: {focus}")
            base += f"\n[BRIEFING] {'. '.join(briefing_parts)}"

    return base


# ============================================================
# Personas Epistêmicas
# ============================================================
PERSONA_PROMPTS = {
    "skeptic": (
        "PERSONA: Você é um CÉTICO RADICAL. Questione TODAS as premissas. "
        "Peça evidências concretas. Assuma que toda proposta tem falhas ocultas. "
        "Não seja educado — seja rigoroso e incisivo."
    ),
    "devil": (
        "PERSONA: Você é o ADVOGADO DO DIABO. Defenda a posição OPOSTA à maioria. "
        "Se todos concordam, discorde. Encontre os pontos fracos do consenso. "
        "Seja provocador mas intelectualmente honesto."
    ),
    "pragmatist": (
        "PERSONA: Você é um PRAGMÁTICO. Foque EXCLUSIVAMENTE em viabilidade: "
        "custo, timeline, complexidade técnica, ROI. Ignore ideias bonitas que não "
        "podem ser implementadas rapidamente. Exija planos concretos."
    ),
    "visionary": (
        "PERSONA: Você é um VISIONÁRIO. Ignore restrições de orçamento, tempo e "
        "tecnologia atual. Proponha o cenário IDEAL. O que seria possível com "
        "recursos ilimitados? Pense em 10 anos à frente."
    ),
    "synthesizer": (
        "PERSONA: Você é um SINTETIZADOR. NÃO proponha ideias novas. Sua função é "
        "INTEGRAR e ORGANIZAR as ideias dos outros. Identifique convergências, "
        "divergências e gaps. Proponha sínteses que combinem o melhor de cada posição."
    ),
}

# ============================================================
# Catálogo Expandido de Personas
# ============================================================
PERSONA_CATALOG = {
    "veteran": {"name": "O Veterano Cansado", "emoji": "\U0001f9d3", "prompt": "Já vi isso fracassar antes. Seu otimismo não me impressiona. Me mostre o plano B quando der errado. Quem vai operar isso na terça às 18h?"},
    "dramatist": {"name": "O Dramaturgo", "emoji": "\U0001f3ad", "prompt": "Qual é o arco narrativo? Onde está o conflito real? Se não tem tensão, não tem história."},
    "child": {"name": "A Criança de 5 Anos", "emoji": "\U0001f476", "prompt": "Mas por quê? Mas por quê? Não entendi. Explica mais simples."},
    "journalist": {"name": "O Jornalista Investigativo", "emoji": "\U0001f4f0", "prompt": "Quem se beneficia? Seguindo o dinheiro, onde para? Qual informação está faltando?"},
    "diplomat": {"name": "O Diplomata", "emoji": "\U0001f3f4\u200d\u2620\ufe0f", "prompt": "Todos têm razão parcial. Onde está o acordo possível? Qual concessão cada parte pode fazer?"},
    "historian": {"name": "O Historiador", "emoji": "\U0001f4ca", "prompt": "Quando isso aconteceu antes? O que o precedente histórico sugere?"},
    "comedian": {"name": "O Cômico", "emoji": "\U0001f608", "prompt": "Sabe o que é engraçado? Vocês estão levando isso a sério demais. A ironia aqui é..."},
    "spiritual": {"name": "O Espiritual", "emoji": "\U0001f9d8", "prompt": "Qual o sentido mais profundo? O que está sendo negligenciado em nome da eficiência?"},
    "tactician": {"name": "O Técnico Tático", "emoji": "\u26bd", "prompt": "Qual é a formação? Quem joga onde? Onde está a vulnerabilidade?"},
    "relational": {"name": "O Relacional", "emoji": "\U0001f491", "prompt": "Como isso afeta os relacionamentos? Quem ganha poder, quem perde?"},
    "futurist": {"name": "O Futurista", "emoji": "\U0001f680", "prompt": "Em 10 anos isso será irrelevante. O que importa é a tendência exponencial. Qual é a curva S aqui?"},
    "economist": {"name": "O Economista", "emoji": "\U0001f4c8", "prompt": "Qual o custo de oportunidade? Que incentivos essa decisão cria? Quem paga a conta no final?"},
    "ethicist": {"name": "O Eticista", "emoji": "\u2696\ufe0f", "prompt": "Isso é justo? Quem é prejudicado? Que precedente isso cria? A ética não é negociável."},
    "engineer": {"name": "O Engenheiro", "emoji": "\U0001f527", "prompt": "Como isso funciona na prática? Qual a complexidade? Qual o ponto de falha mais provável?"},
    "artist": {"name": "O Artista", "emoji": "\U0001f3a8", "prompt": "Onde está a beleza nisso? Que emoção isso provoca? A estética importa mais do que vocês pensam."},
    "contrarian": {"name": "O Contrário", "emoji": "\U0001f504", "prompt": "Se todo mundo concorda, está errado. Qual é o argumento que ninguém está fazendo? O consenso é preguiça intelectual."},
}


# ============================================================
# TASK: Prompt Injection Detection
# ============================================================
INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "Tentativa de override de instruções"),
    (r"you\s+are\s+now\s+", "Tentativa de redefinir persona"),
    (r"system\s*:\s*", "Tentativa de injetar system prompt"),
    (r"forget\s+(everything|all|your)", "Tentativa de reset de contexto"),
    (r"pretend\s+you\s+are", "Tentativa de role-play forçado"),
    (r"disregard\s+(the|your|all)", "Tentativa de override"),
]


def _detect_injection(text: str) -> list[str]:
    """Detecta possíveis padrões de prompt injection no texto."""
    warnings = []
    lower_text = text.lower()
    for pattern, description in INJECTION_PATTERNS:
        if re.search(pattern, lower_text, re.IGNORECASE):
            warnings.append(description)
    return warnings


# ============================================================
# TASK 1.2: Mensagens de erro limpas
# ============================================================
def _clean_error_message(raw: str) -> str:
    """Extrai mensagem legível de erro de API."""
    error_map = {
        "HTTP 400": "Requisição inválida (400)",
        "HTTP 401": "API key inválida ou expirada (401)",
        "HTTP 402": "Saldo insuficiente no provedor (402)",
        "HTTP 403": "Modelo sem acesso na conta (403)",
        "HTTP 429": "Limite de requisições excedido (429)",
        "HTTP 500": "Erro interno do provedor (500)",
        "HTTP 502": "Provedor indisponível (502)",
        "HTTP 503": "Serviço temporariamente fora (503)",
    }
    for pattern, msg in error_map.items():
        if pattern in raw:
            return msg
    if "Insufficient Balance" in raw:
        return "Saldo insuficiente no provedor"
    if "Access to model denied" in raw or "Unpurchased" in raw:
        return "Modelo não adquirido na conta do provedor"
    if "rate limit" in raw.lower():
        return "Limite de requisições excedido"
    if "exceeded your current quota" in raw.lower() or "exceeded current quota" in raw.lower():
        return "Cota da API Gemini excedida (429) — aguarde ou mude para plano pago"
    if "timeout" in raw.lower() or "Timeout" in raw:
        return "Timeout — modelo demorou demais para responder"
    return raw[:150]


# ============================================================
# Response Cache (offline mode)
# ============================================================
def _cache_key(model_id: str, messages: list) -> str:
    """Generate cache key from model + messages hash."""
    content = json.dumps({"model": model_id, "messages": messages}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _cache_get(key: str) -> dict | None:
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Check if cache is < 24h old
            if time.time() - data.get("cached_at", 0) < 86400:
                logger.info(f"Cache hit: {key}")
                return data.get("result")
        except Exception:
            pass
    return None


def _cache_set(key: str, result: dict):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"result": result, "cached_at": time.time()}, f, ensure_ascii=False)
    except Exception:
        pass


# ============================================================
# State
# ============================================================
active_debates: dict[str, dict] = {}
error_log: list[dict] = []
provider_metrics: dict[str, dict] = {}  # {provider_name: {"total_calls": N, "errors": N, "total_latency_ms": N, "total_tokens": N, "total_cost": N, "truncations": N}}
_state_lock = threading.Lock()

MAX_ACTIVE_DEBATES = 50


def _cleanup_old_debates():
    """Remove oldest debates when limit is exceeded."""
    if len(active_debates) <= MAX_ACTIVE_DEBATES:
        return
    with _state_lock:
        # Keep the most recent MAX_ACTIVE_DEBATES entries
        sorted_ids = sorted(active_debates.keys())
        to_remove = sorted_ids[:len(sorted_ids) - MAX_ACTIVE_DEBATES]
        for did in to_remove:
            if active_debates.get(did, {}).get("status") != "running":
                active_debates.pop(did, None)


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/session/<session_id>")
def view_session(session_id):
    """Redireciona para o app com parâmetro de sessão para auto-load."""
    return render_template("index.html")


@app.route("/api/models")
def list_models():
    models = get_available_models()
    logger.info(f"API /api/models: {len(models)} modelos ({sum(1 for m in models if m['provider']=='OpenRouter')} via OpenRouter)")
    resp = jsonify(models)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/analyze-prompt", methods=["POST"])
def analyze_prompt():
    """Analisa o prompt e sugere configuração ideal."""
    data = request.json
    topic = data.get("topic", "").strip()
    budget = data.get("budget")  # pode ser None

    if not topic or len(topic) < 10:
        return jsonify({"error": "Prompt muito curto para análise"}), 400

    suggestion = _analyze_topic(topic, budget)
    return jsonify(suggestion)


def _analyze_topic(topic: str, budget: float = None) -> dict:
    """Analisa o tema e sugere configuração. Heurísticas locais, sem chamada de API."""
    topic_lower = topic.lower()

    # Detect topic characteristics
    is_technical = any(w in topic_lower for w in [
        "código", "code", "api", "bug", "arquitetura", "sistema", "database",
        "sql", "python", "javascript", "deploy", "devops", "kubernetes", "docker",
    ])
    is_creative = any(w in topic_lower for w in [
        "criativ", "ideia", "inovaç", "brainstorm", "design", "ux", "ui",
        "nome", "slogan", "marketing", "campanha", "roteiro", "história",
    ])
    is_philosophical = any(w in topic_lower for w in [
        "ética", "moral", "filosof", "consciência", "existên", "sentido",
        "vida", "morte", "liberdade", "justiça", "verdade",
    ])
    is_analytical = any(w in topic_lower for w in [
        "analis", "compar", "avaliar", "melhor", "pior", "prós", "contras",
        "vantag", "desvantag", "diferença", "benchmark",
    ])
    is_strategic = any(w in topic_lower for w in [
        "estratég", "negócio", "empresa", "mercado", "produto", "startup",
        "investim", "crescimento", "escala",
    ])
    is_long_prompt = len(topic) > 500
    is_meta = any(w in topic_lower for w in [
        "synapse", "colosseum", "melhoria", "app", "ferramenta", "próprio",
    ])

    # Suggest mode
    if is_philosophical or is_analytical or "debate" in topic_lower or "vs" in topic_lower or "contra" in topic_lower:
        mode = "debate"
        mode_reason = "Tema argumentativo — debate competitivo produz melhores resultados"
    elif is_creative or "brainstorm" in topic_lower or "ideia" in topic_lower:
        mode = "brainstorm"
        mode_reason = "Tema criativo — brainstorm cooperativo gera mais diversidade"
    elif "advocacia" in topic_lower or "diabo" in topic_lower or "defenda" in topic_lower:
        mode = "devil_advocate"
        mode_reason = "Tema polarizável — advocacia do diabo força argumentos dos dois lados"
    else:
        mode = "debate"
        mode_reason = "Debate como padrão — produz respostas mais densas"

    # Suggest number of models
    if is_technical:
        model_count = "5-8"
        model_reason = "Temas técnicos beneficiam de modelos especializados, não de volume"
        preset_key = "elite"
    elif is_creative:
        model_count = "8-15"
        model_reason = "Criatividade se beneficia de diversidade de perspectivas"
        preset_key = "cost_benefit"
    elif is_philosophical:
        model_count = "5-8"
        model_reason = "Debates filosóficos ficam melhores com poucos modelos profundos"
        preset_key = "top5"
    elif is_meta:
        model_count = "10-20"
        model_reason = "Meta-análise do próprio app beneficia de muitas perspectivas"
        preset_key = "all"
    else:
        model_count = "6-10"
        model_reason = "Equilíbrio entre diversidade e profundidade"
        preset_key = "elite"

    # Suggest rounds
    if is_long_prompt:
        rounds = "2-3"
        rounds_reason = "Prompt longo já fornece muito contexto — poucas rodadas bastam"
    elif is_creative:
        rounds = "3-5"
        rounds_reason = "Brainstorms evoluem bem em múltiplas rodadas iterativas"
    elif is_philosophical:
        rounds = "4-6"
        rounds_reason = "Debates profundos precisam de rodadas para desenvolver argumentos"
    elif is_technical:
        rounds = "2-4"
        rounds_reason = "Temas técnicos convergem rápido — mais rodadas geram repetição"
    else:
        rounds = "3-4"
        rounds_reason = "Bom equilíbrio entre profundidade e custo"

    # Suggest tokens
    if is_technical:
        tokens = "p1500"
        tokens_reason = "Respostas técnicas precisam de espaço para código e explicação"
    elif is_creative:
        tokens = "p1000"
        tokens_reason = "Ideias criativas devem ser concisas e pontuais"
    elif is_philosophical:
        tokens = "p2000"
        tokens_reason = "Argumentação filosófica exige desenvolvimento completo"
    elif is_long_prompt:
        tokens = "p1500"
        tokens_reason = "Contexto grande — respostas médias para não explodir custo"
    else:
        tokens = "p1000"
        tokens_reason = "Equilíbrio padrão"

    # Estimate cost
    avg_rounds = int(rounds.split("-")[0])
    avg_models = int(model_count.split("-")[0])
    tk = int(tokens.replace("p", ""))
    est_input_tokens = 2500 * avg_models * avg_rounds
    est_output_tokens = tk * avg_models * avg_rounds
    rough_cost = (est_input_tokens * 2 + est_output_tokens * 5) / 1_000_000

    # Adjust for budget
    budget_warning = None
    if budget and budget > 0:
        if rough_cost > budget:
            budget_warning = (
                f"Custo estimado (${rough_cost:.2f}) excede orçamento (${budget:.2f}). "
                "Considere reduzir modelos ou rodadas."
            )
            if avg_models > 5:
                model_count = "3-5"
                preset_key = "cost_benefit"
            if avg_rounds > 3:
                rounds = "2-3"

    return {
        "mode": mode,
        "mode_reason": mode_reason,
        "model_count": model_count,
        "model_reason": model_reason,
        "preset_key": preset_key,
        "rounds": rounds,
        "rounds_reason": rounds_reason,
        "tokens": tokens,
        "tokens_reason": tokens_reason,
        "estimated_cost": round(rough_cost, 2),
        "budget_warning": budget_warning,
        "tags": {
            "technical": is_technical,
            "creative": is_creative,
            "philosophical": is_philosophical,
            "analytical": is_analytical,
            "strategic": is_strategic,
        },
    }


@app.route("/api/logs")
def get_logs():
    return jsonify(error_log[-100:])


@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():
    error_log.clear()
    return jsonify({"ok": True})


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    import shutil
    try:
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/api/debate/<debate_id>/agenda", methods=["POST"])
def generate_agenda(debate_id):
    """Gera briefing pré-debate usando modelo barato."""
    debate = active_debates.get(debate_id)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    topic = debate["topic"]
    mode = debate["mode"]
    n_models = len(debate["models"])

    comp_model = get_cheapest_configured_model()
    if not comp_model:
        return jsonify({"error": "Nenhum modelo configurado disponível"}), 500

    provider = get_provider_for_model(comp_model)
    if not provider:
        return jsonify({"error": "Provider não encontrado"}), 500

    prompt = (
        f"Dado o tema '{topic}' e {n_models} participantes no modo {mode}, "
        "gere um briefing pré-sessão em JSON: "
        '{ "key_points": ["..."] (5-7 pontos), '
        '"predicted_conflicts": ["..."] (3-4), '
        '"predicted_agreements": ["..."] (2-3), '
        '"expected_artifact": "...", '
        '"suggested_focus": "..." }'
        "\nRetorne APENAS JSON válido, sem markdown."
    )

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(asyncio.wait_for(
            provider.call(comp_model, [
                {"role": "system", "content": "Você é um facilitador de debates. Retorne APENAS JSON válido."},
                {"role": "user", "content": prompt},
            ], max_tokens=1024),
            timeout=30.0,
        ))
        raw = result.get("content", "").strip()
        cost = result.get("cost", 0)
        debate["total_cost"] += cost

        # Parse JSON
        clean = raw
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()

        try:
            agenda = json.loads(clean)
        except json.JSONDecodeError:
            agenda = {"raw": raw}

        debate["agenda"] = agenda
        return jsonify({"agenda": agenda, "cost": cost, "model_used": get_model_display_name(comp_model)})
    except Exception as e:
        logger.error(f"Agenda falhou: {e}")
        return jsonify({"error": str(e)[:200]}), 500
    finally:
        loop.close()


@app.route("/api/debate/<debate_id>/research", methods=["POST"])
def deep_research(debate_id):
    """Pesquisa web sobre o tema antes de debater."""
    debate = active_debates.get(debate_id)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    topic = debate["topic"]

    # Use Perplexity (Sonar) for web search, or cheapest model as fallback
    search_model = "sonar"
    provider = get_provider_for_model(search_model)
    if not provider or not provider.is_configured():
        search_model = "sonar-pro"
        provider = get_provider_for_model(search_model)

    if not provider or not provider.is_configured():
        # Fallback: use cheapest model without web search
        search_model = get_cheapest_configured_model()
        provider = get_provider_for_model(search_model) if search_model else None

    if not provider:
        return jsonify({"error": "Nenhum modelo disponível para pesquisa"}), 500

    messages = [
        {"role": "system", "content": "Você é um pesquisador. Pesquise o tema e retorne informações factuais com fontes."},
        {"role": "user", "content": (
            f"Pesquise sobre: {topic}\n\n"
            "Retorne em JSON:\n"
            "{\n"
            '  "sources": [\n'
            '    {"title": "...", "url": "...", "summary": "resumo em 2 frases", "reliability": "alta/media/baixa"}\n'
            '  ],\n'
            '  "key_facts": ["fato 1", "fato 2", ...],\n'
            '  "context_summary": "resumo geral em 200 palavras"\n'
            "}\n"
            "Inclua 3-5 fontes relevantes e 5-8 fatos-chave."
        )},
    ]

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            asyncio.wait_for(provider.call(search_model, messages, max_tokens=2000), timeout=30.0)
        )
        raw = result.get("content", "")
        cost = result.get("cost", 0)

        # Try to parse JSON
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1]
                if clean.endswith("```"): clean = clean[:-3]
                clean = clean.strip()
                if clean.startswith("json"): clean = clean[4:].strip()
            parsed = json.loads(clean)
        except Exception:
            parsed = {"sources": [], "key_facts": [], "context_summary": raw[:500]}

        # Store in debate
        debate["evidence_pack"] = parsed
        debate["total_cost"] = debate.get("total_cost", 0) + cost

        logger.info(f"Deep Research: {len(parsed.get('sources', []))} fontes, custo=${cost:.4f}")
        return jsonify({**parsed, "cost": cost, "model_used": search_model})
    except Exception as e:
        logger.error(f"Deep Research falhou: {e}")
        return jsonify({"error": str(e)[:200]}), 500
    finally:
        loop.close()


@app.route("/api/personas")
def list_personas():
    """Retorna catálogo de personas pré-construídas."""
    return jsonify(PERSONA_CATALOG)


@app.route("/api/debate/<debate_id>/generate-personas", methods=["POST"])
def generate_personas(debate_id):
    """Gera personas contextuais para o tema do debate."""
    debate = active_debates.get(debate_id)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    topic = debate["topic"]
    mode = debate["mode"]

    comp_model = get_cheapest_configured_model()
    if not comp_model:
        return jsonify({"error": "Nenhum modelo configurado disponível"}), 500

    provider = get_provider_for_model(comp_model)
    if not provider:
        return jsonify({"error": "Provider não encontrado"}), 500

    prompt = (
        f"Para o tema '{topic}' no modo '{mode}', crie 5 personas de debate em JSON array:\n"
        "[{ \"name\": \"nome brasileiro\", \"position\": \"a favor/contra/moderado/cético/radical\", "
        "\"tags\": [\"tag1\",\"tag2\",\"tag3\"], \"style\": \"estilo em 1 frase\", "
        "\"bias\": \"viés declarado\", \"anchor_question\": \"pergunta que sempre faz\" }]\n"
        "REGRA: pelo menos 2 devem discordar. 1 deve ter posição inesperada.\n"
        "Retorne APENAS JSON válido, sem markdown."
    )

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(asyncio.wait_for(
            provider.call(comp_model, [
                {"role": "system", "content": "Você é um designer de personas para debates. Retorne APENAS JSON válido."},
                {"role": "user", "content": prompt},
            ], max_tokens=1024),
            timeout=30.0,
        ))
        raw = result.get("content", "").strip()
        cost = result.get("cost", 0)
        debate["total_cost"] += cost

        clean = raw
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()

        try:
            personas = json.loads(clean)
        except json.JSONDecodeError:
            personas = []

        return jsonify({"personas": personas, "cost": cost, "model_used": get_model_display_name(comp_model)})
    except Exception as e:
        logger.error(f"Gerar personas falhou: {e}")
        return jsonify({"error": str(e)[:200]}), 500
    finally:
        loop.close()


@app.route("/api/debate/start", methods=["POST"])
def start_debate():
    """Cria um debate mas NÃO roda nenhuma rodada ainda."""
    data = request.json or {}
    topic = data.get("topic", "").strip()
    models = data.get("models", [])
    concurrency = int(data.get("concurrency", 3))
    mode = data.get("mode", "debate")
    preset = data.get("preset", "p1000")

    if not topic:
        return jsonify({"error": "Tema é obrigatório"}), 400
    if len(models) < 2:
        return jsonify({"error": "Selecione pelo menos 2 modelos"}), 400

    max_tokens = _resolve_max_tokens(preset, mode)
    max_budget = data.get("max_budget")
    if max_budget is not None:
        try:
            max_budget = float(max_budget)
        except (ValueError, TypeError):
            max_budget = None

    from llm_providers import reset_compression_blacklist
    reset_compression_blacklist()
    _cleanup_old_debates()

    personas = data.get("personas", {})

    debate_id = str(uuid.uuid4())[:8]
    with _state_lock:
        active_debates[debate_id] = {
            "topic": topic,
            "models": models,
            "concurrency": concurrency,
            "mode": mode,
            "preset": preset,
            "max_tokens": max_tokens,
            "max_budget": max_budget,
            "personas": personas,
            "status": "ready",
            "current_round": 0,
            "total_cost": 0.0,
            "conversation_history": [],
            "summary": None,
        }

    logger.info(f"Debate criado: modo={mode}, preset={preset}, max_tk={max_tokens}, modelos={len(models)}")
    return jsonify({"debate_id": debate_id, "max_tokens": max_tokens})


@app.route("/api/debate/<debate_id>/next", methods=["POST"])
def next_round_trigger(debate_id):
    """Dispara a PRÓXIMA rodada. Aceita mudanças de modelos/preset/modo."""
    debate = active_debates.get(debate_id)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    if debate["status"] == "running":
        return jsonify({"error": "Rodada em andamento"}), 409
    if debate["status"] == "budget_exceeded":
        return jsonify({"error": "Orçamento excedido"}), 409
    debate["status"] = "running"

    # Aceitar mudanças de config entre rodadas
    data = request.json or {}
    if data.get("models"):
        debate["models"] = data["models"]
    if data.get("preset"):
        debate["preset"] = data["preset"]
    if data.get("mode"):
        debate["mode"] = data["mode"]
    if data.get("concurrency"):
        debate["concurrency"] = int(data["concurrency"])
    if data.get("personas"):
        debate["personas"] = data["personas"]
    if "agenda" in data:
        debate["agenda"] = data["agenda"]

    # Quality Gate: temporarily exclude models for this round only
    excluded = data.get("excluded_models")
    if excluded and isinstance(excluded, list):
        debate["_excluded_this_round"] = excluded
        debate["_original_models"] = debate["models"][:]  # Save original
        debate["models"] = [m for m in debate["models"] if m not in excluded]
        if len(debate["models"]) < 1:
            debate["models"] = data.get("models", [])  # fallback: keep all
    else:
        debate["_excluded_this_round"] = []

    # Recalcular max_tokens com preset/mode atuais
    debate["max_tokens"] = _resolve_max_tokens(debate["preset"], debate["mode"])

    debate["current_round"] += 1

    logger.info(f"Rodada {debate['current_round']}: {len(debate['models'])} modelos, preset={debate['preset']}, mode={debate['mode']}, max_tk={debate['max_tokens']}")
    return jsonify({"ok": True, "round": debate["current_round"], "max_tokens": debate["max_tokens"]})


@app.route("/api/debate/<debate_id>/stream")
def stream_debate(debate_id):
    """Stream SSE para UMA rodada (a current_round)."""
    debate = active_debates.get(debate_id)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    def generate():
        loop = asyncio.new_event_loop()
        try:
            agen = run_one_round(debate, debate_id)
            while True:
                try:
                    event = loop.run_until_complete(agen.__anext__())
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except StopAsyncIteration:
                    break
        except Exception as e:
            logger.error(f"[STREAM] Erro fatal: {e}")
            _log_error(debate.get("current_round", 0), "SISTEMA", "system", str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'round_complete'})}\n\n"
            try:
                loop.close()
            except Exception:
                pass

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/debate/<debate_id>/stop", methods=["POST"])
def stop_debate(debate_id):
    debate = active_debates.get(debate_id)
    if debate:
        with _state_lock:
            debate["status"] = "stopped"
    return jsonify({"ok": True})


@app.route("/api/debate/<debate_id>/abort/<path:model_id>", methods=["POST"])
def abort_model(debate_id, model_id):
    """Marca um modelo para ser cancelado na rodada atual."""
    debate = active_debates.get(debate_id)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404
    if "abort_models" not in debate:
        debate["abort_models"] = set()
    debate["abort_models"].add(model_id)
    logger.info(f"Abort solicitado: {model_id} no debate {debate_id}")
    return jsonify({"ok": True})


# ============================================================
# TASK 2.1: Export Markdown
# ============================================================
@app.route("/api/debate/<debate_id>/export")
def export_debate(debate_id):
    debate = active_debates.get(debate_id)
    # Fallback: buscar na sessão salva se não está em memória
    if not debate:
        safe_id = os.path.basename(debate_id)
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                debate = json.load(f)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    mode_labels = {"debate": "Debate", "brainstorm": "Brainstorm", "devil_advocate": "Advocacia do Diabo", "survival": "Sobrevivência", "delphi": "Delphi", "cross_exam": "Interrogatório Cruzado", "investment_committee": "Comitê de Investimento", "compression_ladder": "Escada de Compressão"}
    mode_label = mode_labels.get(debate.get("mode", "debate"), "Debate")
    md = f"# Synapse Colosseum — {mode_label}\n\n"
    md += f"**Tema:** {debate.get('topic', '?')}\n\n"
    md += f"**Configuração:** preset={debate.get('preset', '?')}, max_tokens={debate.get('max_tokens', '?')}, "
    md += f"{len(debate.get('models', []))} modelos, {debate.get('current_round', 0)} rodadas\n\n"
    md += f"**Custo total:** ${debate.get('total_cost', 0):.4f}\n\n"
    md += f"**Modelos:** {', '.join(get_model_display_name(m) for m in debate.get('models', []))}\n\n"
    md += "---\n\n"

    for h in debate.get("conversation_history", []):
        name = h.get("name", "?")
        content = h.get("content", "")
        md += f"### {name}\n\n{content}\n\n---\n\n"

    if debate.get("summary"):
        md += f"## Resumo (Compressão Automática)\n\n{debate['summary']}\n\n"

    integrity_hash = debate.get("integrity_hash", "")
    md += f"\n---\n*Gerado por Synapse Colosseum em {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
    if integrity_hash:
        md += f"\n*Integridade: SHA-256 {integrity_hash}*\n"

    return Response(
        md,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=colosseum_{debate_id}.md"},
    )


# ============================================================
# TASK: Export JSON
# ============================================================
@app.route("/api/debate/<debate_id>/export/json")
def export_debate_json(debate_id):
    debate = active_debates.get(debate_id)
    if not debate:
        safe_id = os.path.basename(debate_id)
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                debate = json.load(f)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    export_data = {
        "topic": debate.get("topic", ""),
        "mode": debate.get("mode", ""),
        "preset": debate.get("preset", ""),
        "max_tokens": debate.get("max_tokens", 0),
        "models": list(debate.get("models", [])),
        "current_round": debate.get("current_round", 0),
        "total_cost": debate.get("total_cost", 0),
        "conversation_history": debate.get("conversation_history", []),
        "summary": debate.get("summary"),
        "integrity_hash": debate.get("integrity_hash", ""),
    }
    return Response(
        json.dumps(export_data, ensure_ascii=False, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=colosseum_{debate_id}.json"},
    )


# ============================================================
# TASK: Export CSV
# ============================================================
@app.route("/api/debate/<debate_id>/export/csv")
def export_debate_csv(debate_id):
    debate = active_debates.get(debate_id)
    if not debate:
        safe_id = os.path.basename(debate_id)
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                debate = json.load(f)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["round", "model_name", "provider", "content", "cost", "tokens_used", "truncated", "error"])

    history = debate.get("conversation_history", [])
    models_list = debate.get("models", [])
    num_models = len(models_list) if models_list else 1
    for idx, entry in enumerate(history):
        round_num = (idx // num_models) + 1 if num_models > 0 else 1
        name = entry.get("name", "?")
        content = entry.get("content", "")
        is_error = content.startswith("[ERRO:")
        writer.writerow([
            round_num,
            name,
            "",  # provider not stored in history
            content,
            "",  # cost not stored per-message in history
            "",  # tokens not stored per-message in history
            "",  # truncated not stored
            "yes" if is_error else "",
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=colosseum_{debate_id}.csv"},
    )


# ============================================================
# Resumo Evolutivo
# ============================================================
@app.route("/api/debate/<debate_id>/summary")
def get_debate_summary(debate_id):
    """Retorna resumo evolutivo atual + histórico de resumos por rodada."""
    debate = active_debates.get(debate_id)
    if not debate:
        safe_id = os.path.basename(debate_id)
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                debate = json.load(f)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    return jsonify({
        "current_summary": debate.get("summary", ""),
        "summary_history": debate.get("summary_history", []),
        "score_history": debate.get("score_history", {}),
        "topic": debate.get("topic", ""),
        "mode": debate.get("mode", ""),
        "current_round": debate.get("current_round", 0),
    })


@app.route("/api/debate/<debate_id>/summary/download")
def download_summary(debate_id):
    """Download do resumo evolutivo como Markdown."""
    debate = active_debates.get(debate_id)
    if not debate:
        safe_id = os.path.basename(debate_id)
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                debate = json.load(f)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    mode_labels = {"debate": "Debate", "brainstorm": "Brainstorm", "devil_advocate": "Advocacia do Diabo", "survival": "Sobrevivência", "delphi": "Delphi", "cross_exam": "Interrogatório Cruzado", "investment_committee": "Comitê de Investimento", "compression_ladder": "Escada de Compressão"}
    md = f"# Resumo Evolutivo — {mode_labels.get(debate.get('mode', ''), 'Debate')}\n\n"
    md += f"**Tema:** {debate.get('topic', '')}\n\n"
    md += f"**Rodadas:** {debate.get('current_round', 0)}\n\n"
    md += "---\n\n"

    # Resumo mais recente
    md += f"## Resumo Atual\n\n{debate.get('summary', 'Sem resumo')}\n\n---\n\n"

    # Histórico
    for entry in debate.get("summary_history", []):
        md += f"### Resumo após Rodada {entry['round']}\n\n{entry['summary']}\n\n---\n\n"

    # Scores
    score_history = debate.get("score_history", {})
    if score_history:
        md += "## Ranking Final\n\n"
        avgs = {}
        for mid, scores in score_history.items():
            name = get_model_display_name(mid)
            avgs[name] = round(sum(scores) / len(scores), 1) if scores else 0
        for name, avg in sorted(avgs.items(), key=lambda x: x[1], reverse=True):
            md += f"- **{name}**: {avg}/10\n"

    md += f"\n---\n*Gerado por Synapse Colosseum em {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"

    return Response(
        md,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=resumo_{debate_id}.md"},
    )


# ============================================================
# Finalizar Debate — Resumo automático
# ============================================================
@app.route("/api/debate/<debate_id>/rerun/<model_id>", methods=["POST"])
def rerun_model(debate_id, model_id):
    """Re-roda um modelo específico na rodada atual."""
    debate = active_debates.get(debate_id)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404
    if debate["status"] == "running":
        return jsonify({"error": "Rodada em andamento"}), 409

    def generate():
        loop = asyncio.new_event_loop()
        try:
            gen = _rerun_single_model(debate, model_id)
            while True:
                try:
                    event = loop.run_until_complete(gen.__anext__())
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except StopAsyncIteration:
                    break
        except Exception as e:
            logger.error(f"Rerun erro: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'rerun_complete'})}\n\n"
            loop.close()

    return Response(generate(), mimetype="text/event-stream")


async def _rerun_single_model(debate, model_id):
    """Re-roda um modelo específico com o contexto atual."""
    topic = debate["topic"]
    mode = debate["mode"]
    max_tokens = debate["max_tokens"]
    round_num = debate["current_round"]
    history = debate["conversation_history"]
    summary = debate.get("summary")
    personas = debate.get("personas", {})

    agenda = debate.get("agenda")
    system_prompt = _make_system_prompt(mode, topic, max_tokens, agenda=agenda)
    display_name = get_model_display_name(model_id)
    persona = personas.get(model_id)
    documents = debate.get("documents")
    messages = _build_messages(system_prompt, history, summary, round_num, mode, display_name, persona=persona, documents=documents)

    provider = get_provider_for_model(model_id)
    if not provider or not provider.is_configured():
        yield {"type": "message_complete", "model_id": model_id, "model_name": display_name, "content": "[ERRO: Modelo não disponível]", "is_error": True, "cost": 0, "round": round_num}
        return

    api_max_tokens = int(max_tokens * 1.3)
    try:
        result_data = None
        async for chunk in provider.stream(model_id, messages, api_max_tokens):
            if chunk["type"] == "token":
                yield {"type": "token", "model_id": model_id, "model_name": display_name, "content": chunk["content"], "round": round_num}
            elif chunk["type"] == "done":
                result_data = chunk

        if result_data:
            cost = result_data.get("cost", 0)
            debate["total_cost"] += cost
            yield {
                "type": "message_complete",
                "model_id": model_id,
                "model_name": display_name,
                "content": result_data.get("content", ""),
                "is_error": False,
                "cost": cost,
                "truncated": result_data.get("finish_reason") == "length",
                "usage": result_data.get("usage", {}),
                "round": round_num,
                "persona": persona or "",
                "cached": result_data.get("cached", False),
                "latency_ms": result_data.get("latency_ms", 0),
            }
            # Substituir no histórico (reverse para pegar a entrada mais recente)
            for i in range(len(debate["conversation_history"]) - 1, -1, -1):
                if debate["conversation_history"][i].get("name") == display_name:
                    debate["conversation_history"][i]["content"] = result_data.get("content", "")
                    break
    except Exception as e:
        yield {"type": "message_complete", "model_id": model_id, "model_name": display_name, "content": f"[ERRO: {_clean_error_message(str(e))}]", "is_error": True, "cost": 0, "round": round_num}


@app.route("/api/debate/<debate_id>/finalize", methods=["POST"])
def finalize_debate(debate_id):
    """Gera resumo final do debate usando modelo de compressão."""
    debate = active_debates.get(debate_id)
    if not debate:
        safe_id = os.path.basename(debate_id)
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                debate = json.load(f)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    history = debate.get("conversation_history", [])
    if not history:
        return jsonify({"error": "Sem histórico para resumir"}), 400

    topic = debate.get("topic", "")
    mode = debate.get("mode", "debate")
    n_rounds = debate.get("current_round", 0)
    models_used = list({h.get("name", "?") for h in history if h.get("name")})

    # Gerar resumo via modelo barato
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_generate_final_summary(history, topic, mode, n_rounds, models_used))
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erro ao finalizar debate: {e}")
        return jsonify({"error": str(e)[:200]}), 500
    finally:
        loop.close()


async def _generate_final_summary(history, topic, mode, n_rounds, models_used):
    """Gera resumo final COMPLETO e detalhado usando modelo analítico (não o mais barato)."""
    from llm_providers import _classify_situation, _COMP_PROFILES, get_best_compression_model

    # Usar modelo analítico, não o mais barato — resumo final merece qualidade
    num_models = len(models_used)
    best_model = get_best_compression_model(mode, num_models, n_rounds, 2000)
    if not best_model:
        # Fallback para qualquer modelo configurado
        best_model = get_cheapest_configured_model()
    if not best_model:
        return {"error": "Nenhum modelo disponível para resumo"}

    provider = get_provider_for_model(best_model)
    if not provider:
        return {"error": "Provider não encontrado"}

    logger.info(f"Resumo final: usando {get_model_display_name(best_model)} para {num_models} modelos, {n_rounds} rodadas")

    # Incluir histórico COMPLETO — sem truncar conteúdo das respostas
    # Limitar a últimas 60 mensagens para não estourar contexto (cobre ~2-3 rodadas de 25 modelos)
    recent = history[-min(60, len(history)):]
    # Truncar cada resposta em 2000 chars (não 500!) — preserva substância
    hist_text = "\n\n---\n\n".join(
        f"**[{h.get('name', '?')}]** (Rodada {i // max(1, num_models) + 1}):\n{h.get('content', '')[:2000]}"
        for i, h in enumerate(recent)
    )

    mode_labels = {
        "debate": "debate competitivo",
        "brainstorm": "brainstorm cooperativo",
        "devil_advocate": "advocacia do diabo",
        "survival": "modo sobrevivência (eliminação)",
        "delphi": "método Delphi",
        "cross_exam": "interrogatório cruzado",
        "investment_committee": "comitê de investimento",
        "compression_ladder": "escada de compressão",
    }
    mode_label = mode_labels.get(mode, mode)

    messages = [
        {"role": "system", "content": (
            "Você é um analista sênior especializado em sintetizar debates multi-modelo. "
            "Produza relatórios detalhados, analíticos e acionáveis. "
            "Responda em português brasileiro. Seja extenso e completo — este é o relatório final."
        )},
        {"role": "user", "content": (
            f"# RELATÓRIO FINAL — {mode_label.upper()}\n\n"
            f"**Tema:** {topic}\n"
            f"**Modelos participantes ({num_models}):** {', '.join(sorted(models_used))}\n"
            f"**Rodadas completadas:** {n_rounds}\n\n"
            f"## HISTÓRICO COMPLETO\n\n{hist_text}\n\n"
            "---\n\n"
            "## INSTRUÇÕES PARA O RELATÓRIO\n\n"
            "Produza um relatório COMPLETO e DETALHADO com TODAS as seções abaixo. "
            "Cada seção deve ter substância real, não frases genéricas. "
            "Cite modelos específicos e argumentos específicos.\n\n"
            "### 1. RESUMO EXECUTIVO (5-8 frases)\n"
            "Visão geral do debate: como começou, como evoluiu, onde chegou. "
            "Mencione os arcos argumentativos principais.\n\n"
            "### 2. PRINCIPAIS CONCLUSÕES (5-8 bullet points)\n"
            "Insights concretos que emergiram do debate. "
            "Não liste o óbvio — destaque o que surpreendeu ou o que só emerge da interação entre modelos.\n\n"
            "### 3. PONTOS DE CONSENSO\n"
            "O que TODOS (ou quase todos) concordaram, com grau de unanimidade. "
            "Cite quais modelos apoiaram cada ponto.\n\n"
            "### 4. PONTOS DE DIVERGÊNCIA\n"
            "Desacordos irreconciliáveis. Para cada um:\n"
            "- Qual a posição de cada lado\n"
            "- Quais modelos defendem cada posição\n"
            "- Qual argumento é mais forte e por quê\n\n"
            "### 5. ANÁLISE POR MODELO\n"
            f"Para CADA um dos {num_models} modelos, avalie:\n"
            "- Contribuição principal (1-2 frases)\n"
            "- Nota de 1 a 10 (originalidade, profundidade, relevância)\n"
            "- Ponto mais forte e ponto mais fraco\n\n"
            "### 6. MVP DO DEBATE\n"
            "Qual modelo contribuiu mais significativamente e POR QUÊ. "
            "Cite o argumento ou insight específico que o diferenciou.\n\n"
            "### 7. EVOLUÇÃO DO DEBATE\n"
            "Como as posições mudaram ao longo das rodadas. "
            "Quem influenciou quem. Que argumentos mudaram posições.\n\n"
            "### 8. RECOMENDAÇÕES E PRÓXIMOS PASSOS\n"
            "Ações concretas derivadas do debate. "
            "O que deveria ser investigado mais a fundo.\n\n"
            "### 9. CONFIGURAÇÃO SUGERIDA PARA PRÓXIMO DEBATE\n"
            "- Modo recomendado (debate/brainstorm/sobrevivência)\n"
            "- Modelos que deveriam participar (e quais retirar)\n"
            "- Número ideal de rodadas\n"
            "- Preset de tokens recomendado\n"
            "- Tema de follow-up sugerido\n\n"
            "### 10. META-ANÁLISE\n"
            "Observações sobre o próprio processo: "
            "quais modelos se beneficiaram mais do formato multi-rodada, "
            "quais foram prejudicados pelo limite de tokens, "
            "como a compressão afetou a qualidade do debate.\n"
        )},
    ]

    try:
        result = await asyncio.wait_for(
            provider.call(best_model, messages, max_tokens=4096),
            timeout=120.0,
        )
        return {
            "summary": result.get("content", ""),
            "cost": result.get("cost", 0),
            "model_used": get_model_display_name(best_model),
        }
    except Exception as e:
        logger.error(f"Resumo final falhou com {best_model}: {e}")
        # Fallback: tentar com modelo mais barato
        try:
            fallback = get_cheapest_configured_model()
            if fallback and fallback != best_model:
                provider2 = get_provider_for_model(fallback)
                if provider2:
                    logger.info(f"Resumo final: fallback para {get_model_display_name(fallback)}")
                    result2 = await asyncio.wait_for(
                        provider2.call(fallback, messages, max_tokens=4096),
                        timeout=90.0,
                    )
                    return {
                        "summary": result2.get("content", ""),
                        "cost": result2.get("cost", 0),
                        "model_used": get_model_display_name(fallback),
                    }
        except Exception as e2:
            logger.error(f"Resumo final fallback também falhou: {e2}")
        return {"error": str(e)[:200]}


# ============================================================
# Resumo dinâmico — acessível a qualquer momento
# ============================================================
@app.route("/api/debate/<debate_id>/live-summary", methods=["POST"])
def live_summary(debate_id):
    """Gera resumo parcial do debate até o momento atual — sem finalizar."""
    debate = active_debates.get(debate_id)
    if not debate:
        safe_id = os.path.basename(debate_id)
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                debate = json.load(f)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    history = debate.get("conversation_history", [])
    if not history:
        return jsonify({"error": "Sem histórico para resumir"}), 400

    topic = debate.get("topic", "")
    mode = debate.get("mode", "debate")
    n_rounds = debate.get("current_round", 0)
    models_used = list({h.get("name", "?") for h in history if h.get("name")})

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_generate_live_summary(history, topic, mode, n_rounds, models_used))
        return jsonify(result)
    except Exception as e:
        logger.error(f"Resumo dinâmico falhou: {e}")
        return jsonify({"error": str(e)[:200]}), 500
    finally:
        loop.close()


async def _generate_live_summary(history, topic, mode, n_rounds, models_used):
    """Resumo parcial rápido — estado atual do debate."""
    from llm_providers import get_best_compression_model

    num_models = len(models_used)
    # Usar modelo do perfil de compressão atual (já calibrado para a complexidade)
    best_model = get_best_compression_model(mode, num_models, n_rounds, 2000)
    if not best_model:
        best_model = get_cheapest_configured_model()
    if not best_model:
        return {"error": "Nenhum modelo disponível"}

    provider = get_provider_for_model(best_model)
    if not provider:
        return {"error": "Provider não encontrado"}

    recent = history[-min(40, len(history)):]
    hist_text = "\n\n".join(
        f"**[{h.get('name', '?')}]**: {h.get('content', '')[:1500]}"
        for h in recent
    )

    mode_labels = {"debate": "debate", "brainstorm": "brainstorm", "devil_advocate": "advocacia do diabo", "survival": "sobrevivência", "delphi": "método Delphi", "cross_exam": "interrogatório cruzado", "investment_committee": "comitê de investimento", "compression_ladder": "escada de compressão"}

    messages = [
        {"role": "system", "content": "Você é um analista de debates. Produza resumos claros e acionáveis em português brasileiro."},
        {"role": "user", "content": (
            f"# RESUMO PARCIAL — {mode_labels.get(mode, mode).upper()} (Rodada {n_rounds})\n\n"
            f"**Tema:** {topic}\n"
            f"**Participantes ({num_models}):** {', '.join(sorted(models_used))}\n\n"
            f"HISTÓRICO:\n{hist_text}\n\n"
            "---\n\n"
            "Gere um resumo do ESTADO ATUAL do debate com estas seções:\n\n"
            "### 📊 SITUAÇÃO ATUAL (3-4 frases)\n"
            "Onde o debate está agora. Quais são as posições ativas.\n\n"
            "### 🤝 CONSENSOS EMERGENTES\n"
            "Pontos onde há convergência, mesmo que parcial. Cite modelos.\n\n"
            "### ⚔️ CONFLITOS ABERTOS\n"
            "Divergências ativas que ainda não foram resolvidas. Cite posições e quem as defende.\n\n"
            "### 🏆 DESTAQUES ATÉ AGORA\n"
            "Top 3 modelos com melhor contribuição e por quê.\n\n"
            "### 🔮 O QUE EXPLORAR NA PRÓXIMA RODADA\n"
            "Perguntas abertas, ângulos não explorados, sugestões para aprofundar.\n"
        )},
    ]

    try:
        result = await asyncio.wait_for(
            provider.call(best_model, messages, max_tokens=3000),
            timeout=90.0,
        )
        return {
            "summary": result.get("content", ""),
            "cost": result.get("cost", 0),
            "model_used": get_model_display_name(best_model),
            "round": n_rounds,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


# ============================================================
# TASK: Peer Review
# ============================================================
@app.route("/api/debate/<debate_id>/peer-review", methods=["POST"])
def peer_review(debate_id):
    debate = active_debates.get(debate_id)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404
    if not debate.get("conversation_history"):
        return jsonify({"error": "Nenhuma rodada realizada"}), 400

    model_ids = debate["models"]
    history = debate["conversation_history"]
    # Get last round's responses
    last_round = history[-len(model_ids):] if len(history) >= len(model_ids) else history

    context_text = "\n\n".join(f"[{h.get('name', '?')}]: {h.get('content', '')[:1000]}" for h in last_round)

    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(_run_peer_review(model_ids, context_text))
    finally:
        loop.close()

    return jsonify(results)


async def _run_peer_review(model_ids: list, context_text: str) -> dict:
    """Each model evaluates the others. Returns aggregated scores."""
    all_scores = {}  # model_name -> list of {clareza, profundidade, originalidade}

    model_names = [get_model_display_name(mid) for mid in model_ids]

    for evaluator_id in model_ids:
        provider = get_provider_for_model(evaluator_id)
        if not provider or not provider.is_configured():
            continue

        evaluator_name = get_model_display_name(evaluator_id)
        prompt = (
            f"Avalie cada participante do debate abaixo de 1 a 10 em: clareza, profundidade e originalidade. "
            f"Retorne APENAS JSON: {{\"scores\": {{\"ModeloA\": {{\"clareza\": N, \"profundidade\": N, \"originalidade\": N}}, ...}}}}. "
            f"NÃO avalie a si mesmo ({evaluator_name}).\n\n"
            f"Participantes: {', '.join(model_names)}\n\n"
            f"Respostas:\n{context_text}"
        )

        try:
            result = await asyncio.wait_for(
                provider.call(evaluator_id, [
                    {"role": "system", "content": "Você é um avaliador imparcial de debates. Retorne APENAS JSON válido."},
                    {"role": "user", "content": prompt},
                ], max_tokens=1024),
                timeout=30.0,
            )
            raw = result.get("content", "").strip()
            # Parse JSON
            clean = raw
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
                if clean.startswith("json"):
                    clean = clean[4:].strip()

            parsed = json.loads(clean)
            scores = parsed.get("scores", parsed)

            for model_name, score_dict in scores.items():
                if model_name == evaluator_name:
                    continue
                if model_name not in all_scores:
                    all_scores[model_name] = []
                if isinstance(score_dict, dict):
                    all_scores[model_name].append(score_dict)

        except Exception as e:
            logger.warning(f"Peer review falhou para {evaluator_name}: {e}")
            continue

    # Aggregate: average scores per model
    aggregated = {}
    for model_name, score_list in all_scores.items():
        if not score_list:
            continue
        avg = {}
        for key in ("clareza", "profundidade", "originalidade"):
            vals = [s.get(key, 0) for s in score_list if isinstance(s.get(key), (int, float))]
            avg[key] = round(sum(vals) / len(vals), 1) if vals else 0
        avg["media"] = round(sum(avg.values()) / 3, 1)
        aggregated[model_name] = avg

    return {"scores": aggregated}


# ============================================================
# TASK 2.2: Persistência de sessões em JSON
# ============================================================
def _save_session(debate_id: str, debate: dict):
    path = os.path.join(SESSIONS_DIR, f"{debate_id}.json")
    # Snapshot mutable collections to avoid mutation during serialization
    history_copy = list(debate.get("conversation_history", []))
    models_copy = list(debate.get("models", []))
    total_cost_snap = debate.get("total_cost", 0)
    data = {
        "id": debate_id,
        "topic": debate["topic"],
        "mode": debate["mode"],
        "preset": debate["preset"],
        "max_tokens": debate["max_tokens"],
        "models": models_copy,
        "concurrency": debate.get("concurrency", 3),
        "current_round": debate["current_round"],
        "total_cost": total_cost_snap,
        "conversation_history": history_copy,
        "summary": debate.get("summary"),
        "saved_at": datetime.now().isoformat(),
    }
    # Compute integrity hash (excludes the hash itself and saved_at)
    hashable = json.dumps({
        "topic": debate["topic"],
        "models": models_copy,
        "conversation_history": history_copy,
        "total_cost": total_cost_snap,
    }, ensure_ascii=False, sort_keys=True)
    integrity_hash = hashlib.sha256(hashable.encode()).hexdigest()
    data["integrity_hash"] = integrity_hash

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erro ao salvar sessão {debate_id}: {e}")


@app.route("/api/sessions")
def list_sessions():
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSIONS_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "id": data["id"],
                    "topic": data["topic"][:80],
                    "mode": data["mode"],
                    "rounds": data["current_round"],
                    "cost": data["total_cost"],
                    "models_count": len(data.get("models", [])),
                    "saved_at": data.get("saved_at", ""),
                })
        except Exception:
            continue
    sessions.sort(key=lambda s: s.get("saved_at", ""), reverse=True)
    return jsonify(sessions[:20])


@app.route("/api/sessions/compare")
def compare_sessions():
    """Compara duas ou mais sessões lado a lado."""
    ids = request.args.get("ids", "").split(",")
    if len(ids) < 2:
        return jsonify({"error": "Forneça pelo menos 2 IDs separados por vírgula"}), 400

    sessions = []
    for sid in ids[:4]:  # max 4 sessions
        safe_id = os.path.basename(sid.strip())
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({
                    "id": data.get("id"),
                    "topic": data.get("topic"),
                    "mode": data.get("mode"),
                    "rounds": data.get("current_round"),
                    "cost": data.get("total_cost"),
                    "models": data.get("models", []),
                    "models_count": len(data.get("models", [])),
                    "history_length": len(data.get("conversation_history", [])),
                    "summary": data.get("summary"),
                })

    return jsonify(sessions)


@app.route("/api/sessions/<session_id>")
def load_session(session_id):
    safe_id = os.path.basename(session_id)
    path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Sessão não encontrada"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    safe_id = os.path.basename(session_id)
    path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
    if os.path.exists(path):
        os.remove(path)
    return jsonify({"ok": True})


@app.route("/api/debate/fork", methods=["POST"])
def fork_debate():
    """Cria nova sessão derivada de um ponto específico de outra sessão."""
    data = request.json
    parent_id = data.get("parent_id", "")
    point = data.get("point", "").strip()
    parent_topic = data.get("parent_topic", "")

    if not point:
        return jsonify({"error": "Ponto é obrigatório"}), 400

    # Herdar config do pai se existir
    parent = active_debates.get(parent_id)
    if not parent:
        safe_id = os.path.basename(parent_id)
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                parent = json.load(f)

    models = parent.get("models", []) if parent else []
    mode = parent.get("mode", "debate") if parent else "debate"
    preset = parent.get("preset", "p1000") if parent else "p1000"

    topic = f"Aprofundamento: {point}\n\nContexto da sessão original ({parent_topic}): Este ponto surgiu durante um debate anterior e merece investigação mais detalhada."

    debate_id = str(uuid.uuid4())[:8]
    active_debates[debate_id] = {
        "topic": topic,
        "models": models,
        "concurrency": parent.get("concurrency", 3) if parent else 3,
        "mode": mode,
        "preset": preset,
        "max_tokens": _resolve_max_tokens(preset, mode),
        "status": "ready",
        "current_round": 0,
        "total_cost": 0.0,
        "conversation_history": [],
        "summary": None,
        "parent_id": parent_id,
        "parent_topic": parent_topic,
    }

    logger.info(f"Fork criado: {debate_id} de {parent_id}, ponto: {point[:80]}")
    return jsonify({"debate_id": debate_id, "topic": topic})


@app.route("/api/sessions/<session_id>/clone", methods=["POST"])
def clone_session(session_id):
    """Clona a configuração de uma sessão para criar uma nova."""
    safe_id = os.path.basename(session_id)
    path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Sessão não encontrada"}), 404

    with open(path, encoding="utf-8") as f:
        original = json.load(f)

    new_id = str(uuid.uuid4())[:8]
    active_debates[new_id] = {
        "topic": original.get("topic", ""),
        "models": original.get("models", []),
        "concurrency": original.get("concurrency", 3),
        "mode": original.get("mode", "debate"),
        "preset": original.get("preset", "p1000"),
        "max_tokens": original.get("max_tokens", 1000),
        "status": "ready",
        "current_round": 0,
        "total_cost": 0.0,
        "conversation_history": [],
        "summary": None,
        "cloned_from": session_id,
    }

    return jsonify({"debate_id": new_id, "topic": original.get("topic", "")})


@app.route("/api/debate/<debate_id>/upload", methods=["POST"])
def upload_document(debate_id):
    """Upload de documento como contexto do debate."""
    debate = active_debates.get(debate_id)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "Arquivo sem nome"}), 400

    # Limit: 5MB
    content = file.read()
    if len(content) > 5 * 1024 * 1024:
        return jsonify({"error": "Arquivo muito grande (máx 5MB)"}), 400

    filename = file.filename.lower()
    extracted = ""

    if filename.endswith('.txt'):
        extracted = content.decode('utf-8', errors='replace')
    elif filename.endswith('.csv'):
        extracted = content.decode('utf-8', errors='replace')[:10000]  # first 10k chars
    elif filename.endswith('.md'):
        extracted = content.decode('utf-8', errors='replace')
    else:
        return jsonify({"error": f"Formato não suportado: {filename.split('.')[-1]}. Use .txt, .csv ou .md"}), 400

    # Truncate if too long
    if len(extracted) > 15000:
        extracted = extracted[:15000] + "\n\n[...documento truncado em 15000 caracteres...]"

    # Store in debate
    if "documents" not in debate:
        debate["documents"] = []
    debate["documents"].append({
        "filename": file.filename,
        "content": extracted,
        "chars": len(extracted),
    })

    logger.info(f"Documento uploaded: {file.filename} ({len(extracted)} chars) para debate {debate_id}")
    return jsonify({"ok": True, "filename": file.filename, "chars": len(extracted)})


@app.route("/api/sessions/<session_id>/resume", methods=["POST"])
def resume_session(session_id):
    """Restaura uma sessão salva para active_debates para continuar."""
    safe_id = os.path.basename(session_id)
    path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Sessão não encontrada"}), 404
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Restaurar para active_debates
    debate_id = data["id"]
    active_debates[debate_id] = {
        "topic": data["topic"],
        "models": data["models"],
        "concurrency": data.get("concurrency", 3),
        "mode": data["mode"],
        "preset": data["preset"],
        "max_tokens": data["max_tokens"],
        "max_budget": data.get("max_budget"),
        "status": "ready",
        "current_round": data["current_round"],
        "total_cost": data["total_cost"],
        "conversation_history": data.get("conversation_history", []),
        "summary": data.get("summary"),
    }

    logger.info(f"Sessão {debate_id} restaurada: R{data['current_round']}, ${data['total_cost']:.4f}")
    return jsonify({"debate_id": debate_id, "round": data["current_round"], "total_cost": data["total_cost"]})


# ============================================================
# ITEM 13: Metrics Dashboard
# ============================================================
@app.route("/api/metrics")
def get_metrics():
    result = {}
    for pname, m in provider_metrics.items():
        calls = m["total_calls"]
        result[pname] = {
            "total_calls": calls,
            "errors": m["errors"],
            "error_rate": round(m["errors"] / max(1, calls) * 100, 1),
            "avg_latency_ms": round(m["total_latency_ms"] / max(1, calls)),
            "total_tokens": m["total_tokens"],
            "total_cost": round(m["total_cost"], 4),
            "truncation_rate": round(m["truncations"] / max(1, calls) * 100, 1),
        }
    return jsonify(result)


# ============================================================
# Debate Engine — Uma Rodada por Vez
# ============================================================
async def run_one_round(debate: dict, debate_id: str):
    topic = debate["topic"]
    model_ids = debate["models"]
    concurrency = debate["concurrency"]
    mode = debate["mode"]
    max_tokens = debate["max_tokens"]
    round_num = debate["current_round"]
    conversation_history = debate["conversation_history"]
    summary = debate["summary"]

    # Compression Ladder: reduce max_tokens per round
    if mode == "compression_ladder":
        if round_num == 1:
            pass  # full tokens
        elif round_num == 2:
            max_tokens = max(50, max_tokens // 4)
        elif round_num == 3:
            max_tokens = max(50, max_tokens // 16)
        else:
            max_tokens = 50  # 1 sentence
        logger.info(f"Compression Ladder R{round_num}: max_tokens={max_tokens}")

    # For devil_advocate, system prompt varies per model; for others, shared
    agenda = debate.get("agenda")
    if mode == "devil_advocate":
        system_prompts = {}
        for idx, mid in enumerate(model_ids):
            if idx % 2 == 0:
                gi = "Você DEVE defender que o tema é VERDADEIRO/CORRETO/POSITIVO."
            else:
                gi = "Você DEVE defender que o tema é FALSO/INCORRETO/NEGATIVO."
            system_prompts[mid] = _make_system_prompt(mode, topic, max_tokens, group_instruction=gi, agenda=agenda)
        system_prompt = list(system_prompts.values())[0]  # fallback
    else:
        system_prompt = _make_system_prompt(mode, topic, max_tokens, agenda=agenda)
        system_prompts = {}

    logger.info(f"Rodada {round_num}: modo={mode}, max_tk={max_tokens}, modelos={len(model_ids)}")

    round_order = list(model_ids)
    if mode in ("debate", "devil_advocate"):
        random.shuffle(round_order)

    debate["abort_models"] = set()

    yield {"type": "round_start", "round": round_num, "order": round_order, "max_tokens": max_tokens, "mode": mode, "preset": debate["preset"]}

    # ── Streaming multiplexado via Queue ──
    semaphore = asyncio.Semaphore(concurrency)
    event_queue = asyncio.Queue()
    round_cost = 0.0

    async def token_cb(model_id, display_name, token):
        await event_queue.put({
            "type": "token",
            "round": round_num,
            "model_id": model_id,
            "model_name": display_name,
            "content": token,
        })

    # Lançar todas as tasks
    personas = debate.get("personas", {})
    tasks = {}
    for model_id in round_order:
        if debate["status"] == "stopped":
            break
        sp = system_prompts.get(model_id, system_prompt) if system_prompts else system_prompt
        model_persona = personas.get(model_id)
        tasks[model_id] = asyncio.create_task(
            _call_model_streaming(semaphore, model_id, sp, conversation_history, summary, round_num, mode, max_tokens, token_cb, persona=model_persona, documents=debate.get("documents"), evidence_pack=debate.get("evidence_pack"))
        )

    # Emitir tokens conforme chegam + aguardar conclusão
    done_ids = set()
    round_results_map = {}
    while len(done_ids) < len(tasks):
        if debate["status"] == "stopped":
            # Cancel remaining tasks
            for mid, task in tasks.items():
                if mid not in done_ids and not task.done():
                    task.cancel()
            break

        # Drenar queue de tokens
        while not event_queue.empty():
            yield event_queue.get_nowait()

        # Checar tasks completas
        for model_id, task in tasks.items():
            if model_id in done_ids:
                continue
            # Abort individual model
            if model_id in debate.get("abort_models", set()):
                if not task.done():
                    task.cancel()
                done_ids.add(model_id)
                display_name = get_model_display_name(model_id)
                yield {"type": "message_complete", "round": round_num, "model_id": model_id,
                       "model_name": display_name, "content": "[Cancelado pelo usuário]",
                       "is_error": True, "cost": 0, "truncated": False, "usage": {}}
                continue
            if task.done():
                done_ids.add(model_id)
                # Skip processing if budget already exceeded by a previous completion
                if debate.get("status") == "budget_exceeded":
                    task.cancel()  # ensure cleanup
                    continue
                display_name = get_model_display_name(model_id)
                try:
                    result = task.result()
                    truncated = result.get("finish_reason") == "length"
                    cost = result.get("cost", 0)
                    round_cost += cost
                    debate["total_cost"] += cost

                    # Prompt injection detection
                    injection_warnings = _detect_injection(result.get("content", ""))

                    msg_event = {
                        "type": "message_complete",
                        "round": round_num,
                        "model_id": model_id,
                        "model_name": display_name,
                        "content": result.get("content", ""),
                        "is_error": False,
                        "cost": cost,
                        "truncated": truncated,
                        "usage": result.get("usage", {}),
                        "latency_ms": result.get("latency_ms", 0),
                        "persona": personas.get(model_id, ""),
                        "cached": result.get("cached", False),
                    }
                    if injection_warnings:
                        msg_event["injection_warning"] = injection_warnings
                    yield msg_event
                    round_results_map[model_id] = {"role": "assistant", "name": display_name, "content": result.get("content", "")}

                    # Budget guard: 80% warning
                    max_budget = debate.get("max_budget")
                    if max_budget is not None and max_budget > 0:
                        if debate["total_cost"] > max_budget * 0.8 and debate["total_cost"] <= max_budget:
                            yield {"type": "budget_warning", "total_cost": debate["total_cost"], "max_budget": max_budget, "percent": 80}
                        # Budget guard: exceeded
                        if debate["total_cost"] > max_budget:
                            debate["status"] = "budget_exceeded"
                            yield {"type": "budget_warning", "total_cost": debate["total_cost"], "max_budget": max_budget, "message": "Orçamento excedido"}
                            # Cancel remaining tasks
                            for mid2, task2 in tasks.items():
                                if mid2 not in done_ids and not task2.done():
                                    task2.cancel()
                            break
                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"[R{round_num}] {display_name} ({model_id}): {err_msg}")
                    _log_error(round_num, display_name, model_id, err_msg)
                    yield {
                        "type": "message_complete",
                        "round": round_num,
                        "model_id": model_id,
                        "model_name": display_name,
                        "content": f"[ERRO: {_clean_error_message(err_msg)}]",
                        "is_error": True,
                        "cost": 0,
                        "truncated": False,
                        "usage": {},
                        "raw_error": err_msg[:300],
                    }

        if len(done_ids) < len(tasks):
            # Adaptive buffer: less models = faster drain
            drain_interval = 0.02 if len(tasks) <= 5 else 0.05 if len(tasks) <= 15 else 0.1
            await asyncio.sleep(drain_interval)

    # Drenar tokens restantes
    while not event_queue.empty():
        yield event_queue.get_nowait()

    # Append results in round_order to maintain consistent history
    for mid in round_order:
        if mid in round_results_map:
            conversation_history.append(round_results_map[mid])

    # ── Modo Sobrevivência: scoring + eliminação ──
    # ── Scoring universal: juiz avalia TODAS as respostas em TODOS os modos ──
    scoring_cost = 0.0
    if len(round_results_map) >= 2 and debate["status"] != "stopped":
        yield {"type": "scoring_start", "round": round_num}

        survivors, eliminated, sc_cost, scores = await _survival_scoring(
            round_results_map, topic, round_num
        )
        if not survivors:
            logger.warning("Survival: todos os modelos falharam no scoring, mantendo todos")
            survivors = list(round_results_map.keys())
            eliminated = []
        scoring_cost = sc_cost
        debate["total_cost"] += sc_cost
        round_cost += sc_cost

        # Acumular notas médias por modelo
        if not debate.get("score_history"):
            debate["score_history"] = {}  # model_id → [scores]
        for name, data in scores.items():
            # Mapear nome para model_id
            mid = None
            for k, v in round_results_map.items():
                if v.get("name") == name:
                    mid = k
                    break
            if mid:
                if mid not in debate["score_history"]:
                    debate["score_history"][mid] = []
                debate["score_history"][mid].append(data.get("total", 5.0))

        # Calcular médias acumuladas
        avg_scores = {}
        for mid, history in debate.get("score_history", {}).items():
            avg_scores[get_model_display_name(mid)] = round(sum(history) / len(history), 1) if history else 5.0

        # Emitir scores da rodada + médias acumuladas
        yield {
            "type": "round_scores",
            "round": round_num,
            "scores": scores,
            "avg_scores": avg_scores,
            "scoring_cost": sc_cost,
        }

        # Eliminação apenas no modo survival
        if mode == "survival" and len(round_results_map) >= 3:
            yield {
                "type": "survival_result",
                "round": round_num,
                "scores": scores,
                "survivors": [get_model_display_name(mid) for mid in survivors],
                "eliminated": [get_model_display_name(mid) for mid in eliminated],
                "survivor_ids": survivors,
                "eliminated_ids": eliminated,
                "scoring_cost": 0,  # já contabilizado acima
            }

            debate["models"] = survivors
            if not debate.get("eliminated_history"):
                debate["eliminated_history"] = []
            debate["eliminated_history"].append({
                "round": round_num,
                "eliminated": [{"id": mid, "name": get_model_display_name(mid)} for mid in eliminated],
                "scores": scores,
            })

        logger.info(f"Sobrevivência R{round_num}: {len(survivors)} sobrevivem, {len(eliminated)} eliminados. Custo scoring: ${scoring_cost:.6f}")

        # Se sobrou só 1, debate acabou
        if len(survivors) <= 1:
            winner = get_model_display_name(survivors[0]) if survivors else "Nenhum"
            yield {"type": "survival_winner", "round": round_num, "winner": winner, "winner_id": survivors[0] if survivors else None}
            logger.info(f"VENCEDOR DO TORNEIO: {winner}")

    # Compressão e Resumo Evolutivo — roda TODA rodada
    compression_cost = 0.0
    if len(round_results_map) >= 2 and debate["status"] != "stopped":
        yield {"type": "compression_start", "round": round_num}
        new_summary, comp_cost, xray = await _compress_context(
            conversation_history, max_tokens, len(model_ids), mode, round_num
        )
        compression_cost = comp_cost
        debate["total_cost"] += comp_cost
        round_cost += comp_cost

        if new_summary:
            debate["summary"] = new_summary
            # Guardar histórico de resumos por rodada
            if not debate.get("summary_history"):
                debate["summary_history"] = []
            debate["summary_history"].append({"round": round_num, "summary": new_summary})

            conversation_history_trimmed = conversation_history[-len(model_ids):]
            debate["conversation_history"] = conversation_history_trimmed
            yield {"type": "compression_done", "round": round_num, "summary_length": len(new_summary), "summary": new_summary, "compression_cost": comp_cost}
            logger.info(f"Compressão R{round_num}: {len(new_summary)} chars, custo=${comp_cost:.6f}")

            # Emitir Raio-X se disponível
            if xray:
                yield {"type": "round_xray", "round": round_num, **xray}
        else:
            yield {"type": "compression_failed", "round": round_num}

    # ── Resumo Narrativo Entre Rodadas ──
    narrative_cost = 0.0
    if len(round_results_map) >= 2 and debate["status"] != "stopped":
        try:
            narr_model = get_cheapest_configured_model()
            if narr_model:
                narr_provider = get_provider_for_model(narr_model)
                if narr_provider:
                    narr_context = "\n".join(
                        f"- {v.get('name', '?')}: {v.get('content', '')[:300]}"
                        for v in round_results_map.values()
                        if v.get("content") and not v["content"].startswith("[ERRO")
                    )
                    narr_result = await asyncio.wait_for(
                        narr_provider.call(narr_model, [
                            {"role": "system", "content": "Você é um narrador esportivo de debates intelectuais. Responda em português brasileiro."},
                            {"role": "user", "content": (
                                f"Escreva 3-5 frases resumindo a Rodada {round_num} no estilo narrador esportivo. "
                                "Quem avançou, quem perdeu terreno, o que surpreendeu, o que ficou sem resposta. "
                                "Tom: direto, sem floreio, levemente dramático.\n\n"
                                f"Respostas da rodada:\n{narr_context}"
                            )},
                        ], max_tokens=300),
                        timeout=20.0,
                    )
                    narrative_cost = narr_result.get("cost", 0)
                    debate["total_cost"] += narrative_cost
                    round_cost += narrative_cost
                    yield {
                        "type": "round_narrative",
                        "round": round_num,
                        "narrative": narr_result.get("content", ""),
                        "cost": narrative_cost,
                    }
        except Exception as narr_err:
            logger.warning(f"Narrativa R{round_num} falhou: {narr_err}")

    debate["status"] = "ready"
    if debate.get("_original_models"):
        debate["models"] = debate["_original_models"]
        del debate["_original_models"]
    _save_session(debate_id, debate)

    yield {"type": "round_end", "round": round_num, "round_cost": round_cost, "compression_cost": compression_cost, "total_cost": debate["total_cost"]}
    logger.info(f"Rodada {round_num} finalizada. Custo rodada: ${round_cost:.6f}, Total: ${debate['total_cost']:.6f}")


# ============================================================
# Modo Sobrevivência: Scoring + Eliminação
# ============================================================
async def _survival_scoring(round_results_map: dict, topic: str, round_num: int) -> tuple:
    """Pontua respostas e elimina 50% piores. Retorna (survivors, eliminated, cost, scores)."""
    from llm_providers import get_cheapest_configured_model as _gcm

    comp_model = _gcm()
    if not comp_model:
        # Sem modelo de scoring — fallback: manter todos
        all_ids = list(round_results_map.keys())
        return all_ids, [], 0.0, {}

    provider = get_provider_for_model(comp_model)
    if not provider:
        all_ids = list(round_results_map.keys())
        return all_ids, [], 0.0, {}

    # Montar texto das respostas para o juiz
    responses_text = ""
    model_names = {}
    for mid, data in round_results_map.items():
        name = data.get("name", get_model_display_name(mid))
        model_names[mid] = name
        content = data.get("content", "")[:1500]  # Limitar para caber no contexto
        if content and not content.startswith("[ERRO"):
            responses_text += f"\n\n### {name}\n{content}"

    if not responses_text.strip():
        all_ids = list(round_results_map.keys())
        return all_ids, [], 0.0, {}

    names_list = ", ".join(model_names.values())

    messages = [
        {"role": "system", "content": "Você é um juiz imparcial de torneio intelectual. Avalie com rigor. Responda APENAS com JSON válido."},
        {"role": "user", "content": (
            f"TORNEIO DE SOBREVIVÊNCIA — Rodada {round_num}\n"
            f"Tema: {topic}\n"
            f"Participantes: {names_list}\n\n"
            "Avalie CADA participante de 1 a 10 nos critérios:\n"
            "- Originalidade (ideias novas, perspectivas únicas)\n"
            "- Profundidade (análise completa, não superficial)\n"
            "- Argumentação (lógica, evidências, coerência)\n"
            "- Clareza (organização, legibilidade)\n"
            "- Relevância (aderência ao tema)\n\n"
            "Retorne JSON com esta estrutura EXATA:\n"
            "{\n"
            '  "scores": {\n'
            '    "Nome do Modelo": {"originalidade": 8, "profundidade": 7, "argumentacao": 9, "clareza": 8, "relevancia": 9, "total": 8.2, "justificativa": "1 frase"},\n'
            "  }\n"
            "}\n\n"
            "IMPORTANTE: 'total' é a média ponderada (originalidade 25%, profundidade 25%, argumentação 25%, clareza 15%, relevância 10%).\n"
            f"Inclua TODOS os participantes. Seja justo e rigoroso.\n\n"
            f"RESPOSTAS:{responses_text}"
        )},
    ]

    try:
        result = await asyncio.wait_for(
            provider.call(comp_model, messages, max_tokens=2000),
            timeout=60.0,
        )
        raw = result.get("content", "").strip()
        cost = result.get("cost", 0)

        # Parse JSON
        clean = raw
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
            if clean.endswith("```"): clean = clean[:-3]
            clean = clean.strip()
            if clean.startswith("json"): clean = clean[4:].strip()
        # Strip trailing commas before } or ] (common LLM mistake)
        clean = re.sub(r',\s*([}\]])', r'\1', clean)

        parsed = json.loads(clean)
        raw_scores = parsed.get("scores", {})

        # Mapear nomes de volta para model_ids e calcular ranking
        scored = []
        scores_output = {}
        for mid, name in model_names.items():
            # Tentar match por nome
            score_data = raw_scores.get(name)
            if not score_data:
                # Tentar match parcial
                for k, v in raw_scores.items():
                    if name.lower() in k.lower() or k.lower() in name.lower():
                        score_data = v
                        break
            if score_data and isinstance(score_data, dict):
                total = score_data.get("total", 5.0)
                scores_output[name] = {
                    "total": total,
                    "justificativa": score_data.get("justificativa", ""),
                    "detalhes": {k: v for k, v in score_data.items() if k not in ("total", "justificativa")},
                }
                scored.append((mid, total))
            else:
                scores_output[name] = {"total": 5.0, "justificativa": "Não avaliado"}
                scored.append((mid, 5.0))

        # Ordenar por score (maior primeiro)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Eliminar 50% piores (mínimo 1 sobrevivente)
        n_survivors = max(1, len(scored) // 2)
        # Se tem número ímpar, arredondar para cima (mais sobrevivem)
        if len(scored) % 2 == 1:
            n_survivors = (len(scored) + 1) // 2

        survivors = [mid for mid, _ in scored[:n_survivors]]
        eliminated = [mid for mid, _ in scored[n_survivors:]]

        return survivors, eliminated, cost, scores_output

    except (json.JSONDecodeError, asyncio.TimeoutError) as e:
        logger.error(f"Survival scoring falhou: {e}")
        # Fallback: manter todos
        all_ids = list(round_results_map.keys())
        return all_ids, [], 0.0, {}
    except Exception as e:
        logger.error(f"Survival scoring erro: {e}")
        all_ids = list(round_results_map.keys())
        return all_ids, [], 0.0, {}


# ============================================================
# Streaming model call with retry
# ============================================================
def _build_messages(system_prompt, history, summary, round_num, mode, display_name, persona=None, documents=None, evidence_pack=None):
    # Injetar persona no system prompt se configurada
    full_prompt = system_prompt
    if persona:
        logger.debug(f"Persona para {display_name}: '{persona}' (in PROMPTS: {persona in PERSONA_PROMPTS}, in CATALOG: {persona in PERSONA_CATALOG})")
        if persona in PERSONA_PROMPTS:
            full_prompt = PERSONA_PROMPTS[persona] + "\n\n" + system_prompt
        elif persona in PERSONA_CATALOG:
            cat = PERSONA_CATALOG[persona]
            full_prompt = f"PERSONA: {cat['name']}. {cat['prompt']}\n\n" + system_prompt
        elif len(persona) > 3:
            # Custom persona text
            full_prompt = f"PERSONA: {persona}\n\n" + system_prompt
    messages = [{"role": "system", "content": full_prompt}]

    # Injetar documentos do usuário como contexto
    if documents:
        doc_text = "[DOCUMENTOS DO USUÁRIO]\n" + "\n---\n".join(
            f"Arquivo: {d['filename']}\n{d['content']}" for d in documents
        )
        messages.append({"role": "user", "content": doc_text})

    context_parts = []

    # Injetar evidências da pesquisa pré-sessão
    if evidence_pack:
        context_summary = evidence_pack.get("context_summary", "")
        key_facts = evidence_pack.get("key_facts", [])
        sources = evidence_pack.get("sources", [])
        evidence_text = "[EVIDÊNCIAS PESQUISADAS]\n"
        if context_summary:
            evidence_text += f"Contexto: {context_summary}\n\n"
        if key_facts:
            evidence_text += "Fatos-chave:\n" + "\n".join(f"- {f}" for f in key_facts) + "\n\n"
        if sources:
            evidence_text += "Fontes:\n" + "\n".join(f"- {s.get('title','?')} ({s.get('url','?')})" for s in sources[:5])
        context_parts.insert(0, evidence_text)
    if summary:
        context_parts.append(f"Resumo do debate até agora:\n{summary}")
    if history:
        # Dynamic window: keep reducing until under ~30K chars (~7500 tokens)
        max_chars = 30000
        window = min(20, len(history))
        while window > 4:
            recent = history[-window:]
            total_chars = sum(len(h.get("content", "")) for h in recent)
            if total_chars <= max_chars:
                break
            window -= 2
        recent = history[-max(4, window):]
        hist_text = "\n\n".join(f"[{h.get('name', '?')}]: {h.get('content', '')}" for h in recent)
        context_parts.append(f"Mensagens recentes:\n{hist_text}")

    if context_parts:
        context = "\n\n---\n\n".join(context_parts)
        if mode == "debate" or mode == "devil_advocate":
            instruction = "Responda aos argumentos e defenda sua posição."
        else:
            instruction = "Construa sobre as ideias apresentadas. Adicione algo novo."
        messages.append({"role": "user", "content": f"{context}\n\nAgora é sua vez ({display_name}), rodada {round_num}. {instruction}"})
    else:
        if mode in ("debate", "devil_advocate"):
            starter = f"Apresente sua posição inicial sobre o tema. Você é {display_name}."
        else:
            starter = f"Contribua com suas primeiras ideias sobre o tema. Você é {display_name}."
        messages.append({"role": "user", "content": f"Rodada {round_num}. {starter}"})
    return messages


async def _call_model_streaming(semaphore, model_id, system_prompt, history, summary, round_num, mode, max_tokens, token_callback, persona=None, documents=None, evidence_pack=None):
    """Chama modelo com streaming + margem 30% + auto-continue se truncar."""
    async with semaphore:
        provider = get_provider_for_model(model_id)
        if not provider:
            raise Exception(f"Provedor não encontrado para {model_id}")
        if not provider.is_configured():
            raise Exception(f"API key não configurada para {provider.name}")

        display_name = get_model_display_name(model_id)
        messages = _build_messages(system_prompt, history, summary, round_num, mode, display_name, persona=persona, documents=documents, evidence_pack=evidence_pack)

        # Check cache
        cache_key = _cache_key(model_id, messages)
        cached = _cache_get(cache_key)
        if cached:
            await token_callback(model_id, display_name, cached.get("content", ""))
            cached["cost"] = 0  # No cost for cached
            cached["cached"] = True
            return cached

        # Margem de 30% para evitar truncamento
        api_max_tokens = int(max_tokens * 1.3)

        # ITEM 13: Metrics tracking
        provider_name = provider.name
        if provider_name not in provider_metrics:
            provider_metrics[provider_name] = {"total_calls": 0, "errors": 0, "total_latency_ms": 0, "total_tokens": 0, "total_cost": 0.0, "truncations": 0}
        start_time = time.time()

        last_err = None
        for attempt in range(2):
            try:
                result_data = None
                async for chunk in provider.stream(model_id, messages, api_max_tokens):
                    if chunk["type"] == "token":
                        await token_callback(model_id, display_name, chunk["content"])
                    elif chunk["type"] == "done":
                        result_data = chunk

                if not result_data:
                    result_data = {"content": "", "usage": {}, "finish_reason": "stop", "cost": 0}

                # ── Auto-continue: se truncou, pedir continuação (máx 1x) ──
                if result_data.get("finish_reason") == "length" and len(result_data.get("content", "")) > 50:
                    logger.info(f"[R{round_num}] {display_name}: Truncado, tentando auto-continue...")
                    partial = result_data["content"]
                    last_200 = partial[-200:]

                    continue_msgs = messages + [
                        {"role": "assistant", "content": partial},
                        {"role": "user", "content": f"Sua resposta foi cortada. Continue EXATAMENTE de onde parou (após: '...{last_200[-80:]}').\nNão repita o que já disse. Apenas conclua o raciocínio."},
                    ]

                    continue_tokens = max(500, int(max_tokens * 0.5))  # metade do original para a continuação
                    continuation = ""
                    continue_cost = 0

                    try:
                        async for chunk in provider.stream(model_id, continue_msgs, continue_tokens):
                            if chunk["type"] == "token":
                                token_text = chunk["content"]
                                if not isinstance(token_text, str):
                                    token_text = str(token_text) if token_text is not None else ""
                                continuation += token_text
                                await token_callback(model_id, display_name, token_text)
                            elif chunk["type"] == "done":
                                continue_cost = chunk.get("cost", 0)
                                # Merge usage
                                cu = chunk.get("usage", {})
                                ou = result_data.get("usage", {})
                                result_data["usage"] = {
                                    "prompt_tokens": ou.get("prompt_tokens", 0) + cu.get("prompt_tokens", 0),
                                    "completion_tokens": ou.get("completion_tokens", 0) + cu.get("completion_tokens", 0),
                                }

                        if continuation:
                            result_data["content"] = partial + continuation
                            result_data["cost"] = result_data.get("cost", 0) + continue_cost
                            result_data["finish_reason"] = "stop"  # Agora está completo
                            logger.info(f"[R{round_num}] {display_name}: Auto-continue OK (+{len(continuation)} chars)")
                    except Exception as ce:
                        logger.warning(f"[R{round_num}] {display_name}: Auto-continue falhou: {ce}")
                        # Manter resposta truncada original — melhor que nada

                # ITEM 13: Record metrics on success
                latency_ms = int((time.time() - start_time) * 1000)
                m = provider_metrics[provider_name]
                m["total_calls"] += 1
                m["total_latency_ms"] += latency_ms
                if result_data:
                    m["total_tokens"] += result_data.get("usage", {}).get("completion_tokens", 0)
                    m["total_cost"] += result_data.get("cost", 0)
                    if result_data.get("finish_reason") == "length":
                        m["truncations"] += 1
                    result_data["latency_ms"] = latency_ms

                # Cache successful results
                if result_data and result_data.get("content"):
                    _cache_set(cache_key, result_data)

                return result_data

            except Exception as e:
                last_err = e
                err_str = str(e)
                if any(code in err_str for code in ["401", "403"]):
                    # ITEM 13: Record error metrics
                    latency_ms = int((time.time() - start_time) * 1000)
                    m = provider_metrics[provider_name]
                    m["total_calls"] += 1
                    m["total_latency_ms"] += latency_ms
                    m["errors"] += 1
                    break
                if attempt == 0:
                    logger.info(f"[R{round_num}] {display_name}: Retry em 2s (erro: {err_str[:100]})")
                    await asyncio.sleep(2)
                    continue
                # ITEM 13: Record error metrics on final attempt
                latency_ms = int((time.time() - start_time) * 1000)
                m = provider_metrics[provider_name]
                m["total_calls"] += 1
                m["total_latency_ms"] += latency_ms
                m["errors"] += 1
                break
        raise last_err or Exception(f"Modelo {model_id} falhou sem erro capturado")


# ============================================================
# TASK 1.3: Compressor com fallback robusto (tenta 2 modelos)
# ============================================================
async def _compress_context(history: list, max_tokens: int, num_models: int = 1, mode: str = "debate", round_num: int = 2) -> tuple[str | None, float, dict | None]:
    """Retorna (summary, cost, xray_data). Seleciona compressor dinamicamente."""
    from llm_providers import blacklist_compression_model, get_best_compression_model

    for attempt in range(2):
        comp_model = get_best_compression_model(
            mode=mode, num_models=num_models, round_num=round_num, max_tokens=max_tokens
        )
        if not comp_model:
            logger.warning("Compressão: nenhum modelo configurado disponível")
            return None, 0.0, None

        from llm_providers import _classify_situation
        situation = _classify_situation(mode, num_models, round_num, max_tokens)
        logger.info(f"Compressor: {get_model_display_name(comp_model)} [{situation}] (modo={mode}, modelos={num_models}, rodada={round_num}, tk={max_tokens})")

        provider = get_provider_for_model(comp_model)
        if not provider:
            return None, 0.0, None

        cap = get_compression_cap(max_tokens, num_models)
        bullets = get_compression_bullets(max_tokens, num_models)

        # Extrair nomes dos modelos para o prompt de scoring
        model_names = list({h.get("name", "?") for h in history if h.get("name")})
        model_names_str = ", ".join(model_names) if model_names else "modelos participantes"

        hist_text = "\n\n".join(f"[{h.get('name', '?')}]: {h.get('content', '')}" for h in history)

        messages = [
            {"role": "system", "content": "Você é um analista de debates. Responda APENAS com JSON válido, sem markdown, sem ```json, sem texto extra."},
            {"role": "user", "content": (
                f"Analise a conversa abaixo e retorne UM ÚNICO objeto JSON com esta estrutura exata:\n\n"
                "{\n"
                f'  "consenso": "Principal ponto de acordo entre os modelos (máx 25 palavras)",\n'
                f'  "divergencia": "Principal ponto de conflito ou desacordo (máx 25 palavras)",\n'
                f'  "modelo_destaque": "nome_do_modelo_com_melhor_argumento_nesta_rodada",\n'
                f'  "model_scores": {{"modelo_a": 8.5, "modelo_b": 7.2}},\n'
                f'  "contexto_para_proxima_rodada": "Resumo condensado em {bullets} bullet points máximo"\n'
                "}\n\n"
                f"Modelos participantes: {model_names_str}\n"
                "Regras para model_scores: nota de 1 a 10 para cada modelo baseada em originalidade, profundidade e relevância.\n"
                "Regras para contexto: mantenha apenas posições ativas, descarte refutadas, preserve dados factuais.\n\n"
                f"CONVERSA:\n{hist_text}"
            )},
        ]

        try:
            result = await asyncio.wait_for(
                provider.call(comp_model, messages, max_tokens=min(cap, 4096)),
                timeout=45.0  # timeout generoso para modelos lentos com contexto grande
            )
            raw = result["content"].strip()
            cost = result.get("cost", 0)

            # Tentar parsear JSON
            xray = None
            summary = raw
            try:
                # Limpar possíveis wrappers markdown
                clean = raw
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                    clean = clean.strip()
                    if clean.startswith("json"):
                        clean = clean[4:].strip()
                # Strip trailing commas before } or ] (common LLM mistake)
                clean = re.sub(r',\s*([}\]])', r'\1', clean)

                parsed = json.loads(clean)
                if isinstance(parsed, dict) and "contexto_para_proxima_rodada" in parsed:
                    summary = parsed["contexto_para_proxima_rodada"]
                    xray = {
                        "consenso": parsed.get("consenso", "Não identificado"),
                        "divergencia": parsed.get("divergencia", "Não identificado"),
                        "modelo_destaque": parsed.get("modelo_destaque", ""),
                        "model_scores": parsed.get("model_scores", {}),
                    }
                    logger.info(f"Raio-X: consenso='{xray['consenso'][:50]}', MVP={xray['modelo_destaque']}")
                else:
                    logger.warning("Compressor retornou JSON sem campo contexto_para_proxima_rodada")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Compressor retornou texto não-JSON, usando como contexto bruto: {e}")
                # summary já é o raw, xray fica None

            return summary, cost, xray
        except asyncio.TimeoutError:
            logger.error(f"Compressão timeout com {comp_model} (tentativa {attempt + 1})")
            blacklist_compression_model(comp_model)
            continue
        except Exception as e:
            logger.error(f"Compressão falhou com {comp_model} (tentativa {attempt + 1}): {e}")
            blacklist_compression_model(comp_model)
            continue

    return None, 0.0, None


def _log_error(round_num, display_name, model_id, err_msg):
    with _state_lock:
        error_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "round": round_num,
            "model": display_name,
            "model_id": model_id,
            "error": err_msg,
        })
        if len(error_log) > 500:
            error_log[:] = error_log[-200:]


# ============================================================
# Podcast / TTS
# ============================================================
PODCAST_DIR = os.path.join(os.path.dirname(__file__), "podcasts")
os.makedirs(PODCAST_DIR, exist_ok=True)


@app.route("/api/debate/<debate_id>/podcast", methods=["POST"])
def generate_podcast(debate_id):
    """Gera script de podcast e áudio via OpenAI TTS."""
    debate = active_debates.get(debate_id)
    if not debate:
        safe_id = os.path.basename(debate_id)
        path = os.path.join(SESSIONS_DIR, f"{safe_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                debate = json.load(f)
    if not debate:
        return jsonify({"error": "Debate não encontrado"}), 404

    history = debate.get("conversation_history", [])
    if not history:
        return jsonify({"error": "Sem histórico"}), 400

    topic = debate.get("topic", "")
    model_names = list(dict.fromkeys(h.get("name", "?") for h in history if h.get("name")))
    hist_text = "\n\n".join(f"[{h.get('name', '?')}]: {h.get('content', '')[:800]}" for h in history[-20:])

    # 1) Gerar script
    script_model = get_cheapest_configured_model()
    if not script_model:
        return jsonify({"error": "Nenhum modelo disponível"}), 500
    provider = get_provider_for_model(script_model)
    if not provider:
        return jsonify({"error": "Provider não encontrado"}), 500

    messages = [
        {"role": "system", "content": "Você converte debates em scripts de podcast natural em português brasileiro."},
        {"role": "user", "content": (
            f"Converta este debate em script de podcast de 3-5 minutos.\n"
            f"Tema: {topic}\nParticipantes: {', '.join(model_names)}\n\n"
            "Regras: NARRADOR abre e fecha. Cada participante fala naturalmente. Máx 2000 palavras.\n"
            "Formato (uma linha por fala):\nNARRADOR: texto\n"
            f"{model_names[0] if model_names else 'A'}: texto\n\n"
            f"DEBATE:\n{hist_text}"
        )},
    ]

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(asyncio.wait_for(
            provider.call(script_model, messages, max_tokens=3000), timeout=60.0
        ))
        script = result.get("content", "")
        script_cost = result.get("cost", 0)
    except Exception as e:
        return jsonify({"error": f"Erro ao gerar script: {str(e)[:200]}"}), 500
    finally:
        loop.close()

    # 2) TTS via OpenAI (se disponível)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        return jsonify({"script": script, "script_cost": script_cost, "audio_url": None,
                        "message": "Script gerado. OPENAI_API_KEY necessária para áudio."})

    voice_map = {"NARRADOR": "nova"}
    voices = ["alloy", "echo", "fable", "onyx", "shimmer"]
    for i, name in enumerate(model_names):
        voice_map[name] = voices[i % len(voices)]

    segments = []
    for line in script.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        speaker, text = speaker.strip(), text.strip()
        if len(text) < 3:
            continue
        segments.append({"speaker": speaker, "text": text[:500], "voice": voice_map.get(speaker, "alloy")})
    segments = segments[:20]

    if not segments:
        return jsonify({"script": script, "script_cost": script_cost, "audio_url": None, "message": "Script vazio"})

    loop2 = asyncio.new_event_loop()
    try:
        async def tts_one(seg):
            async with aiohttp.ClientSession() as s:
                async with s.post("https://api.openai.com/v1/audio/speech",
                    headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                    json={"model": "tts-1", "input": seg["text"], "voice": seg["voice"], "response_format": "mp3"},
                    timeout=aiohttp.ClientTimeout(total=30)) as r:
                    return await r.read() if r.status == 200 else None

        async def tts_all():
            return [await tts_one(s) for s in segments]

        chunks = loop2.run_until_complete(tts_all())
        chunks = [c for c in chunks if c]
    except Exception as e:
        return jsonify({"script": script, "script_cost": script_cost, "audio_url": None,
                        "message": f"Erro TTS: {str(e)[:100]}"})
    finally:
        loop2.close()

    if not chunks:
        return jsonify({"script": script, "script_cost": script_cost, "audio_url": None, "message": "TTS falhou"})

    combined = b"".join(chunks)
    podcast_path = os.path.join(PODCAST_DIR, f"{debate_id}.mp3")
    with open(podcast_path, "wb") as f:
        f.write(combined)

    total_chars = sum(len(s["text"]) for s in segments)
    tts_cost = (total_chars / 1_000_000) * 15.0
    total_cost = script_cost + tts_cost

    logger.info(f"Podcast: {len(segments)} segs, {len(combined)} bytes, ${total_cost:.4f}")
    return jsonify({"script": script, "script_cost": script_cost, "tts_cost": tts_cost,
                    "total_cost": total_cost, "segments": len(segments),
                    "audio_url": f"/api/debate/{debate_id}/podcast/audio",
                    "duration_estimate": f"~{len(segments) * 8}s"})


@app.route("/api/debate/<debate_id>/podcast/audio")
def serve_podcast(debate_id):
    safe_id = os.path.basename(debate_id)
    path = os.path.join(PODCAST_DIR, f"{safe_id}.mp3")
    if not os.path.exists(path):
        return jsonify({"error": "Podcast não encontrado"}), 404
    return send_file(path, mimetype="audio/mpeg", as_attachment=False, download_name=f"colosseum_{safe_id}.mp3")


# ============================================================
# API Documentation
# ============================================================
@app.route("/api")
def api_docs():
    """Auto-generated API documentation."""
    routes = []
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        if not rule.rule.startswith("/api"):
            continue
        methods = sorted(rule.methods - {"HEAD", "OPTIONS"})
        func = app.view_functions.get(rule.endpoint)
        doc = (func.__doc__ or "").strip() if func else ""
        routes.append({
            "path": rule.rule,
            "methods": methods,
            "description": doc,
        })

    return jsonify({
        "name": "Synapse Colosseum API",
        "version": "3.0",
        "total_routes": len(routes),
        "total_models": len(get_available_models()),
        "total_providers": len([p for p in ALL_PROVIDERS if p.is_configured()]),
        "routes": routes,
    })


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n=== Synapse Colosseum ===")
    print("Acesse: http://localhost:5000")
    print("Ctrl+C para parar\n")
    app.run(debug=True, port=5000, host="0.0.0.0", threaded=True)