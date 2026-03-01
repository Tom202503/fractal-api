#!/usr/bin/env python3
"""
CPFS Multi-LLM Router
Байєсівський роутер між Claude · GPT-4 · Gemini · Ollama · NanoAI

Використання:
  python3 cpfs_router.py                    # інтерактивний чат
  python3 cpfs_router.py --status           # стан агентів
  python3 cpfs_router.py --model claude     # примусово обрати модель
  python3 cpfs_router.py --ask "питання"   # одне питання і вийти

Налаштування:
  Створи .env файл поряд зі скриптом:
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    GEMINI_API_KEY=...
    NANO_AI_API_KEY=...
    OLLAMA_URL=http://localhost:11434
    OLLAMA_MODEL=llama3
"""

from __future__ import annotations
import asyncio, json, os, random, re, sys, time, urllib.request, urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════════
# ENV LOADER  (.env поряд зі скриптом)
# ════════════════════════════════════════════════════════════

def load_env():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

load_env()

# ════════════════════════════════════════════════════════════
# CPFS CORE ENGINE
# ════════════════════════════════════════════════════════════

@dataclass
class StateDistribution:
    agent_id:        str
    mean:            float          # поточний стан [0..1]
    variance:        float          # невизначеність
    stability_score: float          # стабільність агента
    capabilities:    List[str]      # що вміє
    latency_ms:      float = 500.0  # середня затримка
    cost_per_1k:     float = 1.0    # відносна вартість
    available:       bool  = True   # чи відповідає API

    def liquidity(self) -> float:
        """L_I = adaptivity / resource"""
        if not self.available:
            return 0.0
        adaptivity = self.stability_score * (1.0 - self.variance)
        resource   = max(0.1, self.latency_ms / 1000.0) * max(0.1, self.cost_per_1k)
        return adaptivity / resource


class MarkovLLMAgent:
    """Кожна LLM — марківський агент з власним фазовим станом"""

    def __init__(self, agent_id: str, name: str, capabilities: List[str],
                 cost_per_1k: float = 1.0, eta: float = 0.1):
        self.agent_id     = agent_id
        self.name         = name
        self.capabilities = capabilities
        self.cost_per_1k  = cost_per_1k
        self.eta          = eta
        # фазовий стан
        self.mean            = 0.6
        self.variance        = 0.15
        self.stability_score = 0.5
        self.latency_ms      = 1000.0
        self.available       = False   # перевіряється при старті
        self.call_count      = 0
        self.error_count     = 0
        self.total_tokens    = 0

    async def call(self, messages: List[Dict], **kwargs) -> Tuple[str, int]:
        """Повертає (text, tokens). Override у підкласах."""
        raise NotImplementedError

    def endomorphism(self, success: bool, latency_ms: float) -> None:
        """A_{t+1} = A_t + η∇L"""
        feedback = 0.9 if success else 0.1
        grad = feedback - self.stability_score
        self.stability_score = min(1.0, max(0.0, self.stability_score + self.eta * grad))
        # оновлюємо латентність через EWMA
        self.latency_ms = 0.7 * self.latency_ms + 0.3 * latency_ms
        # стохастичний перехід стану
        xi = random.gauss(0, 1) * 0.1
        F  = -0.2 * (self.mean - 0.5)
        self.mean     = min(1.0, max(0.0, self.mean + F * 0.1 + xi * 0.1))
        self.variance = max(0.01, self.variance * (0.95 if success else 1.1))
        # скидаємо лічильник помилок при успіху; вимикаємо після 3 поспіль
        if success:
            self.error_count = 0
            self.available = True
        else:
            self.available = self.error_count < 3

    def dist(self) -> StateDistribution:
        return StateDistribution(
            agent_id=self.agent_id, mean=self.mean, variance=self.variance,
            stability_score=self.stability_score, capabilities=self.capabilities,
            latency_ms=self.latency_ms, cost_per_1k=self.cost_per_1k,
            available=self.available,
        )

    def stats(self) -> Dict:
        return {
            "name": self.name, "available": self.available,
            "stability": round(self.stability_score, 3),
            "latency_ms": round(self.latency_ms),
            "calls": self.call_count, "errors": self.error_count,
            "tokens": self.total_tokens,
            "liquidity": round(self.dist().liquidity(), 3),
        }


# ════════════════════════════════════════════════════════════
# LLM AGENTS
# ════════════════════════════════════════════════════════════

def _post_json(url: str, headers: Dict, body: Dict, timeout: int = 30) -> Dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class ClaudeAgent(MarkovLLMAgent):
    def __init__(self):
        super().__init__("claude", "Claude (Anthropic)",
                         ["reasoning", "code", "analysis", "writing", "long_context"],
                         cost_per_1k=0.015, eta=0.08)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model   = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")

    async def check(self) -> bool:
        self.available = bool(self.api_key and not self.api_key.startswith("YOUR_"))
        if self.available: self.stability_score = 0.7
        return self.available

    async def call(self, messages: List[Dict], **kwargs) -> Tuple[str, int]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY не встановлено")
        loop = asyncio.get_running_loop()
        t0   = time.time()
        # відокремлюємо system prompt від messages (Anthropic API вимагає окреме поле)
        system_text = None
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                api_messages.append(m)
        body: Dict = {"model": self.model, "max_tokens": kwargs.get("max_tokens", 2048),
                      "messages": api_messages}
        if system_text:
            body["system"] = system_text
        try:
            resp = await loop.run_in_executor(None, lambda: _post_json(
                "https://api.anthropic.com/v1/messages",
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
                body
            ))
            text   = resp["content"][0]["text"]
            tokens = resp.get("usage", {}).get("output_tokens", 0)
            latency = (time.time() - t0) * 1000
            self.call_count += 1; self.total_tokens += tokens
            self.endomorphism(True, latency)
            return text, tokens
        except Exception as e:
            self.error_count += 1
            self.endomorphism(False, 5000)
            raise RuntimeError(f"Claude error: {e}")


class GPTAgent(MarkovLLMAgent):
    def __init__(self):
        super().__init__("gpt4", "GPT-4 (OpenAI)",
                         ["reasoning", "code", "analysis", "writing", "math"],
                         cost_per_1k=0.030, eta=0.08)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model   = os.environ.get("OPENAI_MODEL", "gpt-4o")

    async def check(self) -> bool:
        self.available = bool(self.api_key and not self.api_key.startswith("YOUR_"))
        if self.available: self.stability_score = 0.7
        return self.available

    async def call(self, messages: List[Dict], **kwargs) -> Tuple[str, int]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY не встановлено")
        loop = asyncio.get_running_loop()
        t0   = time.time()
        try:
            resp = await loop.run_in_executor(None, lambda: _post_json(
                "https://api.openai.com/v1/chat/completions",
                {"Authorization": f"Bearer {self.api_key}",
                 "Content-Type": "application/json"},
                {"model": self.model, "messages": messages,
                 "max_tokens": kwargs.get("max_tokens", 2048)}
            ))
            text   = resp["choices"][0]["message"]["content"]
            tokens = resp.get("usage", {}).get("completion_tokens", 0)
            latency = (time.time() - t0) * 1000
            self.call_count += 1; self.total_tokens += tokens
            self.endomorphism(True, latency)
            return text, tokens
        except Exception as e:
            self.error_count += 1
            self.endomorphism(False, 5000)
            raise RuntimeError(f"GPT-4 error: {e}")


class GeminiAgent(MarkovLLMAgent):
    def __init__(self):
        super().__init__("gemini", "Gemini (Google)",
                         ["reasoning", "multimodal", "code", "analysis", "writing"],
                         cost_per_1k=0.007, eta=0.08)
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model   = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")

    async def check(self) -> bool:
        self.available = bool(self.api_key and not self.api_key.startswith("YOUR_"))
        if self.available: self.stability_score = 0.7
        return self.available

    async def call(self, messages: List[Dict], **kwargs) -> Tuple[str, int]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY не встановлено")
        loop = asyncio.get_running_loop()
        t0   = time.time()
        # конвертуємо OpenAI формат → Gemini формат (пропускаємо system messages)
        contents = [{"role": "user" if m["role"] == "user" else "model",
                     "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent")
        try:
            resp = await loop.run_in_executor(None, lambda: _post_json(
                url, {"Content-Type": "application/json",
                      "x-goog-api-key": self.api_key},
                {"contents": contents,
                 "generationConfig": {"maxOutputTokens": kwargs.get("max_tokens", 2048)}}
            ))
            text   = resp["candidates"][0]["content"]["parts"][0]["text"]
            tokens = resp.get("usageMetadata", {}).get("candidatesTokenCount", 0)
            latency = (time.time() - t0) * 1000
            self.call_count += 1; self.total_tokens += tokens
            self.endomorphism(True, latency)
            return text, tokens
        except Exception as e:
            self.error_count += 1
            self.endomorphism(False, 5000)
            raise RuntimeError(f"Gemini error: {e}")


class OllamaAgent(MarkovLLMAgent):
    def __init__(self):
        super().__init__("ollama", "Ollama (локально)",
                         ["reasoning", "code", "writing", "privacy", "offline"],
                         cost_per_1k=0.0, eta=0.12)
        self.base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.model    = os.environ.get("OLLAMA_MODEL", "llama3")

    async def check(self) -> bool:
        loop = asyncio.get_running_loop()
        try:
            def _check():
                req = urllib.request.Request(f"{self.base_url}/api/tags")
                with urllib.request.urlopen(req, timeout=3) as r:
                    return json.loads(r.read())
            data = await loop.run_in_executor(None, _check)
            models = [m["name"] for m in data.get("models", [])]
            self.available = len(models) > 0
            if self.available:
                # обираємо перший доступний якщо вказаного немає
                if not any(self.model in m for m in models):
                    self.model = models[0].split(":")[0]
                self.stability_score = 0.8  # локально = стабільно
        except Exception:
            self.available = False
        return self.available

    async def call(self, messages: List[Dict], **kwargs) -> Tuple[str, int]:
        loop = asyncio.get_running_loop()
        t0   = time.time()
        try:
            resp = await loop.run_in_executor(None, lambda: _post_json(
                f"{self.base_url}/api/chat",
                {"Content-Type": "application/json"},
                {"model": self.model, "messages": messages, "stream": False}
            ))
            text   = resp["message"]["content"]
            tokens = resp.get("eval_count", 0)
            latency = (time.time() - t0) * 1000
            self.call_count += 1; self.total_tokens += tokens
            self.endomorphism(True, latency)
            return text, tokens
        except Exception as e:
            self.error_count += 1
            self.endomorphism(False, 5000)
            raise RuntimeError(f"Ollama error: {e}")


class NanoAIAgent(MarkovLLMAgent):
    def __init__(self):
        super().__init__("nanoai", "Nano AI",
                         ["reasoning", "code", "writing", "analysis"],
                         cost_per_1k=0.001, eta=0.1)
        self.api_key  = os.environ.get("NANO_AI_API_KEY", "")
        self.base_url = os.environ.get("NANO_AI_URL", "https://api.nano-gpt.com/v1")
        self.model    = os.environ.get("NANO_AI_MODEL", "gpt-4o-mini")

    async def check(self) -> bool:
        self.available = bool(self.api_key and not self.api_key.startswith("YOUR_"))
        if self.available: self.stability_score = 0.65
        return self.available

    async def call(self, messages: List[Dict], **kwargs) -> Tuple[str, int]:
        if not self.api_key:
            raise RuntimeError("NANO_AI_API_KEY не встановлено")
        loop = asyncio.get_running_loop()
        t0   = time.time()
        try:
            resp = await loop.run_in_executor(None, lambda: _post_json(
                f"{self.base_url}/chat/completions",
                {"Authorization": f"Bearer {self.api_key}",
                 "Content-Type": "application/json"},
                {"model": self.model, "messages": messages,
                 "max_tokens": kwargs.get("max_tokens", 2048)}
            ))
            text   = resp["choices"][0]["message"]["content"]
            tokens = resp.get("usage", {}).get("completion_tokens", 0)
            latency = (time.time() - t0) * 1000
            self.call_count += 1; self.total_tokens += tokens
            self.endomorphism(True, latency)
            return text, tokens
        except Exception as e:
            self.error_count += 1
            self.endomorphism(False, 5000)
            raise RuntimeError(f"NanoAI error: {e}")


# ════════════════════════════════════════════════════════════
# BAYESIAN ROUTER
# ════════════════════════════════════════════════════════════

# Яка модель краща для якого типу задачі
TASK_AFFINITY: Dict[str, Dict[str, float]] = {
    "code":        {"claude": 1.0, "gpt4": 0.95, "gemini": 0.8, "ollama": 0.7, "nanoai": 0.6},
    "reasoning":   {"claude": 1.0, "gpt4": 0.9,  "gemini": 0.85,"ollama": 0.65,"nanoai": 0.55},
    "writing":     {"claude": 1.0, "gpt4": 0.85, "gemini": 0.8, "ollama": 0.7, "nanoai": 0.65},
    "math":        {"gpt4": 1.0,   "claude": 0.9, "gemini": 0.85,"ollama": 0.6, "nanoai": 0.5},
    "multimodal":  {"gemini": 1.0, "gpt4": 0.9,  "claude": 0.7, "ollama": 0.3, "nanoai": 0.4},
    "privacy":     {"ollama": 1.0, "claude": 0.3, "gpt4": 0.3,  "gemini": 0.3, "nanoai": 0.5},
    "fast":        {"nanoai": 1.0, "gemini": 0.9, "ollama": 0.85,"gpt4": 0.6,  "claude": 0.6},
    "cheap":       {"ollama": 1.0, "nanoai": 0.95,"gemini": 0.8, "claude": 0.4, "gpt4": 0.3},
    "analysis":    {"claude": 1.0, "gpt4": 0.9,  "gemini": 0.85,"ollama": 0.65,"nanoai": 0.6},
}

TASK_KEYWORDS: Dict[str, List[str]] = {
    "code":      ["код", "code", "python", "javascript", "function", "debug", "script",
                  "програм", "реалізуй", "implement"],
    "math":      ["математик", "рівнян", "обчисл", "calculate", "math", "formula",
                  "integral", "derivative", "число"],
    "multimodal":["зображ", "image", "фото", "photo", "картин", "picture", "visual"],
    "privacy":   ["приватн", "локальн", "конфіденц", "private", "local", "offline",
                  "secret", "sensitive"],
    "fast":      ["швидко", "fast", "quick", "терміново", "urgent", "asap"],
    "cheap":     ["дешево", "cheap", "безкоштовно", "free", "економ", "budget"],
    "writing":   ["напиш", "write", "текст", "text", "стаття", "article", "лист", "email"],
    "analysis":  ["аналіз", "analys", "порівняй", "compare", "evaluate", "оціни"],
}

def _is_ascii_keyword(kw: str) -> bool:
    return all(ord(ch) < 128 for ch in kw)


def detect_task_type(prompt: str) -> str:
    prompt_lower = prompt.lower()
    scores: Dict[str, int] = {}
    for task, keywords in TASK_KEYWORDS.items():
        count = 0
        for kw in keywords:
            if _is_ascii_keyword(kw):
                # ASCII keywords — word boundary to avoid "local" in "localhost"
                if re.search(r'(?:^|\W)' + re.escape(kw) + r'(?:$|\W)', prompt_lower):
                    count += 1
            else:
                # Cyrillic keywords — prefix/substring match ("зображ" → "зображення")
                if kw in prompt_lower:
                    count += 1
        scores[task] = count
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else "reasoning"


class BayesianRouter:
    """P(A|D) ∝ P(D|A) · P(A) — вибір агента через байєс"""

    def __init__(self, agents: List[MarkovLLMAgent]):
        self.agents = {a.agent_id: a for a in agents}

    def route(self, prompt: str, force: Optional[str] = None) -> Optional[MarkovLLMAgent]:
        available = [a for a in self.agents.values() if a.available]
        if not available:
            return None

        if force:
            agent = self.agents.get(force)
            if agent and agent.available:
                return agent
            print(f"  ⚠ Модель '{force}' недоступна, обираємо автоматично")

        task = detect_task_type(prompt)
        affinity = TASK_AFFINITY.get(task, {})

        scores: Dict[str, float] = {}
        for agent in available:
            d    = agent.dist()
            li   = d.liquidity()                           # L_I
            af   = affinity.get(agent.agent_id, 0.5)       # task affinity
            # P(A|D) ∝ P(D|A) · P(A)
            score = li * af * (1.0 - d.variance)
            scores[agent.agent_id] = score

        best_id = max(scores, key=lambda k: scores[k])
        return self.agents[best_id]

    def attractor(self) -> Dict:
        """Мультимодальний атрактор T'"""
        dists = [a.dist() for a in self.agents.values()]
        if not dists:
            return {}
        available = [d for d in dists if d.available]
        if not available:
            return {"stability": 0.0, "available_agents": 0}
        tw   = sum(d.stability_score for d in available) or 1.0
        mean = sum(d.mean * d.stability_score for d in available) / tw
        var  = sum(d.variance * d.stability_score for d in available) / tw
        return {
            "mean": round(mean, 3), "variance": round(var, 3),
            "stability": round(1.0 - var, 3),
            "available_agents": len(available),
            "total_agents": len(dists),
            "liquidity_index": round(sum(d.liquidity() for d in available) / len(available), 3),
        }


# ════════════════════════════════════════════════════════════
# TERMINAL UI
# ════════════════════════════════════════════════════════════

COLORS = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "blue":   "\033[94m",
    "purple": "\033[95m",
    "cyan":   "\033[96m",
    "red":    "\033[91m",
    "gray":   "\033[90m",
}

def c(color: str, text: str) -> str:
    return f"{COLORS.get(color,'')}{text}{COLORS['reset']}"

def print_banner():
    print(c("bold", c("cyan", """
╔═══════════════════════════════════════════╗
║   CPFS Multi-LLM Router  v1.0            ║
║   Claude · GPT-4 · Gemini · Ollama · Nano ║
╚═══════════════════════════════════════════╝""")))
    print(c("gray", "  Байєсівський роутер між нейромережами\n"))

def print_status(agents: List[MarkovLLMAgent], router: BayesianRouter):
    print(c("bold", "\n  Стан агентів:"))
    print(f"  {'Модель':<22} {'Статус':<12} {'Stability':<12} {'Latency':<12} {'L_I':<8} {'Calls'}")
    print("  " + "─" * 72)
    for agent in agents:
        s = agent.stats()
        status = c("green", "● доступний") if s["available"] else c("red", "○ недоступний")
        stab   = c("green", f"{s['stability']:.2f}") if s["stability"] > 0.6 else c("yellow", f"{s['stability']:.2f}")
        li     = c("cyan", f"{s['liquidity']:.3f}")
        print(f"  {agent.name:<22} {status:<20} {stab:<20} {s['latency_ms']:>6}ms     {li:<16} {s['calls']}")

    att = router.attractor()
    print(f"\n  {c('bold', 'Атрактор T'+'\'')}: "
          f"mean={c('cyan', str(att.get('mean', '—')))} "
          f"stability={c('green', str(att.get('stability', '—')))} "
          f"L_I={c('yellow', str(att.get('liquidity_index', '—')))} "
          f"агентів={att.get('available_agents', 0)}/{att.get('total_agents', 0)}")

def print_help():
    print(c("bold", "\n  Команди:"))
    cmds = [
        ("/status",          "стан всіх агентів і атрактора"),
        ("/model claude",    "примусово обрати Claude"),
        ("/model gpt4",      "примусово обрати GPT-4"),
        ("/model gemini",    "примусово обрати Gemini"),
        ("/model ollama",    "примусово обрати Ollama"),
        ("/model nanoai",    "примусово обрати Nano AI"),
        ("/model auto",      "повернути автоматичний вибір"),
        ("/history",         "показати історію розмови"),
        ("/clear",           "очистити історію"),
        ("/keys",            "як додати API ключі"),
        ("/help",            "ця підказка"),
        ("/exit або /q",     "вийти"),
    ]
    for cmd, desc in cmds:
        print(f"  {c('cyan', cmd):<30} {c('gray', desc)}")

def print_keys_help():
    env_path = Path(__file__).parent / ".env"
    print(c("bold", f"\n  Створи файл: {env_path}"))
    print(c("gray", "  Вміст файлу:\n"))
    print("""  ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
  OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
  GEMINI_API_KEY=xxxxxxxxxxxxxxxx
  NANO_AI_API_KEY=xxxxxxxxxxxxxxxx
  OLLAMA_URL=http://localhost:11434
  OLLAMA_MODEL=llama3

  Де отримати ключі:
  Claude  → https://console.anthropic.com
  GPT-4   → https://platform.openai.com/api-keys
  Gemini  → https://aistudio.google.com/app/apikey
  Nano AI → https://nano-gpt.com
  Ollama  → https://ollama.ai (безкоштовно, локально)
""")


# ════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════

async def init_agents() -> Tuple[List[MarkovLLMAgent], BayesianRouter]:
    agents = [ClaudeAgent(), GPTAgent(), GeminiAgent(), OllamaAgent(), NanoAIAgent()]
    print(c("gray", "  Перевірка доступності агентів..."))
    checks = await asyncio.gather(*[a.check() for a in agents], return_exceptions=True)
    available = sum(1 for ok in checks if ok is True)
    print(c("green" if available > 0 else "red",
            f"  Доступно {available}/{len(agents)} агентів\n"))
    return agents, BayesianRouter(agents)


async def chat_loop(agents: List[MarkovLLMAgent], router: BayesianRouter,
                    single_ask: Optional[str] = None,
                    force_model: Optional[str] = None):
    history: List[Dict] = []
    forced_model: Optional[str] = force_model

    async def ask(prompt: str) -> None:
        nonlocal forced_model
        history.append({"role": "user", "content": prompt})
        agent = router.route(prompt, forced_model)
        if not agent:
            print(c("red", "  ✗ Жоден агент недоступний. Додай API ключі (/keys)"))
            history.pop()
            return

        task = detect_task_type(prompt)
        print(c("gray", f"  → {agent.name} (task: {task}, L_I: {agent.dist().liquidity():.3f})"))
        print()

        try:
            text, tokens = await agent.call(history)
            print(c("bold", f"  {agent.name}:"))
            print()
            # форматуємо відповідь
            for line in text.split("\n"):
                print(f"  {line}")
            print()
            if tokens:
                print(c("gray", f"  [{tokens} токенів]"))
            history.append({"role": "assistant", "content": text})
        except Exception as e:
            print(c("red", f"  ✗ Помилка: {e}"))
            print(c("yellow", "  Спробую інший агент..."))
            history.pop()
            # fallback — наступний доступний
            fallback = next(
                (a for a in agents if a.available and a.agent_id != agent.agent_id), None
            )
            if fallback:
                try:
                    fallback_messages = history + [{"role": "user", "content": prompt}]
                    text, tokens = await fallback.call(fallback_messages)
                    print(c("bold", f"  {fallback.name} (fallback):"))
                    print()
                    for line in text.split("\n"):
                        print(f"  {line}")
                    print()
                    history.append({"role": "user", "content": prompt})
                    history.append({"role": "assistant", "content": text})
                except Exception as e2:
                    print(c("red", f"  ✗ Fallback теж не спрацював: {e2}"))
                    # повертаємо user message в історію щоб контекст не зламався
                    history.append({"role": "user", "content": prompt})

    # single --ask mode
    if single_ask:
        await ask(single_ask)
        return

    print(c("gray", "  Введи запит або /help для списку команд\n"))

    while True:
        try:
            prefix = c("cyan", f"[{forced_model or 'auto'}]") if forced_model else c("cyan", "[auto]")
            user_input = input(f"  {prefix} {c('bold', '›')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(c("gray", "\n  До побачення."))
            break

        if not user_input:
            continue

        # команди
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd   = parts[0].lower()

            if cmd in ("/exit", "/q", "/quit"):
                print(c("gray", "\n  До побачення."))
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/status":
                print_status(agents, router)
            elif cmd == "/keys":
                print_keys_help()
            elif cmd == "/clear":
                history.clear()
                print(c("green", "  ✓ Історію очищено"))
            elif cmd == "/history":
                if not history:
                    print(c("gray", "  Історія порожня"))
                else:
                    for msg in history:
                        role = c("cyan", msg["role"])
                        print(f"  {role}: {msg['content'][:100]}...")
            elif cmd == "/model":
                if len(parts) < 2:
                    print(c("yellow", "  Вкажи модель: /model claude|gpt4|gemini|ollama|nanoai|auto"))
                elif parts[1] == "auto":
                    forced_model = None
                    print(c("green", "  ✓ Автоматичний вибір увімкнено"))
                else:
                    model_id = parts[1].lower()
                    agent = router.agents.get(model_id)
                    if not agent:
                        print(c("red", f"  ✗ Невідома модель: {model_id}"))
                    elif not agent.available:
                        print(c("yellow", f"  ⚠ {agent.name} недоступна (немає ключа або сервер офлайн)"))
                    else:
                        forced_model = model_id
                        print(c("green", f"  ✓ Обрано: {agent.name}"))
            else:
                print(c("yellow", f"  Невідома команда: {cmd}. Введи /help"))
            continue

        await ask(user_input)


async def main():
    args = sys.argv[1:]

    # parse args
    single_ask   = None
    force_model  = None
    show_status  = False

    i = 0
    while i < len(args):
        if args[i] == "--ask" and i + 1 < len(args):
            single_ask = args[i + 1]; i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            force_model = args[i + 1]; i += 2
        elif args[i] == "--status":
            show_status = True; i += 1
        else:
            i += 1

    print_banner()
    agents, router = await init_agents()

    if show_status:
        print_status(agents, router)
        return

    await chat_loop(agents, router, single_ask, force_model)


if __name__ == "__main__":
    asyncio.run(main())
