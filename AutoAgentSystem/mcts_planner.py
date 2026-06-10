import json
import re
from dataclasses import dataclass

from runtime import BaseEngine, ChatMessage


TOPOLOGIES = ("direct", "flat_fanout", "network_peer", "hierarchical_tree", "pipeline", "review_debate")

LLM_TOPOLOGY_PLANNER_PROMPT = """
You are a topology advisor for an autonomous recursive multi-agent system.

Your only job is to recommend which topology best fits the user's task.
Do not solve the task. Do not write the final answer. Do not propose concrete agent names or first-level roles.

Root is always present and is always the top-level coordinator. The topology describes the structure below Root.

Choose exactly one topology:
- direct: simple task; Root can answer directly without delegation.
- flat_fanout: Root runs independent specialist analyses in parallel, then synthesizes.
- network_peer: Root creates same-level peer agents that should consult, review, or support each other.
- hierarchical_tree: Root creates first-level leads/managers; each lead may create deeper specialists when needed.
- pipeline: Root creates ordered phase agents where earlier output feeds later phases.
- review_debate: Root creates proponent/skeptic/judge or review agents for trade-off, risk, or quality evaluation.

Return only valid JSON:
{
  "topology": "direct | flat_fanout | network_peer | hierarchical_tree | pipeline | review_debate",
  "confidence": 0.0,
  "rationale": "one short reason",
  "execution_notes": ["short advisory note", "..."]
}

Rules:
- Recommend only the topology, not the concrete agent list.
- For network_peer, explain that same-level agents should collaborate.
- For hierarchical_tree, explain that Root may create leads/managers if needed.
- For pipeline, explain that ordered phases may be useful.
- This is advisory guidance only; agents may adapt during execution.
""".strip()


@dataclass(frozen=True)
class MCTSPlan:
    topology: str
    confidence: float
    rationale: str
    execution_notes: tuple[str, ...]
    source: str = "llm"

    @property
    def should_inject(self) -> bool:
        return self.topology != "direct"

    @property
    def score(self) -> float:
        return self.confidence

    def to_prompt(self, user_input: str) -> str:
        if not self.should_inject:
            return user_input

        notes = "\n".join(f"- {note}" for note in self.execution_notes)
        return (
            f"{user_input}\n\n"
            "[MCTS task-planning brief]\n"
            "Planner mode: LLM topology advisor. This brief is advisory, not a hard runtime controller.\n"
            f"Selected topology: {self.topology}\n"
            f"Selection confidence: {self.confidence:.2f}\n"
            f"Why this topology fits: {self.rationale}\n"
            "Root policy: Root is always the top-level coordinator; the selected topology describes the structure below Root.\n"
            "Suggested coordination guidance:\n"
            f"{notes}\n"
            "Root should decide the concrete agents, delegation boundaries, and collaboration links based on the "
            "selected topology and the user's task. Avoid delegating the whole request to a single catch-all agent."
        )

    def short_summary(self) -> str:
        return f"{self.topology} | source={self.source} | confidence={self.confidence:.2f}"


class MCTSTaskPlanner:
    """
    LLM-only topology advisor.

    The planner recommends only the topology. It does not solve the user task, name agents, run manual scoring, or
    hard-control delegation. The selected topology is injected into the Root prompt, then the multi-agent system
    handles concrete delegation, peer collaboration, and synthesis.
    """

    def __init__(self, iterations: int = 64, exploration_weight: float = 1.25):
        # Kept for backwards compatibility with existing AutoAgentSystem constructor arguments.
        self.iterations = iterations
        self.exploration_weight = exploration_weight

    async def plan(self, task: str, engine: BaseEngine | None = None) -> MCTSPlan:
        if engine is None:
            return fallback_plan()

        try:
            return await self._llm_plan(task, engine) or fallback_plan()
        except Exception:
            return fallback_plan()

    async def _llm_plan(self, task: str, engine: BaseEngine) -> MCTSPlan | None:
        messages = [
            ChatMessage.system(LLM_TOPOLOGY_PLANNER_PROMPT),
            ChatMessage.user(
                "User task:\n"
                f"{task}\n\n"
                "Return only the JSON object."
            ),
        ]
        completion = await engine.complete(
            messages,
            temperature=0.1,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        data = _parse_json_object(completion.message.content or "")
        if not data:
            return None

        topology = normalize_topology(data.get("topology"))
        if topology not in TOPOLOGIES:
            return None

        confidence = _clamp_float(data.get("confidence"), 0.5)
        rationale = str(data.get("rationale") or "The LLM advisor selected this topology for the task.").strip()
        notes = _clean_string_list(data.get("execution_notes") or data.get("notes"), limit=8)
        if not notes:
            notes = ["Use the selected topology as guidance, then adapt during execution if needed."]

        return MCTSPlan(
            topology=topology,
            confidence=confidence,
            rationale=rationale[:600],
            execution_notes=tuple(notes),
            source="llm",
        )


def fallback_plan() -> MCTSPlan:
    return MCTSPlan(
        topology="direct",
        confidence=0.05,
        rationale="The LLM topology advisor was unavailable, so no topology recommendation was injected.",
        execution_notes=("Root should decide whether delegation is needed from the original task.",),
        source="fallback",
    )


def normalize_topology(value: object) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "direct": "direct",
        "single": "direct",
        "single_agent": "direct",
        "flat": "flat_fanout",
        "flat_fanout": "flat_fanout",
        "flat_supervisor": "flat_fanout",
        "flatsupervisor": "flat_fanout",
        "supervisor": "flat_fanout",
        "parallel": "flat_fanout",
        "fanout": "flat_fanout",
        "network": "network_peer",
        "network_peer": "network_peer",
        "peer": "network_peer",
        "peer_to_peer": "network_peer",
        "p2p": "network_peer",
        "hierarchical": "hierarchical_tree",
        "hierarchical_tree": "hierarchical_tree",
        "hierarchy": "hierarchical_tree",
        "tree": "hierarchical_tree",
        "pipeline": "pipeline",
        "sequential": "pipeline",
        "review": "review_debate",
        "debate": "review_debate",
        "review_debate": "review_debate",
        "critic": "review_debate",
    }
    return aliases.get(key)


def _parse_json_object(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _clamp_float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(0.99, max(0.05, number))


def _clean_string_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text[:180])
        if len(cleaned) >= limit:
            break
    return cleaned
