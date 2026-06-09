import hashlib
import math
import random
from dataclasses import dataclass


TOPOLOGIES = ("direct", "flat_fanout", "hierarchical_tree", "pipeline", "review_debate")


@dataclass(frozen=True)
class MCTSPlan:
    topology: str
    score: float
    confidence: float
    rationale: str
    first_level_roles: tuple[str, ...]
    execution_notes: tuple[str, ...]

    @property
    def should_inject(self) -> bool:
        return self.topology != "direct"

    def to_prompt(self, user_input: str) -> str:
        if not self.should_inject:
            return user_input

        roles = "\n".join(f"{idx}. {role}" for idx, role in enumerate(self.first_level_roles, start=1))
        notes = "\n".join(f"- {note}" for note in self.execution_notes)
        return (
            f"{user_input}\n\n"
            "[MCTS task-planning brief]\n"
            f"Selected topology: {self.topology}\n"
            f"Selection confidence: {self.confidence:.2f}\n"
            f"Why this topology fits: {self.rationale}\n"
            "Recommended first-level delegation roles:\n"
            f"{roles}\n"
            "Execution constraints:\n"
            f"{notes}\n"
            "Follow this plan unless the user's request clearly requires a smaller topology. "
            "Do not delegate the whole request to a single catch-all agent. "
            "Use each recommended role as the stable delegate `who` value, and reuse the same role name for later "
            "subtasks of the same type."
        )

    def short_summary(self) -> str:
        roles = ", ".join(self.first_level_roles) if self.first_level_roles else "none"
        return f"{self.topology} | score={self.score:.2f} | confidence={self.confidence:.2f} | roles={roles}"


class MCTSTaskPlanner:
    """
    Lightweight task decomposition planner inspired by MCTS-style policy search.

    It does not call the LLM. Instead, it explores topology candidates with stochastic rollouts and scores each
    candidate against task-shape heuristics. The selected plan is injected before delegation so the root agent can
    choose a topology intentionally instead of making a one-shot split.
    """

    def __init__(self, iterations: int = 64, exploration_weight: float = 1.25):
        self.iterations = max(8, iterations)
        self.exploration_weight = exploration_weight

    def plan(self, task: str) -> MCTSPlan:
        features = TaskFeatures.from_text(task)
        rng = random.Random(_stable_seed(task))
        nodes = {topology: _SearchNode(topology=topology) for topology in TOPOLOGIES}

        for _ in range(self.iterations):
            node = self._select(nodes, rng)
            rollout = self._rollout(node.topology, features, rng)
            node.visits += 1
            node.total_score += rollout

        ranked = sorted(nodes.values(), key=lambda n: n.mean_score, reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else best
        confidence = min(0.99, max(0.05, 0.55 + (best.mean_score - second.mean_score) / 2))
        return self._build_plan(best.topology, best.mean_score, confidence, features)

    def _select(self, nodes: dict[str, "_SearchNode"], rng: random.Random) -> "_SearchNode":
        unvisited = [node for node in nodes.values() if node.visits == 0]
        if unvisited:
            return rng.choice(unvisited)

        total_visits = sum(node.visits for node in nodes.values())
        return max(
            nodes.values(),
            key=lambda n: n.mean_score + self.exploration_weight * math.sqrt(math.log(total_visits) / n.visits),
        )

    def _rollout(self, topology: str, features: "TaskFeatures", rng: random.Random) -> float:
        base = topology_fit_score(topology, features)
        cost_penalty = topology_cost_penalty(topology, features)
        divergence_penalty = topology_divergence_penalty(topology, features)
        coverage_bonus = topology_coverage_bonus(topology, features)
        noise = rng.uniform(-0.06, 0.06)
        return max(0.0, min(1.0, base + coverage_bonus - cost_penalty - divergence_penalty + noise))

    def _build_plan(self, topology: str, score: float, confidence: float, features: "TaskFeatures") -> MCTSPlan:
        return MCTSPlan(
            topology=topology,
            score=score,
            confidence=confidence,
            rationale=topology_rationale(topology, features),
            first_level_roles=roles_for_topology(topology, features),
            execution_notes=execution_notes_for_topology(topology, features),
        )


@dataclass
class _SearchNode:
    topology: str
    visits: int = 0
    total_score: float = 0.0

    @property
    def mean_score(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_score / self.visits


@dataclass(frozen=True)
class TaskFeatures:
    text: str
    length: int
    aspect_count: int
    asks_architecture: bool
    asks_strategy: bool
    asks_research: bool
    asks_implementation: bool
    asks_review: bool
    asks_comparison: bool
    asks_sequential: bool
    is_simple: bool

    @classmethod
    def from_text(cls, text: str) -> "TaskFeatures":
        normalized = text.lower()
        separators = sum(text.count(token) for token in ("、", "，", ",", ";", "；", "\n"))
        aspect_words = count_keywords(
            normalized,
            (
                "架構",
                "architecture",
                "模組",
                "module",
                "風險",
                "risk",
                "部署",
                "deploy",
                "評估",
                "evaluate",
                "改進",
                "improve",
                "roadmap",
                "需求",
                "requirement",
                "資料流程",
                "pipeline",
            ),
        )
        length = len(text)
        return cls(
            text=text,
            length=length,
            aspect_count=max(1, separators + aspect_words),
            asks_architecture=has_any(normalized, ("架構", "系統設計", "architecture", "module", "模組")),
            asks_strategy=has_any(normalized, ("策略", "規劃", "roadmap", "改進", "建議", "proposal", "plan")),
            asks_research=has_any(normalized, ("研究", "research", "調查", "survey", "文獻")),
            asks_implementation=has_any(normalized, ("實作", "implementation", "部署", "deploy", "程式", "code", "工程")),
            asks_review=has_any(normalized, ("審查", "review", "風險", "risk", "安全", "security", "critique")),
            asks_comparison=has_any(normalized, ("比較", "compare", "trade-off", "權衡", "優缺點")),
            asks_sequential=has_any(normalized, ("流程", "pipeline", "步驟", "階段", "phase", "先", "再")),
            is_simple=length < 80 and aspect_words <= 1 and separators <= 1,
        )


def topology_fit_score(topology: str, features: TaskFeatures) -> float:
    if topology == "direct":
        return 0.82 if features.is_simple else 0.25
    if topology == "flat_fanout":
        return 0.35 + min(0.32, features.aspect_count * 0.055)
    if topology == "hierarchical_tree":
        score = 0.35
        if features.asks_architecture:
            score += 0.2
        if features.asks_strategy:
            score += 0.14
        if features.asks_implementation:
            score += 0.1
        if features.aspect_count >= 4:
            score += 0.14
        return score
    if topology == "pipeline":
        score = 0.32
        if features.asks_sequential:
            score += 0.2
        if features.asks_research:
            score += 0.12
        if features.asks_implementation:
            score += 0.1
        return score
    if topology == "review_debate":
        score = 0.25
        if features.asks_review:
            score += 0.25
        if features.asks_comparison:
            score += 0.18
        if features.asks_strategy:
            score += 0.08
        return score
    return 0.0


def topology_cost_penalty(topology: str, features: TaskFeatures) -> float:
    if topology == "direct":
        return 0.0
    if features.is_simple:
        return 0.22
    return {
        "flat_fanout": 0.08,
        "hierarchical_tree": 0.13,
        "pipeline": 0.1,
        "review_debate": 0.12,
    }[topology]


def topology_divergence_penalty(topology: str, features: TaskFeatures) -> float:
    if topology == "hierarchical_tree" and features.aspect_count <= 2:
        return 0.12
    if topology == "flat_fanout" and features.aspect_count >= 7:
        return 0.1
    if topology == "review_debate" and not (features.asks_review or features.asks_comparison):
        return 0.09
    return 0.0


def topology_coverage_bonus(topology: str, features: TaskFeatures) -> float:
    if topology == "hierarchical_tree" and features.asks_architecture and features.asks_strategy:
        return 0.1
    if topology == "pipeline" and features.asks_research and features.asks_implementation:
        return 0.08
    if topology == "flat_fanout" and 3 <= features.aspect_count <= 5:
        return 0.08
    if topology == "review_debate" and features.asks_review and features.asks_comparison:
        return 0.08
    return 0.0


def topology_rationale(topology: str, features: TaskFeatures) -> str:
    if topology == "direct":
        return "The task appears narrow enough for a single expert answer."
    if topology == "flat_fanout":
        return "The task has independent dimensions that can be analyzed in parallel and synthesized."
    if topology == "hierarchical_tree":
        return (
            "The task is broad and multi-domain, so manager agents should own major workstreams and delegate "
            "specialized analysis inside each subtree."
        )
    if topology == "pipeline":
        return "The task benefits from ordered phases where earlier findings guide later design and review."
    return "The task needs trade-off analysis, critique, or risk review before final synthesis."


def roles_for_topology(topology: str, features: TaskFeatures) -> tuple[str, ...]:
    if topology == "direct":
        return ()
    if topology == "flat_fanout":
        roles = ["Architecture Analyst", "Risk and Evaluation Analyst", "Improvement Strategy Analyst"]
        if features.asks_implementation:
            roles.append("Implementation Analyst")
        return tuple(roles[:4])
    if topology == "hierarchical_tree":
        roles = ["System Architecture Lead", "Implementation and Operations Lead", "Evaluation and Risk Lead"]
        if features.asks_strategy:
            roles.append("Roadmap and Improvement Lead")
        return tuple(roles[:4])
    if topology == "pipeline":
        return ("Research/Discovery Agent", "Design Agent", "Critique Agent", "Synthesis Agent")
    return ("Proponent Agent", "Skeptic/Risk Agent", "Trade-off Judge", "Final Synthesizer")


def execution_notes_for_topology(topology: str, features: TaskFeatures) -> tuple[str, ...]:
    common = (
        "Keep each delegated task bounded and role-specific.",
        "Prefer 2-4 first-level delegates; avoid uncontrolled task explosion.",
        "Wait for delegated results before final synthesis.",
    )
    if topology == "hierarchical_tree":
        return (
            *common,
            "Team leads may further delegate only if their workstream contains distinct specialist subtasks.",
            "Final answer should include a compact tree summary and concrete recommendations.",
        )
    if topology == "pipeline":
        return (
            *common,
            "Run phases in order; do not start critique until design assumptions are available.",
        )
    if topology == "review_debate":
        return (
            *common,
            "Ask review agents to disagree constructively and surface failure modes.",
        )
    if topology == "flat_fanout":
        return (
            *common,
            "Run specialists in parallel and merge overlapping findings.",
        )
    return common


def has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def count_keywords(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
