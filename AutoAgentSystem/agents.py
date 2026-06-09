import asyncio
import datetime
import inspect
import logging

from runtime import AIFunction, ChatMessage

import events
from base_agent import BaseAgent
from delegation import DelegationBase
from namer import Namer
from tool_config import ToolConfigType
from tools import ToolBase

log = logging.getLogger(__name__)

DEFAULT_ROOT_PROMPT = (
    "# Role: Chief Autonomous Agent\n\n"
    "You are a strategic leader with the ability to analyze user goals, break down complex missions, and delegate work "
    "to capable specialist agents. You should:\n"
    "- Carefully analyze the user's intent\n"
    "- Plan high-level subtasks and strategies\n"
    "- Dynamically assign agents with the right expertise to each subtask\n"
    "- Let agents collaborate or consult each other when necessary\n"
    "- Coordinate until a cohesive, high-quality final answer is produced\n\n"
    "# Autonomous topology selection\n\n"
    "Before acting, silently choose the smallest useful agent topology for the user's task. Do not rely on the user to "
    "name the topology. Choose it yourself from the task shape:\n"
    "- Direct: use no delegation for simple factual, conversational, or single-step tasks.\n"
    "- Flat fan-out: create several specialist agents in parallel when the task has independent dimensions.\n"
    "- Hierarchical tree: create manager/lead agents that each may delegate to specialists when the task is broad, "
    "multi-domain, strategic, architectural, or needs both analysis and recommendations.\n"
    "- Pipeline: create sequential agents when the output of one phase should feed the next phase, such as research -> "
    "design -> critique -> final synthesis.\n"
    "- Debate/review: create competing or reviewing agents when the task benefits from trade-off analysis, risk review, "
    "or quality assurance.\n\n"
    "Use these heuristics:\n"
    "- If a task can be answered well by one expert, answer directly or delegate to one specialist.\n"
    "- If a task has 2-4 independent aspects, use flat fan-out.\n"
    "- If a task has domains that themselves contain subtasks, use a hierarchical tree with team leads and specialists.\n"
    "- If a task asks for an architecture, strategy, product plan, research plan, implementation roadmap, or improvement "
    "proposal, prefer hierarchical tree or pipeline unless it is clearly small.\n"
    "- Never delegate the whole user request as one catch-all task. Delegated instructions must be scoped, role-specific, "
    "and useful for synthesis.\n\n"
    "Agent reuse rule: the same specialist type should be handled by the same helper agent. When delegating, use a "
    "stable role name in the `who` argument, such as 'System Architecture Lead'. If a later subtask belongs to that "
    "same role/type, delegate to the same `who` value instead of creating a new same-type helper.\n\n"
    "When you use delegation, briefly state the topology you selected and why, then delegate. After specialists return, "
    "synthesize the final answer and include a compact agent topology summary if it helps the user understand the work.\n\n"
    "If the user message contains an [MCTS task-planning brief], treat it as an internal execution plan produced by the "
    "system. Follow it unless it is clearly overkill for the task. Do not quote the brief verbatim; use it to guide "
    "delegation and synthesis.\n\n"
    "Never assume all tasks are about research. Some may require writing, generation, tool use, or creativity.\n"
    "Think like a leader. Delegate, monitor, and adapt.\n"
    "The current time is {time}."
)

DEFAULT_DELEGATE_PROMPT = (
    "You are {name}, a specialist agent who can help the main agent accomplish part of a mission.\n"
    "- First, understand your assigned task and explain your approach.\n"
    "- If needed, you may break it down and further delegate subtasks or collaborate with others.\n"
    "- If your assigned task is still broad or contains multiple independent expert areas, choose a smaller topology for "
    "your own subtree instead of doing everything yourself.\n"
    "- If you delegate further, keep each child task bounded and role-specific, then synthesize their findings.\n"
    "- Reuse the same stable `who` role name when a later subtask belongs to an existing specialist type.\n"
    "- You may use tools or APIs if useful.\n"
    "- Produce a clear, concise and actionable result for your task.\n"
    "The current time is {time}."
)

def get_system_prompt(agent: "BaseAgent") -> str:
    now = datetime.datetime.now().strftime("%a %d %b %Y, %I:%M%p")
    return agent.system_prompt.format(name=agent.name, time=now)

class RecursiveAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retry_attempts", 10)
        super().__init__(*args, **kwargs)
        self.namer = Namer()
        self.delegator = None
        self.tools = []
        self.task_description = None
        self.role_name = None
        self.reuse_key = None

    def _register_tools(self, delegator: DelegationBase | None, tools: list[ToolBase]):
        new_functions = {}
        self.delegator = delegator
        if delegator:
            new_functions.update(get_tool_functions(delegator))
        self.tools = tools
        for inst in tools:
            new_functions.update(get_tool_functions(inst))
        self.functions = new_functions

    def get_tool(self, cls: type[ToolBase]) -> ToolBase | None:
        return next((t for t in self.tools if type(t) is cls), None)

    async def create_delegate_agent(self, instructions: str, *, role_name: str | None = None, reuse_key: str | None = None):
        name = self.app.namer.get_name()
        agent_inst = RecursiveAgent(
            self.app.delegate_engine,
            app=self.app,
            parent=self,
            name=name,
            dispatch_creation=False,
            system_prompt=self.app.delegate_system_prompt,
            **self.app.delegate_agent_kwargs,
        )
        agent_inst.task_description = instructions
        agent_inst.role_name = role_name
        agent_inst.reuse_key = reuse_key
        await self.register_child_agent(agent_inst, instructions)
        self.app.register_reusable_agent(reuse_key, agent_inst)
        self.app.dispatch(
            events.AgentDelegated(
                parent_id=self.id,
                child_id=agent_inst.id,
                parent_message_idx=len(self.chat_history) - 1,
                child_message_idx=len(agent_inst.chat_history),
                instructions=instructions,
            )
        )
        return agent_inst

    async def register_child_agent(self, agent_inst, instructions: str | None):
        if self.app.delegation_scheme is None or self.depth == self.app.max_delegation_depth:
            delegation_scheme_inst = None
        else:
            delegation_scheme_inst = self.app.delegation_scheme(app=self.app, agent=agent_inst)

        tool_insts = []
        for t, config in self.app.tool_configs.items():
            if config.get("always_include", False):
                tool_insts.append(t(app=self.app, agent=agent_inst, **config.get("kwargs", {})))

        agent_inst._register_tools(delegator=delegation_scheme_inst, tools=tool_insts)
        if delegation_scheme_inst:
            await delegation_scheme_inst.setup()
        await asyncio.gather(*(t.setup() for t in tool_insts))
        self.app.on_agent_creation(agent_inst)

    async def get_prompt(self) -> list[ChatMessage]:
        if self.system_prompt is not None:
            self.always_included_messages[0] = ChatMessage.system(get_system_prompt(self))
        return await super().get_prompt()

    async def cleanup(self):
        if self.delegator:
            await self.delegator.cleanup()
        await asyncio.gather(*(t.cleanup() for t in self.tools))
        await super().cleanup()

    async def close(self):
        if self.delegator:
            await self.delegator.close()
        await asyncio.gather(*(t.close() for t in self.tools))
        await super().close()

async def create_root_agent(
    *args,
    app,
    delegation_scheme: type[DelegationBase] | None,
    tool_configs: ToolConfigType,
    root_has_tools: bool,
    **kwargs,
) -> RecursiveAgent:
    agent_inst = RecursiveAgent(*args, app=app, dispatch_creation=False, **kwargs)
    if delegation_scheme:
        delegation_scheme_inst = delegation_scheme(app=app, agent=agent_inst)
    else:
        delegation_scheme_inst = None

    tool_insts = []
    for t, config in tool_configs.items():
        if config.get("always_include_root", False) or (config.get("always_include", False) and root_has_tools):
            tool_insts.append(t(app=app, agent=agent_inst, **config.get("kwargs", {})))

    agent_inst._register_tools(delegator=delegation_scheme_inst, tools=tool_insts)

    if delegation_scheme_inst:
        await delegation_scheme_inst.setup()
    await asyncio.gather(*(t.setup() for t in tool_insts))

    app.on_agent_creation(agent_inst)
    return agent_inst

def get_tool_functions(inst: ToolBase) -> dict[str, AIFunction]:
    functions = {}
    for name, member in inspect.getmembers(inst, predicate=inspect.ismethod):
        if not hasattr(member, "__ai_function__"):
            continue
        f = AIFunction(member, **member.__ai_function__)
        if f.name in functions:
            raise ValueError(f"AIFunction {f.name!r} is already registered!")
        functions[f.name] = f
    return functions
