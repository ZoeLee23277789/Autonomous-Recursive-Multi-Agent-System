import asyncio
import datetime
import inspect
import logging

from kani import AIFunction, ChatMessage

import events
from base_kani import BaseKani
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
    "Never assume all tasks are about research. Some may require writing, generation, tool use, or creativity.\n"
    "Think like a leader. Delegate, monitor, and adapt.\n"
    "The current time is {time}."
)

DEFAULT_DELEGATE_PROMPT = (
    "You are {name}, a specialist agent who can help the main agent accomplish part of a mission.\n"
    "- First, understand your assigned task and explain your approach.\n"
    "- If needed, you may break it down and further delegate subtasks or collaborate with others.\n"
    "- You may use tools or APIs if useful.\n"
    "- Produce a clear, concise and actionable result for your task.\n"
    "The current time is {time}."
)

def get_system_prompt(kani: "BaseKani") -> str:
    now = datetime.datetime.now().strftime("%a %d %b %Y, %I:%M%p")
    return kani.system_prompt.format(name=kani.name, time=now)

class ReDelKani(BaseKani):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retry_attempts", 10)
        super().__init__(*args, **kwargs)
        self.namer = Namer()
        self.delegator = None
        self.tools = []
        self.task_description = None

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

    async def create_delegate_kani(self, instructions: str):
        name = self.namer.get_name()
        kani_inst = ReDelKani(
            self.app.delegate_engine,
            app=self.app,
            parent=self,
            name=name,
            dispatch_creation=False,
            system_prompt=self.app.delegate_system_prompt,
            **self.app.delegate_kani_kwargs,
        )
        kani_inst.task_description = instructions
        await self.register_child_kani(kani_inst, instructions)
        self.app.dispatch(
            events.KaniDelegated(
                parent_id=self.id,
                child_id=kani_inst.id,
                parent_message_idx=len(self.chat_history) - 1,
                child_message_idx=len(kani_inst.chat_history),
                instructions=instructions,
            )
        )
        return kani_inst

    async def register_child_kani(self, kani_inst, instructions: str | None):
        if self.app.delegation_scheme is None or self.depth == self.app.max_delegation_depth:
            delegation_scheme_inst = None
        else:
            delegation_scheme_inst = self.app.delegation_scheme(app=self.app, kani=kani_inst)

        tool_insts = []
        for t, config in self.app.tool_configs.items():
            if config.get("always_include", False):
                tool_insts.append(t(app=self.app, kani=kani_inst, **config.get("kwargs", {})))

        kani_inst._register_tools(delegator=delegation_scheme_inst, tools=tool_insts)
        if delegation_scheme_inst:
            await delegation_scheme_inst.setup()
        await asyncio.gather(*(t.setup() for t in tool_insts))
        self.app.on_kani_creation(kani_inst)

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

async def create_root_kani(
    *args,
    app,
    delegation_scheme: type[DelegationBase] | None,
    tool_configs: ToolConfigType,
    root_has_tools: bool,
    **kwargs,
) -> ReDelKani:
    kani_inst = ReDelKani(*args, app=app, dispatch_creation=False, **kwargs)
    if delegation_scheme:
        delegation_scheme_inst = delegation_scheme(app=app, kani=kani_inst)
    else:
        delegation_scheme_inst = None

    tool_insts = []
    for t, config in tool_configs.items():
        if config.get("always_include_root", False) or (config.get("always_include", False) and root_has_tools):
            tool_insts.append(t(app=app, kani=kani_inst, **config.get("kwargs", {})))

    kani_inst._register_tools(delegator=delegation_scheme_inst, tools=tool_insts)

    if delegation_scheme_inst:
        await delegation_scheme_inst.setup()
    await asyncio.gather(*(t.setup() for t in tool_insts))

    app.on_kani_creation(kani_inst)
    return kani_inst

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
