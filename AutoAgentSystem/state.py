import enum
from typing import TYPE_CHECKING

from runtime import AIFunction, ChatMessage, ChatRole
from pydantic import BaseModel

if TYPE_CHECKING:
    from base_agent import BaseAgent


class RunState(enum.Enum):
    """
    * ``RunState.STOPPED``: This agent is not currently running anything or waiting on a child.
    * ``RunState.RUNNING``: This agent is currently generating text.
    * ``RunState.WAITING``: This agent is waiting for the results of a sub-agent.
    * ``RunState.ERRORED``: This agent has run into a fatal error. Its internal state is indeterminate.
    """

    STOPPED = "stopped"  # not currently running anything or waiting on a child
    RUNNING = "running"  # gpt-4 is generating something
    WAITING = "waiting"  # waiting on a child
    ERRORED = "errored"  # panic


class AIFunctionState(BaseModel):
    name: str
    desc: str
    auto_retry: bool
    auto_truncate: int | None
    after: ChatRole
    json_schema: dict

    @classmethod
    def from_aifunction(cls, f: AIFunction):
        return cls(
            name=f.name,
            desc=f.desc,
            auto_retry=f.auto_retry,
            auto_truncate=f.auto_truncate,
            after=f.after,
            json_schema=f.json_schema,
        )


class AgentState(BaseModel):
    id: str
    depth: int
    parent: str | None
    children: list[str]
    always_included_messages: list[ChatMessage]
    chat_history: list[ChatMessage]
    state: RunState
    name: str
    engine_type: str
    engine_repr: str = ""
    functions: list[AIFunctionState]

    @classmethod
    def from_agent(cls, ai: "BaseAgent", **kwargs):
        return cls(
            id=ai.id,
            depth=ai.depth,
            parent=ai.parent.id if ai.parent else None,
            children=list(ai.children),
            always_included_messages=ai.always_included_messages,
            chat_history=ai.chat_history,
            state=ai.state,
            name=ai.name,
            engine_type=type(ai.engine).__name__,
            engine_repr=repr(ai.engine),
            functions=[AIFunctionState.from_aifunction(f) for f in ai.functions.values()],
            **kwargs,
        )
