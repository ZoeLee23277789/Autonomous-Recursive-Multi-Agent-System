import abc
import time
from typing import Literal

from runtime import ChatMessage, ChatRole
from pydantic import BaseModel, Field

from state import AgentState, RunState



class BaseEvent(BaseModel, abc.ABC):
    """The base event that all other events should inherit from."""

    __log_event__ = True  # whether or not the event should be logged
    type: str
    timestamp: float = Field(default_factory=time.time)


# server events
class Error(BaseEvent):
    type: Literal["error"] = "error"
    msg: str


class AgentSpawn(AgentState, BaseEvent):
    """
    A new agent was spawned. Includes the state of the agent. See :class:`.BaseAgent`.

    The ID can be the same as an existing ID, in which case this event should overwrite the previous state.
    """

    type: Literal["agent_spawn"] = "agent_spawn"


class AgentDelegated(BaseEvent):
    """An agent was just delegated."""

    type: Literal["agent_delegated"] = "agent_delegated"
    parent_id: str
    child_id: str
    parent_message_idx: int
    child_message_idx: int
    instructions: str


class AgentStateChange(BaseEvent):
    """
    An agent's run state changed.

    This is primarily used for rendering the color of a node in the web interface.
    """

    type: Literal["agent_state_change"] = "agent_state_change"
    id: str
    state: RunState


class TokensUsed(BaseEvent):
    """An agent just finished a request to the engine, which used this many tokens."""

    type: Literal["tokens_used"] = "tokens_used"
    id: str
    prompt_tokens: int
    completion_tokens: int


class AgentMessage(BaseEvent):
    """An agent added a message to its chat history."""

    type: Literal["agent_message"] = "agent_message"
    id: str
    msg: ChatMessage


class RootMessage(BaseEvent):
    """
    The root agent has a new result.

    This will be fired *in addition* to an ``agent_message`` event.
    """

    type: Literal["root_message"] = "root_message"
    msg: ChatMessage


class StreamDelta(BaseEvent):
    """An agent is streaming and emitted a new token."""

    __log_event__ = False

    type: Literal["stream_delta"] = "stream_delta"
    id: str
    delta: str
    role: ChatRole


class RoundComplete(BaseEvent):
    """The root agent has finished a full round and control should be handed off to the user."""

    type: Literal["round_complete"] = "round_complete"
    session_id: str


class SessionClose(BaseEvent):
    """The agent session is closing and clients should be redirected to the home page."""

    __log_event__ = False

    type: Literal["session_close"] = "session_close"
    session_id: str


class SessionMetaUpdate(BaseEvent):
    """Some part of the session metadata has been updated."""

    type: Literal["session_meta_update"] = "session_meta_update"
    title: str


# user events
class SendMessage(BaseEvent):
    type: Literal["send_message"] = "send_message"
    content: str
