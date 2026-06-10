from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

import asyncio
import contextlib
import functools
import logging
import time
import uuid
import events
from eventlogger import OtelEventLogger

from collections.abc import AsyncIterable
from typing import Any, Awaitable, Callable
from weakref import WeakValueDictionary

from runtime import BaseEngine, ChatRole, OpenAIEngine

from base_agent import BaseAgent
from delegation.delegate_and_wait import DelegateWait
from delegation.delegate_one import DelegateOne
from agents import DEFAULT_DELEGATE_PROMPT, DEFAULT_ROOT_PROMPT, create_root_agent
from mcts_planner import MCTSPlan, MCTSTaskPlanner
from namer import Namer
from tool_config import ToolConfigType, validate_tool_configs
from utils import AUTOGENERATE_TITLE, AutogenerateTitle, generate_conversation_title
try:
    from graphviz import Digraph
except ImportError:
    GRAPHVIZ_IMPORT_ERROR = True

    class Digraph:
        def __init__(self, *args, **kwargs):
            pass

        def node(self, *args, **kwargs):
            pass

        def edge(self, *args, **kwargs):
            pass

        def render(self, *args, **kwargs):
            return None
else:
    GRAPHVIZ_IMPORT_ERROR = False

try:
    from tools.browsing.impl import Browsing, ArxivSearch
except ImportError:
    Browsing = None
    ArxivSearch = None

log = logging.getLogger(__name__)


@functools.cache
def default_engine():
    return OpenAIEngine(
        model="gpt-4o-mini",
        temperature=0.3,        # 控制隨機性：越低越穩定
        top_p=0.9,              # 控制 nucleus sampling
        max_tokens=1028         # 每次回應最多 token
    )


class AutoAgentSystem:
    """This class represents a single session of a recursive multi-agent system.

    It's responsible for:

    * all delegation configuration options
    * all the spawned agents and their relations within the session
    * dispatching all events from the session
    * logging events

    All arguments to the constructor are keyword arguments.
    """

    def __init__(
        self,
        *,
        # engines
        root_engine: BaseEngine = None,
        delegate_engine: BaseEngine = None,
        # prompt/agent
        root_system_prompt: str | None = DEFAULT_ROOT_PROMPT,
        root_agent_kwargs: dict = None,
        delegate_system_prompt: str | None = DEFAULT_DELEGATE_PROMPT,
        delegate_agent_kwargs: dict = None,
        # delegation/function calling
        delegation_scheme: type | None = DelegateWait,
        max_delegation_depth: int = 4,
        mcts_planning: bool = True,
        mcts_iterations: int = 64,
        tool_configs: ToolConfigType = None,
        root_has_tools: bool = False,
        # logging
        title: str | AutogenerateTitle | None = AUTOGENERATE_TITLE,
        log_dir: Path = None,
        clear_existing_log: bool = False,
        session_id: str = None,
    ):
        self.visualizer = TreeVisualizer()
        self.global_task_log = []  # 🧠 全局任務追蹤記憶體
        self.namer = Namer()
        self.reusable_agents = {}
        self.agent_edges = []
        self._agent_edge_keys = set()
        self.peer_notes = []
        self.active_topology = None
        self.pipeline_roles = []
        self.pipeline_stage_index = 0
        self.pipeline_running_role = None
        self.pipeline_last_agent_id = None
        """
        :param root_engine: The engine to use for the root agent. Requires function calling. (default: gpt-4o)
        :param delegate_engine: The engine to use for each delegate agent. Requires function calling. (default: gpt-4o)
        :param root_system_prompt: The system prompt for the root agent. See ``agents`` for default.
        :param root_agent_kwargs: Additional keyword args to pass to the underlying chat-agent runtime.
        :param delegate_system_prompt: The system prompt for each delegate agent. See ``agents`` for default.
        :param delegate_agent_kwargs: Additional keyword args to pass to the underlying chat-agent runtime.
        :param delegation_scheme: A class that each agent capable of delegation will use to provide the delegation tool.
            See ``delegation`` for examples. Can be ``None`` to disable delegation.
        :param max_delegation_depth: The maximum delegation depth. Agents created at this depth will not inherit from the
            ``delegation_scheme`` class.
        :param tool_configs: A mapping of tool mixin classes to their configurations (see :class:`.ToolConfig`).
        :param root_has_tools: Whether the root agent should have access to the configured tools (default
            False).
        :param title: The title of this session. Set to ``AUTOGENERATE_TITLE`` to automatically generate one
            (default), or ``None`` to disable title generation.
        :param log_dir: A path to a directory to save logs for this session. Defaults to
            ``$AUTO_AGENT_HOME/instances/{session_id}/`` (default ``~/.auto_agent/instances/{session_id}``).
        :param clear_existing_log: If the log directory has existing events, clear them before writing new events.
            Otherwise, append to existing events.
        :param session_id: The ID of this session. Generally this should not be set manually; it is used for loading
            previous states.
        """
        if root_engine is None:
            root_engine = default_engine()
        if delegate_engine is None:
            delegate_engine = default_engine()
        if root_agent_kwargs is None:
            root_agent_kwargs = {}
        if delegate_agent_kwargs is None:
            delegate_agent_kwargs = {}
        if tool_configs is None:
            tool_configs = {}

        validate_tool_configs(tool_configs)

        # engines
        self.root_engine = root_engine
        self.delegate_engine = delegate_engine
        # prompt/agent
        self.root_system_prompt = root_system_prompt
        self.root_agent_kwargs = root_agent_kwargs
        self.delegate_system_prompt = delegate_system_prompt
        self.delegate_agent_kwargs = delegate_agent_kwargs
        # delegation/function calling
        self.delegation_scheme = delegation_scheme
        self.max_delegation_depth = max_delegation_depth
        self.mcts_planning = mcts_planning
        self.mcts_iterations = mcts_iterations
        self.mcts_planner = MCTSTaskPlanner(iterations=mcts_iterations)
        self.last_mcts_plan: MCTSPlan | None = None
        self.tool_configs = tool_configs
        # 註冊工具
        default_tools = {}
        if Browsing is not None:
            default_tools[Browsing] = {
                "always_include": True,
                "kwargs": {}
            }
        if ArxivSearch is not None:
            default_tools[ArxivSearch] = {
                "always_include": True,
                "kwargs": {}
            }
        self.tool_configs.update(default_tools)
        self.root_has_tools = root_has_tools

        # internals
        self._init_lock = asyncio.Lock()

        # events
        self.listeners = []
        self.event_queue = asyncio.Queue()
        self.dispatch_task = None
        # state
        self.session_id = session_id or f"{int(time.time())}-{uuid.uuid4()}"
        if title is AUTOGENERATE_TITLE:
            self.title = None
            self.add_listener(self.create_title_listener)
        else:
            self.title = title
        # logging
        self.logger = OtelEventLogger(self, self.session_id, log_dir=log_dir, clear_existing_log=clear_existing_log)
        self.add_listener(self.logger.log_event)
        # agents
        self.agents = WeakValueDictionary()
        self.root_agent = None
        self._closed = False

    def get_config(self, **kwargs):
        """
        Get a dictionary with arguments suitable for passing to an AutoAgentSystem constructor to create a new instance with
        mostly the same configuration.

        By default, the title, log_dir, and session_id will not be copied. Explicitly set these as keyword
        arguments if you want to copy them.

        Pass keyword arguments to override existing configuration options (valid arguments are same as constructor).
        """
        config = {
            "root_engine": self.root_engine,
            "delegate_engine": self.delegate_engine,
            "root_system_prompt": self.root_system_prompt,
            "root_agent_kwargs": self.root_agent_kwargs,
            "delegate_system_prompt": self.delegate_system_prompt,
            "delegate_agent_kwargs": self.delegate_agent_kwargs,
            "delegation_scheme": self.delegation_scheme,
            "max_delegation_depth": self.max_delegation_depth,
            "mcts_planning": self.mcts_planning,
            "mcts_iterations": self.mcts_iterations,
            "tool_configs": self.tool_configs,
            "root_has_tools": self.root_has_tools,
        }
        config.update(kwargs)
        return config

    def register_reusable_agent(self, reuse_key: str | None, agent: BaseAgent):
        if reuse_key:
            self.reusable_agents[reuse_key] = agent

    def get_reusable_agent(self, reuse_key: str | None):
        if not reuse_key:
            return None
        return self.reusable_agents.get(reuse_key)

    def on_agent_reuse(self, parent: BaseAgent, child: BaseAgent):
        parent.children[child.id] = child
        self.add_agent_edge(parent.id, child.id, "reuse")

    def add_agent_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        *,
        label: str | None = None,
        metadata: dict | None = None,
    ):
        if not source_id or not target_id or source_id == target_id:
            return None

        label = label if label is not None else (None if edge_type == "delegation" else edge_type)
        metadata = metadata or {}
        edge_key = (source_id, target_id, edge_type, label)
        if edge_key in self._agent_edge_keys:
            return None

        edge = {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "label": label,
            "metadata": metadata,
        }
        self._agent_edge_keys.add(edge_key)
        self.agent_edges.append(edge)
        self.visualizer.add_edge(source_id, target_id, edge_type=edge_type, label=label)
        self.dispatch(events.AgentEdge(**edge))
        return edge

    def add_peer_note(self, agent: BaseAgent, topic: str, content: str):
        note = {
            "agent_id": agent.id,
            "agent": agent.name,
            "role": getattr(agent, "role_name", None),
            "depth": agent.depth,
            "topic": topic,
            "content": content,
        }
        self.peer_notes.append(note)
        return note

    def get_peer_notes(self, *, depth: int | None = None, exclude_agent_id: str | None = None, topic: str | None = None):
        notes = self.peer_notes
        if depth is not None:
            notes = [note for note in notes if note["depth"] == depth]
        if exclude_agent_id:
            notes = [note for note in notes if note["agent_id"] != exclude_agent_id]
        if topic:
            topic_key = topic.strip().lower()
            notes = [
                note for note in notes
                if topic_key in note["topic"].lower() or topic_key in note["content"].lower()
            ]
        return notes

    def on_peer_note_read(self, reader: BaseAgent, note: dict):
        if note.get("agent_id") == reader.id:
            return
        peer = self.agents.get(note.get("agent_id"))
        if peer is None:
            return
        if peer.depth == reader.depth:
            self.add_agent_edge(reader.id, peer.id, "peer", label="peer", metadata={"source": "read_peer_notes"})

    def activate_plan(self, plan: MCTSPlan | None):
        self.last_mcts_plan = plan
        self.active_topology = plan.topology if plan else None
        self.peer_notes = []
        self.pipeline_roles = list(plan.first_level_roles) if plan and plan.topology == "pipeline" else []
        self.pipeline_stage_index = 0
        self.pipeline_running_role = None
        self.pipeline_last_agent_id = None

    def _normalize_role_key(self, role_name: str | None):
        if not role_name:
            return None
        return " ".join(role_name.strip().lower().split())

    def validate_pipeline_delegate(self, parent: BaseAgent, role_name: str | None):
        if self.active_topology in ("network_peer", "review_debate", "flat_fanout") and parent is not self.root_agent:
            return (
                f"{self.active_topology} mode is active. First-level agents should collaborate with peers using "
                "link_peer, publish_peer_note, read_peer_notes, or consult_peer instead of creating subordinate agents."
            )

        if self.active_topology != "pipeline":
            return None

        if parent is not self.root_agent:
            return (
                "Pipeline mode is active. Phase agents should complete their assigned phase directly, publish/read "
                "peer notes if context is needed, and avoid creating subordinate agents unless Root changes the plan."
            )

        if not self.pipeline_roles:
            return None

        if self.pipeline_running_role:
            return (
                f"Pipeline phase {self.pipeline_running_role!r} is still running. "
                "Use wait(until='all') before starting the next pipeline phase."
            )

        if self.pipeline_stage_index >= len(self.pipeline_roles):
            return None

        expected = self.pipeline_roles[self.pipeline_stage_index]
        if self._normalize_role_key(role_name) != self._normalize_role_key(expected):
            return (
                f"Pipeline order requires the next stage to be {expected!r}. "
                "Delegate only that stage now, wait for it, then continue to the following stage."
            )
        return None

    def mark_pipeline_role_started(self, helper: BaseAgent):
        if self.active_topology != "pipeline" or helper.parent is not self.root_agent:
            return
        role = getattr(helper, "role_name", None) or helper.name
        self.pipeline_running_role = role
        if self.pipeline_last_agent_id and self.pipeline_last_agent_id != helper.id:
            self.add_agent_edge(self.pipeline_last_agent_id, helper.id, "handoff", label="handoff")

    def mark_pipeline_role_completed(self, helper: BaseAgent):
        if self.active_topology != "pipeline" or helper.parent is not self.root_agent:
            return
        role = getattr(helper, "role_name", None) or helper.name
        if self._normalize_role_key(role) == self._normalize_role_key(self.pipeline_running_role):
            self.pipeline_running_role = None
            self.pipeline_last_agent_id = helper.id
            if self.pipeline_stage_index < len(self.pipeline_roles):
                expected = self.pipeline_roles[self.pipeline_stage_index]
                if self._normalize_role_key(role) == self._normalize_role_key(expected):
                    self.pipeline_stage_index += 1

    def prepare_task_prompt(self, user_input: str, announce: bool = False) -> str:
        if not self.mcts_planning:
            self.activate_plan(None)
            return user_input

        plan = self.mcts_planner.plan(user_input)
        self.activate_plan(plan)
        if announce and plan.should_inject:
            print(f"\n[🧭 MCTS 任務規劃] {plan.short_summary()}\n")
        return plan.to_prompt(user_input)

    async def ensure_init(self):
        """Called at least once before any messaging happens. Used to do async init. Must be idempotent."""
        async with self._init_lock:  # lock in case of parallel calls - no double creation
            if self.root_agent is None:
                self.root_agent = await create_root_agent(
                    self.root_engine,
                    # create_root_agent args
                    app=self,
                    delegation_scheme=self.delegation_scheme,
                    tool_configs=self.tool_configs,
                    root_has_tools=self.root_has_tools,
                    # BaseAgent args
                    name="root",
                    # runtime args
                    system_prompt=self.root_system_prompt,
                    **self.root_agent_kwargs,
                )
            if self.dispatch_task is None:
                self.dispatch_task = asyncio.create_task(
                    self._dispatch_task(), name=f"agent-dispatch-{self.session_id}"
                )
        return self.root_agent

    # === entrypoints ===
    async def chat_from_queue(self, q: asyncio.Queue):
        """Get chat messages from a provided queue. Used internally in the visualization server."""
        await self.ensure_init()
        while True:
            # main loop
            try:
                user_msg = await q.get()
                log.info(f"Message from queue: {user_msg.content!r}")
                if isinstance(user_msg, events.BaseEvent):
                    self.dispatch(user_msg)
                else:
                    self.dispatch(events.SendMessage(content=user_msg.content))
                planned_content = self.prepare_task_prompt(user_msg.content)
                async for stream in self.root_agent.full_round_stream(planned_content):
                    msg = await stream.message()
                    if msg.role == ChatRole.ASSISTANT:
                        log.info(f"AI: {msg}")
            except Exception:
                log.exception("Error in chat_from_queue:")
            finally:
                self.dispatch(events.RoundComplete(session_id=self.session_id))
                await self.drain()
                await self.logger.write_state()  # autosave

    # async def chat_in_terminal(self):
    #     await self.ensure_init()
    #     while True:
    #         try:
    #             user_input = input("USER: ")
    #             if user_input.strip().lower() in ("exit", "quit"):
    #                 print("👋 使用者中斷。再見！")
    #                 await self.close()
    #                 break
    
    #             async for stream in self.root_agent.full_round_stream(user_input):
    #                 msg = await stream.message()
    #                 if msg.role == ChatRole.ASSISTANT:
    #                     print(f"AI: {msg.text}")
    #         except KeyboardInterrupt:
    #             print("\n👋 使用者中斷（Ctrl+C）。再見！")
    #             await self.close()
    #             break
    #         finally:
    #             self.dispatch(events.RoundComplete(session_id=self.session_id))
    #             await self.logger.write_state()
    #             self.visualizer.render("agent_tree", view=True)  # <- 加這行

    async def chat_in_terminal(self):
        await self.ensure_init()
        while True:
            shutting_down = False
            try:
                user_input = input("USER: ")
                if not user_input.strip():
                    continue
                if user_input.strip().lower() in ("exit", "quit"):
                    print("👋 使用者中斷。再見！")
                    shutting_down = True
                    await self.close()
                    break
    
                self.dispatch(events.SendMessage(content=user_input))
                planned_input = self.prepare_task_prompt(user_input, announce=True)
                async for stream in self.root_agent.full_round_stream(planned_input):
                    print("AI:", end="", flush=True)
                    content = ""
                    async for token in stream:
                        print(token, end="", flush=True)
                        content += token
                    print()  # 換行結尾
    
                    msg = await stream.message()
                    if msg.tool_calls:
                        print(f"\n[🛠️ Tool Call]: {msg.tool_calls}")

            except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
                print("\n👋 使用者中斷（Ctrl+C）。再見！")
                shutting_down = True
                await self.close()
                break

            except Exception as e:
                message = str(e)
                if "invalid_api_key" in message or "Incorrect API key" in message or "AuthenticationError" in type(e).__name__:
                    import os

                    key = os.getenv("OPENAI_API_KEY", "")
                    key_hint = f"{key[:10]}...{key[-4:]}" if key else "missing"
                    print(
                        "\n❌ OpenAI API key 無效、已失效，或目前 project 無法呼叫聊天模型。\n"
                        f"目前程式讀到的 key：{key_hint}\n"
                        "請確認 .env 的 OPENAI_API_KEY，或用下方 chat completion 測試指令確認。\n"
                    )
                    continue
                if "Connection error" in message or "APIConnectionError" in type(e).__name__:
                    print("\n❌ 無法連線到 OpenAI API。請確認網路、防火牆、代理設定後再試。\n")
                    continue
                log.exception("Error in chat_in_terminal:")
                print(f"\n❌ 執行時發生錯誤：{e}\n")
                continue

            finally:
                if not shutting_down and not self._closed:
                    self.dispatch(events.RoundComplete(session_id=self.session_id))
                    await self.drain()
                    await self.logger.write_state()
                    self.visualizer.render("agent_tree", view=True)



    async def query(self, query: str) -> AsyncIterable[events.BaseEvent]:
        """Run one round with the given query.

        Yields all loggable events from the app (i.e. no stream deltas) during the query. To get only messages
        from the root, filter for `events.RootMessage`.
        """
        await self.ensure_init()

        # register a new listener which passes events into a local queue
        q = asyncio.Queue()
        self.add_listener(q.put)

        # submit query to the root agent to run in bg
        async def _task():
            try:
                self.dispatch(events.SendMessage(content=query))
                planned_query = self.prepare_task_prompt(query)
                async for _ in self.root_agent.full_round(planned_query):
                    pass
            finally:
                self.dispatch(events.RoundComplete(session_id=self.session_id))
                await self.drain()
                await self.logger.write_state()  # autosave

        task = asyncio.create_task(_task())

        # yield from the q until we get a RoundComplete
        while True:
            event = await q.get()
            if event.__log_event__:
                yield event
            if event.type == "round_complete":
                break

        # ensure task is completed and cleanup
        await task
        self.remove_listener(q.put)

    # === events ===
    def add_listener(self, callback: Callable[[events.BaseEvent], Awaitable[Any]]):
        """
        Add a listener which is called for every event dispatched by the system.
        The listener must be an asynchronous function that takes in an event in a single argument.
        """
        self.listeners.append(callback)

    def remove_listener(self, callback):
        """Remove a listener added by :meth:`add_listener`."""
        self.listeners.remove(callback)

    # async def _dispatch_task(self):
    #     while True:
    #         # noinspection PyBroadException
    #         try:
    #             event = await self.event_queue.get()
    #             # get listeners, call them
    #             await asyncio.gather(*(callback(event) for callback in self.listeners), return_exceptions=True)
    #         except Exception:
    #             log.exception("Exception when dispatching event:")
    #         finally:
    #             self.event_queue.task_done()
    async def _dispatch_task(self):
        while True:
            try:
                event = await self.event_queue.get()
            except asyncio.CancelledError:
                break  # 直接跳出 loop
            else:
                try:
                    await asyncio.gather(*(callback(event) for callback in self.listeners), return_exceptions=True)
                finally:
                    self.event_queue.task_done()

    def dispatch(self, event: events.BaseEvent):
        """Dispatch an event to all listeners.
        Technically this just adds it to a queue and then an async background task dispatches it."""
        self.event_queue.put_nowait(event)

    async def drain(self):
        """Wait until all events have finished processing."""
        await self.event_queue.join()

    # --- agent lifecycle ---
    def on_agent_creation(self, ai: BaseAgent):
        """Register a new agent, handle parent-child bookkeeping, and dispatch an AgentSpawn event."""
        self.agents[ai.id] = ai
        self.visualizer.add_node(ai.id, label=self.agent_visual_label(ai), depth=ai.depth)
        if ai.parent:
            ai.parent.children[ai.id] = ai
            self.add_agent_edge(ai.parent.id, ai.id, "delegation")
        self.dispatch(events.AgentSpawn.from_agent(ai))

    def agent_visual_label(self, ai: BaseAgent) -> str:
        role = getattr(ai, "role_name", None)
        role_line = f"\n{role}" if role and role != ai.name else ""
        return f"{ai.name}{role_line}\ndepth={ai.depth}"

    # === resources + app lifecycle ===
    async def create_title_listener(self, event):
        """A listener that generates a conversation title after 4 root message events."""
        if (
            self.title is None
            and isinstance(event, events.RootMessage)
            and self.logger.event_count["root_message"] >= 4
            and event.msg.role == ChatRole.ASSISTANT
            and event.msg.content
        ):
            self.title = "..."  # prevent another message from generating a title
            try:
                self.title = await generate_conversation_title(self.root_agent)
                self.dispatch(events.SessionMetaUpdate(title=self.title))
            except Exception:
                log.exception("Could not generate conversation title:")
                self.title = None
            finally:
                self.remove_listener(self.create_title_listener)

    async def close(self):
        """Clean up all the app resources."""
        if self._closed:
            return
        self._closed = True

        if self.dispatch_task is not None and not self.dispatch_task.done():
            self.dispatch(events.SessionClose(session_id=self.session_id))
            await self.drain()
        if self.dispatch_task is not None:
            self.dispatch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.dispatch_task

        close_tasks = [self.logger.close()]
        if self.root_agent is not None:
            close_tasks.append(self.root_agent.close())
        close_tasks.extend(child.close() for child in list(self.agents.values()) if child is not self.root_agent)
        await asyncio.gather(*close_tasks, return_exceptions=True)

        
class TreeVisualizer:
    def __init__(self):
        self.graph = self._new_graph()
        self.nodes = {}
        self.edge_records = []
        self.edges = set()
        self.warned_missing_graphviz = False

    def _new_graph(self):
        return Digraph(comment="Recursive Agent Tree")

    def add_node(self, name: str, label: str = None, depth: int | None = None):
        if label is None:
            label = name
        self.nodes[name] = {"label": label, "depth": depth}
        self.graph.node(name, label=label)

    def add_edge(self, parent: str, child: str, edge_type: str = "delegation", **attrs):
        attrs = {key: value for key, value in attrs.items() if value is not None}
        if edge_type == "peer":
            attrs.setdefault("label", "peer")
            attrs.setdefault("style", "dashed")
            attrs.setdefault("dir", "both")
        elif edge_type in ("reuse", "handoff", "review"):
            attrs.setdefault("label", edge_type)
            attrs.setdefault("style", "dashed")
        edge = (parent, child, edge_type, attrs.get("label"))
        if edge not in self.edges:
            self.graph.edge(parent, child, **attrs)
            self.edges.add(edge)
            self.edge_records.append({
                "source": parent,
                "target": child,
                "edge_type": edge_type,
                "attrs": attrs,
            })

    def render(self, output_path="agent_tree", view=True):
        if GRAPHVIZ_IMPORT_ERROR:
            if not self.warned_missing_graphviz:
                print(
                    "\n⚠️ 未安裝 Python graphviz 套件，所以沒有產生 agent_tree.png。\n"
                    "請執行：python -m pip install graphviz\n"
                )
                self.warned_missing_graphviz = True
            return None
        self.graph.render(output_path, format="png", view=view)
