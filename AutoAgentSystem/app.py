from dotenv import load_dotenv
load_dotenv()

import asyncio
import functools
import logging
import time
import uuid
import events
from eventlogger import EventLogger

from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any, Awaitable, Callable
from weakref import WeakValueDictionary

import kani.exceptions
from kani import ChatRole, chat_in_terminal_async, ChatMessage
from kani.engines import BaseEngine

from base_kani import BaseKani
from delegation.delegate_and_wait import DelegateWait
from delegation.delegate_one import DelegateOne
from kanis import DEFAULT_DELEGATE_PROMPT, DEFAULT_ROOT_PROMPT, create_root_kani
from tool_config import ToolConfigType, validate_tool_configs
from utils import AUTOGENERATE_TITLE, AutogenerateTitle, generate_conversation_title
from graphviz import Digraph

from tools.browsing.impl import Browsing, ArxivSearch

log = logging.getLogger(__name__)


@functools.cache
def default_engine():
    try:
        from kani.engines.openai import OpenAIEngine
    except kani.exceptions.MissingModelDependencies:
        raise ImportError(
            'Default OpenAI engine is not installed. You can either install it using `pip install "kani[openai]"` or'
            " specify the engine to use in your ReDel system."
        )

    return OpenAIEngine(
        model="gpt-4o", # gpt-3.5-turbo
        temperature=0.7,        # 控制隨機性：越低越穩定
        top_p=0.9,              # 控制 nucleus sampling
        max_tokens=1028       # 每次回應最多 token
    )


class AutoAgentSystem:
    """This class represents a single session of a recursive multi-agent system.

    It's responsible for:

    * all delegation configuration options
    * all the spawned kani and their relations within the session
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
        # prompt/kani
        root_system_prompt: str | None = DEFAULT_ROOT_PROMPT,
        root_kani_kwargs: dict = None,
        delegate_system_prompt: str | None = DEFAULT_DELEGATE_PROMPT,
        delegate_kani_kwargs: dict = None,
        # delegation/function calling
        delegation_scheme: type | None = DelegateWait,
        max_delegation_depth: int = 4,
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
        """
        :param root_engine: The engine to use for the root kani. Requires function calling. (default: gpt-4o)
            See :external+kani:doc:`engines` for a list of available engines and their capabilities.
        :param delegate_engine: The engine to use for each delegate kani. Requires function calling. (default: gpt-4o)
            See :external+kani:doc:`engines` for a list of available engines and their capabilities.
        :param root_system_prompt: The system prompt for the root kani. See ``redel.kanis`` for default.
        :param root_kani_kwargs: Additional keyword args to pass to :class:`kani.Kani`.
        :param delegate_system_prompt: The system prompt for the each delegate kani. See ``redel.kanis`` for default.
        :param delegate_kani_kwargs: Additional keyword args to pass to :class:`kani.Kani`.
        :param delegation_scheme: A class that each kani capable of delegation will use to provide the delegation tool.
            See ``redel.delegation`` for examples. Can be ``None`` to disable delegation.
        :param max_delegation_depth: The maximum delegation depth. Kanis created at this depth will not inherit from the
            ``delegation_scheme`` class.
        :param tool_configs: A mapping of tool mixin classes to their configurations (see :class:`.ToolConfig`).
        :param root_has_tools: Whether the root kani should have access to the configured tools (default
            False).
        :param title: The title of this session. Set to ``redel.AUTOGENERATE_TITLE`` to automatically generate one
            (default), or ``None`` to disable title generation.
        :param log_dir: A path to a directory to save logs for this session. Defaults to
            ``$REDEL_HOME/instances/{session_id}/`` (default ``~/.redel/instances/{session_id}``).
        :param clear_existing_log: If the log directory has existing events, clear them before writing new events.
            Otherwise, append to existing events.
        :param session_id: The ID of this session. Generally this should not be set manually; it is used for loading
            previous states.
        """
        if root_engine is None:
            root_engine = default_engine()
        if delegate_engine is None:
            delegate_engine = default_engine()
        if root_kani_kwargs is None:
            root_kani_kwargs = {}
        if delegate_kani_kwargs is None:
            delegate_kani_kwargs = {}
        if tool_configs is None:
            tool_configs = {}

        validate_tool_configs(tool_configs)

        # engines
        self.root_engine = root_engine
        self.delegate_engine = delegate_engine
        # prompt/kani
        self.root_system_prompt = root_system_prompt
        self.root_kani_kwargs = root_kani_kwargs
        self.delegate_system_prompt = delegate_system_prompt
        self.delegate_kani_kwargs = delegate_kani_kwargs
        # delegation/function calling
        self.delegation_scheme = delegation_scheme
        self.max_delegation_depth = max_delegation_depth
        self.tool_configs = tool_configs
        # 註冊工具
        self.tool_configs.update({
            Browsing: {
                "always_include": True,
                "kwargs": {}
            },
            ArxivSearch: {
                "always_include": True,
                "kwargs": {}
            }
        })
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
        self.logger = EventLogger(self, self.session_id, log_dir=log_dir, clear_existing_log=clear_existing_log)
        self.add_listener(self.logger.log_event)
        # kanis
        self.kanis = WeakValueDictionary()
        self.root_kani = None

    def get_config(self, **kwargs):
        """
        Get a dictionary with arguments suitable for passing to a ReDel constructor to create a new instance with
        mostly the same configuration.

        By default, the title, log_dir, and session_id will not be copied. Explicitly set these as keyword
        arguments if you want to copy them.

        Pass keyword arguments to override existing configuration options (valid arguments are same as constructor).
        """
        config = {
            "root_engine": self.root_engine,
            "delegate_engine": self.delegate_engine,
            "root_system_prompt": self.root_system_prompt,
            "root_kani_kwargs": self.root_kani_kwargs,
            "delegate_system_prompt": self.delegate_system_prompt,
            "delegate_kani_kwargs": self.delegate_kani_kwargs,
            "delegation_scheme": self.delegation_scheme,
            "max_delegation_depth": self.max_delegation_depth,
            "tool_configs": self.tool_configs,
            "root_has_tools": self.root_has_tools,
        }
        config.update(kwargs)
        return config

    async def ensure_init(self):
        """Called at least once before any messaging happens. Used to do async init. Must be idempotent."""
        async with self._init_lock:  # lock in case of parallel calls - no double creation
            if self.root_kani is None:
                self.root_kani = await create_root_kani(
                    self.root_engine,
                    # create_root_kani args
                    app=self,
                    delegation_scheme=self.delegation_scheme,
                    tool_configs=self.tool_configs,
                    root_has_tools=self.root_has_tools,
                    # BaseKani args
                    name="root",
                    # Kani args
                    system_prompt=self.root_system_prompt,
                    **self.root_kani_kwargs,
                )
            if self.dispatch_task is None:
                self.dispatch_task = asyncio.create_task(
                    self._dispatch_task(), name=f"redel-dispatch-{self.session_id}"
                )
        return self.root_kani

    # === entrypoints ===
    async def chat_from_queue(self, q: asyncio.Queue):
        """Get chat messages from a provided queue. Used internally in the visualization server."""
        await self.ensure_init()
        while True:
            # main loop
            try:
                user_msg = await q.get()
                log.info(f"Message from queue: {user_msg.content!r}")
                async for stream in self.root_kani.full_round_stream(user_msg.content):
                    msg = await stream.message()
                    if msg.role == ChatRole.ASSISTANT:
                        log.info(f"AI: {msg}")
            except Exception:
                log.exception("Error in chat_from_queue:")
            finally:
                self.dispatch(events.RoundComplete(session_id=self.session_id))
                await self.logger.write_state()  # autosave

    # async def chat_in_terminal(self):
    #     """Chat with the defined system in the terminal. Prints function calls and root messages to the terminal."""
    #     await self.ensure_init()
    #     while True:
    #         try:
    #             await chat_in_terminal_async(self.root_kani, show_function_args=True, rounds=1)
    #             self.visualizer.render("agent_tree", view=True)
    #         except KeyboardInterrupt:
    #             await self.close()
    #         finally:
    #             self.dispatch(events.RoundComplete(session_id=self.session_id))
    #             await self.logger.write_state()  # autosave
    
    # async def chat_in_terminal(self):
    #     await self.ensure_init()
    #     while True:
    #         try:
    #             user_input = input("USER: ")
    #             if user_input.strip().lower() in ("exit", "quit"):
    #                 print("👋 使用者中斷。再見！")
    #                 await self.close()
    #                 break
    
    #             async for stream in self.root_kani.full_round_stream(user_input):
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
            try:
                user_input = input("USER: ")
                if user_input.strip().lower() in ("exit", "quit"):
                    print("👋 使用者中斷。再見！")
                    await self.close()
                    break
    
                async for stream in self.root_kani.full_round_stream(user_input):
                    print("AI:", end="", flush=True)
                    content = ""
                    async for token in stream:
                        print(token, end="", flush=True)
                        content += token
                    print()  # 換行結尾
    
                    msg = await stream.message()
                    if msg.tool_calls:
                        print(f"\n[🛠️ Tool Call]: {msg.tool_calls}")
    
            except KeyboardInterrupt:
                print("\n👋 使用者中斷（Ctrl+C）。再見！")
                await self.close()
                break
            finally:
                self.dispatch(events.RoundComplete(session_id=self.session_id))
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

        # submit query to the kani to run in bg
        async def _task():
            try:
                async for _ in self.root_kani.full_round(query):
                    pass
            finally:
                self.dispatch(events.RoundComplete(session_id=self.session_id))
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

    # --- kani lifecycle ---
    def on_kani_creation(self, ai: BaseKani):
        """Called by the redel kani constructor.
        Registers a new kani in the app, handles parent-child bookkeeping, and dispatches a KaniSpawn event."""
        self.kanis[ai.id] = ai
        self.visualizer.add_node(ai.name, label=ai.name)
        if ai.parent:
            ai.parent.children[ai.id] = ai
            self.visualizer.add_edge(ai.parent.name, ai.name)
        self.dispatch(events.KaniSpawn.from_kani(ai))

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
                self.title = await generate_conversation_title(self.root_kani)
                self.dispatch(events.SessionMetaUpdate(title=self.title))
            except Exception:
                log.exception("Could not generate conversation title:")
                self.title = None
            finally:
                self.remove_listener(self.create_title_listener)

    async def close(self):
        """Clean up all the app resources."""
        self.dispatch(events.SessionClose(session_id=self.session_id))
        await self.drain()
        if self.dispatch_task is not None:
            self.dispatch_task.cancel()
        await asyncio.gather(
            self.logger.close(),
            self.root_kani.close(),
            *(child.close() for child in self.kanis.values()),
        )

        
class TreeVisualizer:
    def __init__(self):
        self.graph = Digraph(comment="Recursive Agent Tree")
        self.edges = set()

    def add_node(self, name: str, label: str = None):
        if label is None:
            label = name
        self.graph.node(name, label=label)

    def add_edge(self, parent: str, child: str):
        edge = (parent, child)
        if edge not in self.edges:
            self.graph.edge(parent, child)
            self.edges.add(edge)

    def render(self, output_path="agent_tree", view=True):
        self.graph.render(output_path, format="png", view=view)