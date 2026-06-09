import contextlib
import json
import logging
import os
import pathlib
import time
from collections import Counter
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import events
from config import DEFAULT_LOG_DIR
from state import RunState
from utils import read_jsonl

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    trace = None
    Status = None
    StatusCode = None


if TYPE_CHECKING:
    from .app import AutoAgentSystem


log = logging.getLogger(__name__)


class EventLogger:
    def __init__(self, app: "AutoAgentSystem", session_id: str, log_dir: pathlib.Path = None, clear_existing_log: bool = False):
        self.app = app
        self.session_id = session_id
        self.last_modified = time.time()
        self.log_dir = log_dir or (DEFAULT_LOG_DIR / session_id)
        self.clear_existing_log = clear_existing_log

        self.aof_path = self.log_dir / "events.jsonl"
        self.state_path = self.log_dir / "state.json"

        self.event_count = Counter()
        self._suppress_flag = 0

    @cached_property
    def event_file(self):
        # we use a cached property here to only lazily create the log dir if we need it
        self.log_dir.mkdir(exist_ok=True, parents=True)

        if self.clear_existing_log:
            return open(self.aof_path, "w", buffering=1, encoding="utf-8")

        if self.aof_path.exists():
            existing_events = read_jsonl(self.aof_path)
            self.event_count = Counter(event["type"] for event in existing_events)
        return open(self.aof_path, "a", buffering=1, encoding="utf-8")

    async def log_event(self, event: events.BaseEvent):
        if self._suppress_flag:
            return
        if not event.__log_event__:
            return
        self.last_modified = time.time()
        # since this is a synch operation we don't need a lock here (though it is thread-unsafe)
        self.event_file.write(event.model_dump_json())
        self.event_file.write("\n")
        self.event_count[event.type] += 1

    async def write_state(self):
        """Write the full state of the app to the state file, with a basic checksum against the AOF to check validity"""
        if self._suppress_flag:
            return
        self.log_dir.mkdir(exist_ok=True, parents=True)
        state = [ai.get_save_state().model_dump(mode="json") for ai in self.app.agents.values()]
        data = {
            "id": self.session_id,
            "title": self.app.title,
            "last_modified": self.last_modified,
            "n_events": self.event_count.total(),
            "state": state,
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    async def close(self):
        # if we haven't done anything, don't write anything
        if not self.event_count.total():
            return
        await self.write_state()
        self.event_file.close()

    @contextlib.contextmanager
    def suppress_logs(self):
        """Don't dispatch any events while in this body."""
        self._suppress_flag += 1
        try:
            yield
        finally:
            self._suppress_flag -= 1


class OtelEventLogger(EventLogger):
    """Event logger that mirrors agent lifecycle events to OpenTelemetry spans."""

    def __init__(self, *args, auto_configure: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if auto_configure:
            configure_otel_from_env()
        self.tracer = trace.get_tracer("auto-agent-system") if trace else None
        self._round_active = False
        self._round_index = 0
        self._spans: dict[str, Any] = {}
        self._span_depths: dict[str, int] = {}
        self._token_usage = defaultdict(lambda: {"prompt": 0, "completion": 0})

    async def log_event(self, event: events.BaseEvent):
        await super().log_event(event)
        if self._suppress_flag or not event.__log_event__:
            return
        self._record_otel_event(event)

    async def close(self):
        self._end_all_spans()
        await super().close()

    def _record_otel_event(self, event: events.BaseEvent):
        if not self.tracer:
            return

        if isinstance(event, events.SendMessage):
            self._start_round(event)
            return

        if isinstance(event, events.AgentSpawn):
            self._remember_agent(event)
            if self._round_active:
                span = self._ensure_agent_span(
                    event.id,
                    name=event.name,
                    depth=event.depth,
                    parent_id=event.parent,
                    extra_attributes={
                        "agent.engine_type": event.engine_type,
                        "agent.state": event.state.value,
                    },
                )
                _add_event(span, "agent_spawn", {"agent.name": event.name})
            return

        if isinstance(event, events.AgentDelegated):
            if not self._round_active:
                self._round_active = True
            parent_span = self._ensure_agent_span(event.parent_id)
            child_span = self._ensure_agent_span(event.child_id, parent_id=event.parent_id)
            attrs = {
                "agent.parent_id": event.parent_id,
                "agent.child_id": event.child_id,
                "delegation.instructions_preview": _preview(event.instructions),
            }
            _add_event(parent_span, "agent_delegated", attrs)
            _add_event(child_span, "delegated", attrs)
            return

        if isinstance(event, events.AgentStateChange):
            if self._round_active or event.state != RunState.STOPPED:
                span = self._ensure_agent_span(event.id)
                if span:
                    span.set_attribute("agent.state", event.state.value)
                    _add_event(span, "state_change", {"agent.state": event.state.value})
                    if event.state == RunState.ERRORED:
                        _set_error_status(span, "Agent entered errored state")
            return

        if isinstance(event, events.TokensUsed):
            span = self._ensure_agent_span(event.id)
            if span:
                usage = self._token_usage[event.id]
                usage["prompt"] += event.prompt_tokens
                usage["completion"] += event.completion_tokens
                span.set_attribute("llm.usage.prompt_tokens", usage["prompt"])
                span.set_attribute("llm.usage.completion_tokens", usage["completion"])
                span.set_attribute("llm.usage.total_tokens", usage["prompt"] + usage["completion"])
                _add_event(
                    span,
                    "tokens_used",
                    {
                        "llm.usage.prompt_tokens": event.prompt_tokens,
                        "llm.usage.completion_tokens": event.completion_tokens,
                    },
                )
            return

        if isinstance(event, events.Error):
            for span in self._spans.values():
                _add_event(span, "error", {"error.message": event.msg})
                _set_error_status(span, event.msg)
            return

        if isinstance(event, events.RoundComplete):
            for span in self._spans.values():
                _add_event(span, "round_complete", {"session.id": event.session_id})
            self._end_all_spans()

    def _start_round(self, event: events.SendMessage):
        self._end_all_spans()
        self._round_active = True
        self._round_index += 1

        root_agent = getattr(self.app, "root_agent", None)
        if root_agent is None:
            return

        root_span = self._ensure_agent_span(
            root_agent.id,
            name=root_agent.name,
            depth=root_agent.depth,
            parent_id=None,
            extra_attributes={
                "round.index": self._round_index,
                "user.prompt.preview": _preview(event.content),
            },
        )
        _add_event(root_span, "user_message", {"user.prompt.preview": _preview(event.content)})

    def _remember_agent(self, event: events.AgentSpawn):
        self._span_depths[event.id] = event.depth

    def _ensure_agent_span(
        self,
        agent_id: str,
        *,
        name: str | None = None,
        depth: int | None = None,
        parent_id: str | None = None,
        extra_attributes: dict[str, Any] | None = None,
    ):
        if not self.tracer or not agent_id:
            return None
        if agent_id in self._spans:
            return self._spans[agent_id]

        agent = getattr(self.app, "agents", {}).get(agent_id)
        if agent is not None:
            name = name or agent.name
            depth = agent.depth if depth is None else depth
            parent_id = parent_id or (agent.parent.id if agent.parent else None)

        if parent_id and parent_id not in self._spans:
            self._ensure_agent_span(parent_id)

        parent_span = self._spans.get(parent_id)
        parent_context = trace.set_span_in_context(parent_span) if parent_span else None
        attributes = {
            "session.id": self.session_id,
            "agent.id": agent_id,
            "agent.name": name or agent_id,
        }
        if depth is not None:
            attributes["agent.depth"] = depth
            self._span_depths[agent_id] = depth
        if parent_id:
            attributes["agent.parent_id"] = parent_id
        if extra_attributes:
            attributes.update(extra_attributes)

        span_name = f"agent:{name or agent_id}"
        span = self.tracer.start_span(span_name, context=parent_context, attributes=attributes)
        self._spans[agent_id] = span
        return span

    def _end_all_spans(self):
        if not self._spans:
            self._round_active = False
            return

        ordered_agent_ids = sorted(self._spans, key=lambda agent_id: self._span_depths.get(agent_id, 0), reverse=True)
        for agent_id in ordered_agent_ids:
            self._spans[agent_id].end()
        _force_flush_traces()
        self._spans.clear()
        self._token_usage.clear()
        self._round_active = False


def configure_otel_from_env():
    """Configure a simple OTLP exporter when AUTO_AGENT_OTEL_ENABLED is set."""
    if not _truthy(os.getenv("AUTO_AGENT_OTEL_ENABLED")) or trace is None:
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        log.warning("OpenTelemetry is enabled, but SDK/exporter packages are not installed.")
        return

    provider = trace.get_tracer_provider()
    if provider.__class__.__module__.startswith("opentelemetry.sdk."):
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "auto-agent-system")
    tracer_provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter_name = os.getenv("AUTO_AGENT_OTEL_EXPORTER", "otlp").lower()
    if exporter_name == "console":
        exporter = ConsoleSpanExporter()
    else:
        endpoint, insecure = _otel_grpc_endpoint_from_env()
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _preview(text: str, limit: int = 200) -> str:
    return text[:limit]


def _add_event(span, name: str, attributes: dict[str, Any] | None = None):
    if span:
        span.add_event(name, attributes=attributes or {})


def _set_error_status(span, description: str):
    if span and Status and StatusCode:
        span.set_status(Status(StatusCode.ERROR, description))


def _otel_grpc_endpoint_from_env() -> tuple[str, bool]:
    raw_endpoint = (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or "localhost:4317"
    )
    parsed = urlparse(raw_endpoint)
    if parsed.scheme and parsed.netloc:
        endpoint = parsed.netloc
        insecure = parsed.scheme == "http"
    else:
        endpoint = raw_endpoint
        insecure = True

    insecure = insecure or _truthy(os.getenv("OTEL_EXPORTER_OTLP_INSECURE"))
    return endpoint.rstrip("/"), insecure


def otel_status_summary() -> str:
    if not _truthy(os.getenv("AUTO_AGENT_OTEL_ENABLED")):
        return "disabled"
    if trace is None:
        return "enabled, but opentelemetry packages are missing"
    endpoint, insecure = _otel_grpc_endpoint_from_env()
    return f"enabled -> {endpoint} ({'insecure' if insecure else 'secure'})"


def _force_flush_traces():
    if not trace:
        return
    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if force_flush:
        force_flush()
