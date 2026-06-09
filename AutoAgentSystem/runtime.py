import asyncio
import inspect
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, AsyncIterable, Callable, get_args, get_origin

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class FunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    id: str | None = None
    function: FunctionCall | None = None


class ChatMessage(BaseModel):
    role: ChatRole
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @property
    def text(self) -> str | None:
        return self.content

    @text.setter
    def text(self, value: str | None):
        self.content = value

    @classmethod
    def system(cls, content: str):
        return cls(role=ChatRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str):
        return cls(role=ChatRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[ToolCall] | None = None):
        return cls(role=ChatRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def function(cls, name: str, content: Any, tool_call_id: str | None = None):
        return cls(role=ChatRole.FUNCTION, name=name, content=str(content), tool_call_id=tool_call_id)


class BaseCompletion(BaseModel):
    message: ChatMessage
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AIParam:
    def __init__(self, desc: str = ""):
        self.desc = desc


class AIFunction:
    def __init__(
        self,
        func: Callable,
        *,
        name: str | None = None,
        desc: str | None = None,
        auto_retry: bool = True,
        auto_truncate: int | None = None,
        after: ChatRole = ChatRole.ASSISTANT,
        json_schema: dict | None = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.desc = desc or inspect.getdoc(func) or ""
        self.auto_retry = auto_retry
        self.auto_truncate = auto_truncate
        self.after = after
        self.json_schema = json_schema or build_json_schema(func)

    async def __call__(self, **kwargs):
        result = self.func(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def ai_function(
    func: Callable | None = None,
    *,
    name: str | None = None,
    desc: str | None = None,
    auto_retry: bool = True,
    auto_truncate: int | None = None,
    after: ChatRole = ChatRole.ASSISTANT,
):
    def decorator(wrapped: Callable):
        wrapped.__ai_function__ = {
            "name": name,
            "desc": desc,
            "auto_retry": auto_retry,
            "auto_truncate": auto_truncate,
            "after": after,
        }
        return wrapped

    if func is not None:
        return decorator(func)
    return decorator


def build_json_schema(func: Callable) -> dict:
    sig = inspect.signature(func)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        schema, desc = annotation_to_schema(param.annotation)
        if desc:
            schema["description"] = desc
        properties[param_name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def annotation_to_schema(annotation: Any) -> tuple[dict, str | None]:
    desc = None

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        for meta in args[1:]:
            if isinstance(meta, AIParam):
                desc = meta.desc

    origin = get_origin(annotation)
    if annotation in (str, inspect.Parameter.empty):
        schema = {"type": "string"}
    elif annotation is int:
        schema = {"type": "integer"}
    elif annotation is float:
        schema = {"type": "number"}
    elif annotation is bool:
        schema = {"type": "boolean"}
    elif annotation is dict or origin is dict:
        schema = {"type": "object"}
    elif annotation is list or origin is list:
        schema = {"type": "array"}
    else:
        schema = {"type": "string"}

    return schema, desc


@dataclass
class BaseEngine:
    max_context_size: int = 128000

    async def complete(
        self,
        messages: list[ChatMessage],
        functions: dict[str, AIFunction] | None = None,
        **kwargs,
    ) -> BaseCompletion:
        raise NotImplementedError("Engine subclasses must implement complete().")

    async def close(self):
        pass


class OpenAIEngine(BaseEngine):
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1028,
        max_context_size: int = 128000,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(max_context_size=max_context_size)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.extra_kwargs = kwargs
        self._client = None

    def __repr__(self):
        return f"OpenAIEngine(model={self.model!r})"

    @property
    def client(self):
        if self._client is None:
            # SSLKEYLOGFILE is only for TLS debugging. A stale, unwritable path breaks httpx/OpenAI client creation.
            os.environ.pop("SSLKEYLOGFILE", None)
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError("OpenAI SDK is required for OpenAIEngine. Install `openai`.") from exc
            self._client = AsyncOpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
        return self._client

    async def complete(
        self,
        messages: list[ChatMessage],
        functions: dict[str, AIFunction] | None = None,
        **kwargs,
    ) -> BaseCompletion:
        request = {
            "model": self.model,
            "messages": [message_to_openai(m) for m in messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            **self.extra_kwargs,
            **kwargs,
        }
        request = {key: value for key, value in request.items() if value is not None}

        if functions:
            request["tools"] = [function_to_openai_tool(f) for f in functions.values()]

        response = await self.client.chat.completions.create(**request)
        choice = response.choices[0].message
        message = openai_to_message(choice)
        usage = getattr(response, "usage", None)
        return BaseCompletion(
            message=message,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


class StreamManager:
    def __init__(self, stream: AsyncIterable[str | BaseCompletion], role: ChatRole = ChatRole.ASSISTANT):
        self._stream = stream.__aiter__()
        self.role = role
        self._chunks: list[str] = []
        self._completion: BaseCompletion | None = None
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        while not self._done:
            try:
                item = await self._stream.__anext__()
            except StopAsyncIteration:
                self._done = True
                if self._completion is None:
                    self._completion = BaseCompletion(message=ChatMessage.assistant("".join(self._chunks)))
                raise

            if isinstance(item, BaseCompletion):
                self._completion = item
                continue

            token = str(item)
            self._chunks.append(token)
            return token

        raise StopAsyncIteration

    async def completion(self) -> BaseCompletion:
        if not self._done:
            async for _ in self:
                pass
        if self._completion is None:
            self._completion = BaseCompletion(message=ChatMessage.assistant("".join(self._chunks)))
        return self._completion

    async def message(self) -> ChatMessage:
        return (await self.completion()).message


class ChatAgentRuntime:
    def __init__(self, engine: BaseEngine, *, system_prompt: str | None = None, **kwargs):
        self.engine = engine
        self.system_prompt = system_prompt
        self.functions: dict[str, AIFunction] = {}
        self.chat_history: list[ChatMessage] = []
        self.always_included_messages: list[ChatMessage] = []
        if system_prompt is not None:
            self.always_included_messages.append(ChatMessage.system(system_prompt))
        self.always_len = sum(self.message_token_len(m) for m in self.always_included_messages)

    async def get_prompt(self) -> list[ChatMessage]:
        return [*self.always_included_messages, *self.chat_history]

    def message_token_len(self, message: ChatMessage) -> int:
        return max(1, len(message.text or "") // 4)

    async def add_to_history(self, message: ChatMessage):
        self.chat_history.append(message)

    async def add_completion_to_history(self, completion: BaseCompletion):
        await self.add_to_history(completion.message)
        return completion.message

    async def get_model_completion(self, include_functions: bool = True, **kwargs) -> BaseCompletion:
        functions = self.functions if include_functions else None
        return await self.engine.complete(await self.get_prompt(), functions=functions, **kwargs)

    async def get_model_stream(self, include_functions: bool = True, **kwargs):
        yield await self.get_model_completion(include_functions=include_functions, **kwargs)

    async def chat_round(self, prompt: str | ChatMessage, **kwargs) -> ChatMessage:
        msg = prompt if isinstance(prompt, ChatMessage) else ChatMessage.user(prompt)
        await self.add_to_history(msg)
        completion = await self.get_model_completion(**kwargs)
        return await self.add_completion_to_history(completion)

    async def chat_round_str(self, prompt: str, **kwargs) -> str:
        msg = await self.chat_round(prompt, **kwargs)
        return msg.text or ""

    def chat_round_stream(self, prompt: str | ChatMessage, **kwargs) -> StreamManager:
        async def _impl():
            msg = await self.chat_round(prompt, **kwargs)
            if msg.text:
                yield msg.text
            yield BaseCompletion(message=msg)

        return StreamManager(_impl(), role=ChatRole.ASSISTANT)

    async def full_round(self, prompt: str | ChatMessage, max_function_rounds: int = 5, **kwargs):
        msg = prompt if isinstance(prompt, ChatMessage) else ChatMessage.user(prompt)
        await self.add_to_history(msg)

        for _ in range(max_function_rounds):
            completion = await self.get_model_completion(**kwargs)
            assistant_msg = await self.add_completion_to_history(completion)
            yield assistant_msg

            if not assistant_msg.tool_calls:
                return

            await self._handle_tool_calls(assistant_msg.tool_calls)

        # If the model spends every allowed round calling tools, force one final synthesis pass.
        completion = await self.get_model_completion(include_functions=False, **kwargs)
        assistant_msg = await self.add_completion_to_history(completion)
        yield assistant_msg

    async def full_round_stream(self, prompt: str | ChatMessage, **kwargs):
        async for msg in self.full_round(prompt, **kwargs):
            if msg.role != ChatRole.ASSISTANT:
                continue

            async def _impl(message=msg):
                if message.text:
                    yield message.text
                yield BaseCompletion(message=message)

            yield StreamManager(_impl(), role=msg.role)

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]):
        for tool_call in tool_calls:
            if tool_call.function is None:
                continue
            function = self.functions.get(tool_call.function.name)
            if function is None:
                result = f"Unknown tool: {tool_call.function.name}"
            else:
                try:
                    kwargs = json.loads(tool_call.function.arguments or "{}")
                    result = await function(**kwargs)
                except Exception as exc:
                    result = f"Tool {tool_call.function.name} failed: {exc}"
            await self.add_to_history(
                ChatMessage.function(tool_call.function.name, result, tool_call_id=tool_call.id)
            )

    async def cleanup(self):
        pass

    async def close(self):
        if hasattr(self.engine, "close"):
            await self.engine.close()


def message_to_openai(message: ChatMessage) -> dict:
    if message.role == ChatRole.FUNCTION:
        payload = {
            "role": "tool" if message.tool_call_id else "function",
            "content": message.content or "",
        }
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        else:
            payload["name"] = message.name or "function"
        return payload

    payload = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.name:
        payload["name"] = message.name
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name if tc.function else "",
                    "arguments": tc.function.arguments if tc.function else "{}",
                },
            }
            for tc in message.tool_calls
        ]
    return payload


def openai_to_message(message: Any) -> ChatMessage:
    tool_calls = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        tool_calls.append(
            ToolCall(
                id=getattr(tool_call, "id", None),
                function=FunctionCall(
                    name=getattr(function, "name", ""),
                    arguments=getattr(function, "arguments", "{}"),
                ),
            )
        )
    return ChatMessage.assistant(getattr(message, "content", None), tool_calls=tool_calls or None)


def function_to_openai_tool(function: AIFunction) -> dict:
    return {
        "type": "function",
        "function": {
            "name": function.name,
            "description": function.desc,
            "parameters": function.json_schema,
        },
    }
