import asyncio
import logging
import re
from typing import Annotated

from runtime import AIParam, ChatRole, ai_function
try:
    from rapidfuzz import fuzz
except ImportError:
    from difflib import SequenceMatcher

    class fuzz:
        @staticmethod
        def ratio(a, b):
            return int(SequenceMatcher(None, a, b).ratio() * 100)

try:
    import openai
except ImportError:
    class _OpenAIShim:
        class RateLimitError(Exception):
            pass

    openai = _OpenAIShim()

import events
from state import RunState
from delegation._base import DelegationBase

log = logging.getLogger(__name__)


class DelegateWait(DelegationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helpers = {}
        self.helper_futures = {}
        self.helper_aliases = {}
        self.completed_results = {}
        self.completed_task_results = {}
        self.last_wait_all_result = None
        self.request_semaphore = asyncio.Semaphore(3)

    def normalize_name(self, name: str | None):
        if not name:
            return None
        return " ".join(name.strip().lower().split())

    def normalize_task(self, instructions: str):
        return " ".join(instructions.strip().lower().split())

    def infer_role_name(self, who: str | None, instructions: str):
        if who and self.resolve_helper_name(who) is None:
            return " ".join(who.strip().split())

        match = re.match(r"\s*(?:you are|as)\s+(?:the\s+)?([^.,:;]+)", instructions, flags=re.IGNORECASE)
        if not match:
            return None

        role = " ".join(match.group(1).strip().split())
        role_markers = ("agent", "analyst", "engineer", "expert", "lead", "researcher", "specialist")
        if any(marker in role.lower() for marker in role_markers):
            return role
        return None

    def reuse_key_for(self, role_name: str | None):
        return self.normalize_name(role_name)

    def remember_alias(self, alias: str | None, helper_name: str):
        alias_key = self.normalize_name(alias)
        if alias_key:
            self.helper_aliases[alias_key] = helper_name
        self.helper_aliases[self.normalize_name(helper_name)] = helper_name

    def resolve_helper_name(self, who: str | None):
        if not who:
            return None
        if who in self.helpers:
            return who
        return self.helper_aliases.get(self.normalize_name(who))

    def find_duplicate_task(self, instructions: str):
        normalized = self.normalize_task(instructions)
        for task in self.app.global_task_log:
            existing = self.normalize_task(task["task"])
            if normalized == existing or fuzz.ratio(normalized, existing) >= 98:
                return task
        return None

    def find_completed_result(self, instructions: str, helper_name: str | None = None):
        normalized = self.normalize_task(instructions)
        if helper_name and helper_name in self.completed_results:
            helper = self.helpers.get(helper_name)
            previous_task = self.normalize_task(getattr(helper, "task_description", "") or "")
            if previous_task and (normalized == previous_task or fuzz.ratio(normalized, previous_task) >= 88):
                return helper_name, self.completed_results[helper_name]

        for task_text, result in self.completed_task_results.items():
            if normalized == task_text or fuzz.ratio(normalized, task_text) >= 98:
                return result
        return None

    @ai_function(
        desc=(
            "Delegate a bounded role-specific subtask to another agent. Use this to build the topology you selected: "
            "flat fan-out, hierarchical team leads, pipeline phases, or review/debate agents. Reuse the same stable "
            "who value for the same specialist role instead of creating a new agent of the same type. Returns immediately."
        )
    )
    async def delegate(
        self,
        instructions: Annotated[
            str,
            AIParam(
                "Detailed, scoped instructions for the helper. Include its role, boundaries, expected output, and "
                "whether it may further delegate if its subtask is still broad."
            ),
        ],
        who: Annotated[
            str,
            AIParam(
                "Stable specialist role or existing helper name, such as 'System Architecture Lead'. Reuse the same "
                "value when a later subtask belongs to the same role/type."
            ),
        ] = None,
    ):
        log.info(f"Delegated with instructions: {instructions}")

        helper_name = self.resolve_helper_name(who)
        role_name = self.infer_role_name(who, instructions)
        reuse_key = self.reuse_key_for(role_name)
        cached_result = self.find_completed_result(instructions, helper_name)
        if cached_result:
            cached_helper, cached_text = cached_result
            return f"{cached_helper}:{cached_text}"

        duplicate_task = self.find_duplicate_task(instructions)
        if duplicate_task:
            duplicate_helper = duplicate_task.get("agent")
            duplicate_result = duplicate_task.get("result")
            if duplicate_result:
                return f"{duplicate_helper}:{duplicate_result}"
            if duplicate_helper in self.helper_futures:
                return f"{duplicate_helper!r} is already working on a similar task. Use wait(until='all') to collect results."
            if duplicate_helper in self.completed_results:
                return f"{duplicate_helper}:{self.completed_results[duplicate_helper]}"
            return "⚠️ 類似任務已經被分派過了，請等待或統整既有結果。"

        if getattr(self.agent, "depth", 0) >= 4:
            return f"⚠️ 已達最大遞迴層數（4）。請在目前層級內完成任務。"

        if len(self.helpers) >= 4 and "__AUTO_WAITING__" not in self.helper_futures:
            print(f"\n🔁 已建立 {len(self.helpers)} 個 sub-agent，啟動自動統整機制...\n")
            self.helper_futures["__AUTO_WAITING__"] = asyncio.create_task(self._auto_wait_all())

        if self.agent.last_user_message and fuzz.ratio(instructions, self.agent.last_user_message.content) > 80:
            return "You shouldn't delegate the entire task to a helper. Break it into smaller parts if necessary."

        if helper_name and helper_name in self.helpers:
            if helper_name in self.helper_futures:
                return f"{helper_name!r} is still working. Wait or delegate to someone else."
            helper = self.helpers[helper_name]
            self.app.dispatch(events.AgentDelegated(
                parent_id=self.agent.id,
                child_id=helper.id,
                parent_message_idx=len(self.agent.chat_history) - 1,
                child_message_idx=len(helper.chat_history),
                instructions=instructions,
            ))
        else:
            helper = self.app.get_reusable_agent(reuse_key)
            if helper is self.agent:
                return (
                    f"You are already the {role_name or 'requested specialist'}. "
                    "Handle this subtask directly instead of delegating to yourself."
                )

            if helper is not None:
                if getattr(helper, "state", RunState.STOPPED) != RunState.STOPPED:
                    return f"{helper.name!r} is already handling {role_name or 'a similar role'}. Wait before reusing it."
                self.helpers[helper.name] = helper
                self.remember_alias(who or role_name, helper.name)
                self.remember_alias(role_name, helper.name)
                self.app.on_agent_reuse(self.agent, helper)
                self.app.dispatch(events.AgentDelegated(
                    parent_id=self.agent.id,
                    child_id=helper.id,
                    parent_message_idx=len(self.agent.chat_history) - 1,
                    child_message_idx=len(helper.chat_history),
                    instructions=instructions,
                ))
                print(f"\n[♻️ 任務沿用] Agent: {helper.name} ({role_name or 'same role'})")
                print("📄 新任務：")
                print(instructions)
                print("-" * 40 + "\n")
            else:
                helper = await self.create_delegate_agent(instructions, role_name=role_name, reuse_key=reuse_key)
                self.helpers[helper.name] = helper
                self.remember_alias(who or role_name, helper.name)
                self.remember_alias(role_name, helper.name)
                print(f"\n[✅ 任務指派] Agent: {helper.name}")
                print("📄 被指派的任務：")
                print(instructions)
                print("-" * 40 + "\n")
            helper.task_description = instructions
            self.app.global_task_log.append({
                "agent": helper.name,
                "agent_id": helper.id,
                "reuse_key": reuse_key,
                "role": role_name,
                "task": instructions,
                "status": "assigned"
            })

        return await self._task_with_helper(helper, instructions)

    async def _task_with_helper(self, helper, instructions):
        async def internal():
            try:
                result = []
                retries = 1
                delay = 2
                for attempt in range(retries):
                    try:
                        async with self.request_semaphore:
                            log.info(f"Starting full_round_stream for {helper.name}")
                            content = ""
                            async for stream in helper.full_round_stream(instructions):
                                async for token in stream:
                                    content += token
                            result.append(content)
                        break
                    except openai.RateLimitError:
                        log.warning(f"[{helper.name}] Rate limit. Retry {attempt + 1}/{retries} after {delay}s")
                        await asyncio.sleep(delay)
                        delay *= 2
                else:
                    return f"[{helper.name}] was rate limited. Try again later.", helper.name

                output = "\n".join(result)
                for entry in self.app.global_task_log:
                    if entry["agent"] == helper.name and entry["task"] == instructions:
                        entry["status"] = "completed"
                        entry["result"] = output

                self.completed_results[helper.name] = output
                self.completed_task_results[self.normalize_task(instructions)] = (helper.name, output)
                await helper.cleanup()
                return output, helper.name
            except Exception as e:
                log.exception(f"{helper.name}-{helper.depth} encountered an exception!")
                for entry in self.app.global_task_log:
                    if entry["agent"] == helper.name and entry["task"] == instructions:
                        entry["status"] = f"failed: {e}"
                try:
                    role_name = getattr(helper, "role_name", None)
                    reuse_key = getattr(helper, "reuse_key", None)
                    new_helper = await self.create_delegate_agent(
                        instructions,
                        role_name=role_name,
                        reuse_key=reuse_key,
                    )
                    new_helper.task_description = instructions
                    self.helpers[new_helper.name] = new_helper
                    self.remember_alias(role_name, new_helper.name)
                    print(f"\n[🔁 任務重新委派] 由 {helper.name} 改為 {new_helper.name}")
                    self.app.global_task_log.append({
                        "agent": new_helper.name,
                        "agent_id": new_helper.id,
                        "reuse_key": reuse_key,
                        "role": role_name,
                        "task": instructions,
                        "status": "reassigned"
                    })
                    return await self._task_with_helper(new_helper, instructions)
                except Exception as retry_error:
                    log.exception(f"Failed to reassign task to new agent after failure: {retry_error}")
                    return f"{helper.name} failed and reassignment also failed: {retry_error}", helper.name

        self.helper_futures[helper.name] = asyncio.create_task(internal())
        return f"{helper.name!r} is helping you with this request."

    async def _auto_wait_all(self):
        active_futures = [fut for name, fut in self.helper_futures.items() if name != "__AUTO_WAITING__"]

        if not active_futures:
            print("⚠️ 沒有任何有效的子任務正在執行，跳過自動統整。")
            return "⚠️ 沒有任何有效的子任務正在執行，跳過自動統整。"

        with self.agent.run_state(RunState.WAITING):
            done, _ = await asyncio.wait(active_futures, return_when=asyncio.ALL_COMPLETED)

        results = []
        for future in done:
            if future.done():
                result, helper_name = future.result()
                results.append(f"{helper_name}:{result}")

        self.helper_futures.clear()

        print("\n📦 所有 sub-agent 統整回報如下：\n")
        print("\n\n=====\n\n".join(results))
        return "\n\n=====\n\n".join(results)

    @ai_function(desc='Wait for one or all sub-agents to finish their tasks and return results.', auto_truncate=6000)
    async def wait(
        self,
        until: Annotated[str, AIParam('Name of the helper. Use "next" or "all".')],
    ):
        resolved_until = self.resolve_helper_name(until) or until
        if (
            until not in ("next", "all")
            and resolved_until not in self.helper_futures
            and resolved_until not in self.completed_results
        ):
            return 'The "until" param must be a running helper name, "next", or "all".'

        if until == "next":
            active_futures = {name: future for name, future in self.helper_futures.items() if name != "__AUTO_WAITING__"}
            if not active_futures:
                if self.last_wait_all_result:
                    return self.last_wait_all_result
                return "There are no active sub-agents to wait for."
            with self.agent.run_state(RunState.WAITING):
                done, _ = await asyncio.wait(active_futures.values(), return_when=asyncio.FIRST_COMPLETED)
            future = done.pop()
            try:
                res = future.result()
                if isinstance(res, tuple) and len(res) == 2:
                    result, helper_name = res
                else:
                    result = str(res)
                    helper_name = "unknown"
            except Exception as e:
                result = f"Exception: {e}"
                helper_name = "unknown"
            self.helper_futures.pop(helper_name, None)
            return f"{helper_name}:{result}"

        elif until == "all":
            active_futures = {name: future for name, future in self.helper_futures.items() if name != "__AUTO_WAITING__"}
            if not active_futures:
                if self.last_wait_all_result:
                    return self.last_wait_all_result
                return "No sub-agents were successfully assigned. Please try delegating again."
            with self.agent.run_state(RunState.WAITING):
                done, _ = await asyncio.wait(active_futures.values(), return_when=asyncio.ALL_COMPLETED)
            results = []
            for future in done:
                try:
                    res = future.result()
                    if isinstance(res, tuple) and len(res) == 2:
                        result, helper_name = res
                    else:
                        result = str(res)
                        helper_name = "unknown"
                except Exception as e:
                    result = f"Exception: {e}"
                    helper_name = "unknown"
                results.append(f"{helper_name}:{result}")
            for name in active_futures:
                self.helper_futures.pop(name, None)
            self.last_wait_all_result = "\n\n=====\n\n".join(results)
            return self.last_wait_all_result

        else:
            helper_name = resolved_until
            if helper_name not in self.helper_futures:
                if helper_name in self.completed_results:
                    return f"{helper_name}:{self.completed_results[helper_name]}"
                return f"No active helper named {until}."
            future = self.helper_futures.pop(helper_name)
            with self.agent.run_state(RunState.WAITING):
                try:
                    res = await future
                    if isinstance(res, tuple) and len(res) == 2:
                        result, _ = res
                    else:
                        result = str(res)
                except Exception as e:
                    result = f"Exception: {e}"
            return f"{helper_name}:{result}"
