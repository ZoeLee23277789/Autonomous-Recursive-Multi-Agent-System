import asyncio
import logging
from typing import Annotated

from kani import AIParam, ChatRole, ai_function
from rapidfuzz import fuzz
import openai

import events
from state import RunState
from delegation._base import DelegationBase

log = logging.getLogger(__name__)


class DelegateWait(DelegationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helpers = {}
        self.helper_futures = {}
        self.request_semaphore = asyncio.Semaphore(3)

    def is_duplicate_task(self, instructions: str):
        return any(instructions == task["task"] for task in self.app.global_task_log)

    @ai_function(desc="Delegate a subtask to another agent with specific instructions. Returns immediately.")
    async def delegate(
        self,
        instructions: Annotated[str, AIParam("Detailed instructions for your helper.")],
        who: Annotated[str, AIParam("Name of an existing helper to continue with (optional).")] = None,
    ):
        log.info(f"Delegated with instructions: {instructions}")

        if self.is_duplicate_task(instructions):
            return f"⚠️ 類似任務已經被分派過了，跳過重複指派。"

        if getattr(self, "depth", 0) >= 4:
            return f"⚠️ 已達最大遞迴層數（4）。請在目前層級內完成任務。"

        if len(self.helpers) >= 4 and "__AUTO_WAITING__" not in self.helper_futures:
            print(f"\n🔁 已建立 {len(self.helpers)} 個 sub-agent，啟動自動統整機制...\n")
            self.helper_futures["__AUTO_WAITING__"] = asyncio.create_task(self._auto_wait_all())

        if self.kani.last_user_message and fuzz.ratio(instructions, self.kani.last_user_message.content) > 80:
            return "You shouldn't delegate the entire task to a helper. Break it into smaller parts if necessary."

        if who and who in self.helpers:
            if who in self.helper_futures:
                return f"{who!r} is still working. Wait or delegate to someone else."
            helper = self.helpers[who]
            self.app.dispatch(events.KaniDelegated(
                parent_id=self.kani.id,
                child_id=helper.id,
                parent_message_idx=len(self.kani.chat_history) - 1,
                child_message_idx=len(helper.chat_history),
                instructions=instructions,
            ))
        else:
            helper = await self.create_delegate_kani(instructions)
            helper.task_description = instructions
            self.helpers[helper.name] = helper
            print(f"\n[✅ 任務指派] Agent: {helper.name}")
            print("📄 被指派的任務：")
            print(instructions)
            print("-" * 40 + "\n")
            self.app.global_task_log.append({
                "agent": helper.name,
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

                for entry in self.app.global_task_log:
                    if entry["agent"] == helper.name and entry["task"] == instructions:
                        entry["status"] = "completed"

                await helper.cleanup()
                return "\n".join(result), helper.name
            except Exception as e:
                log.exception(f"{helper.name}-{helper.depth} encountered an exception!")
                for entry in self.app.global_task_log:
                    if entry["agent"] == helper.name and entry["task"] == instructions:
                        entry["status"] = f"failed: {e}"
                try:
                    new_helper = await self.create_delegate_kani(instructions)
                    new_helper.task_description = instructions
                    self.helpers[new_helper.name] = new_helper
                    print(f"\n[🔁 任務重新委派] 由 {helper.name} 改為 {new_helper.name}")
                    self.app.global_task_log.append({
                        "agent": new_helper.name,
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

        with self.kani.run_state(RunState.WAITING):
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
        if until not in self.helper_futures and until not in ("next", "all"):
            return 'The "until" param must be a running helper name, "next", or "all".'

        if until == "next":
            if not self.helper_futures:
                return "There are no active sub-agents to wait for."
            with self.kani.run_state(RunState.WAITING):
                done, _ = await asyncio.wait(self.helper_futures.values(), return_when=asyncio.FIRST_COMPLETED)
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
            if not self.helper_futures:
                return "No sub-agents were successfully assigned. Please try delegating again."
            with self.kani.run_state(RunState.WAITING):
                done, _ = await asyncio.wait(self.helper_futures.values(), return_when=asyncio.ALL_COMPLETED)
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
            self.helper_futures.clear()
            return "\n\n=====\n\n".join(results)

        else:
            if until not in self.helper_futures:
                return f"No active helper named {until}."
            future = self.helper_futures.pop(until)
            with self.kani.run_state(RunState.WAITING):
                try:
                    res = await future
                    if isinstance(res, tuple) and len(res) == 2:
                        result, _ = res
                    else:
                        result = str(res)
                except Exception as e:
                    result = f"Exception: {e}"
            return f"{until}:{result}"

