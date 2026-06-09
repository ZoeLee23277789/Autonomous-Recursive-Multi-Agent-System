import logging
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

from state import RunState
from delegation._base import DelegationBase


log = logging.getLogger(__name__)


class DelegateOne(DelegationBase):
    """
    Delegate and immediately wait for the result of the sub-agent.
    Can be called in parallel to run multiple sub-agents in parallel.
    """

    @ai_function(
        desc=(
            "Delegate one bounded role-specific subtask and immediately wait for the result. Use when the selected "
            "topology needs a specialist result before continuing."
        )
    )
    async def delegate(
        self,
        instructions: Annotated[
            str,
            AIParam(
                "Detailed, scoped instructions for the helper. Include its role, boundaries, expected output, and "
                "whether it may further delegate if the subtask is still broad."
            ),
        ],
    ):
        """
        Ask a capable helper for help looking up a piece of information or performing an action.
        Do not simply repeat what the user said as instructions.
        You should use this to break up complex user queries into multiple simpler steps.
        NOTE: Helpers cannot see previous parts of your conversation.
        """
        log.info(f"Delegated with instructions: {instructions}")
        # if the instructions are >80% the same as the current goal, bonk
        if self.agent.last_user_message and fuzz.ratio(instructions, self.agent.last_user_message.content) > 80:
            return (
                "You shouldn't delegate the entire task to a helper. Handle it yourself, or if it's still too complex,"
                " try breaking it up into smaller steps and call this again."
            )

        # wait for child
        helper = await self.create_delegate_agent(instructions)
        with self.agent.run_state(RunState.WAITING):
            result = []
            async for stream in helper.full_round_stream(instructions, max_function_rounds=5):  # TODO temp
                msg = await stream.message()
                log.info(msg)
                if msg.role == ChatRole.ASSISTANT and msg.content:
                    result.append(msg.content)
            await helper.cleanup()
            return "\n".join(result)
