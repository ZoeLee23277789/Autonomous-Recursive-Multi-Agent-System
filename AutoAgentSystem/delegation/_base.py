from typing import TYPE_CHECKING

from tools import ToolBase

if TYPE_CHECKING:
    from agents import RecursiveAgent


class DelegationBase(ToolBase):
    """
    This class is a base that all delegation implementations should inherit from.

    It extends :class:`.ToolBase` with an interface for creating delegate agent instances.
    """

    async def create_delegate_agent(self, instructions: str) -> "RecursiveAgent":
        r"""
        Call this method to get a fresh :class:`.RecursiveAgent` instance.

        This method will handle setting up the new agent in the computation graph as well as its tools, engine, and
        always included prompt based on the app configuration. It will *not* launch the agent with the given
        instructions -- this must be done by the calling function.

        The calling function is thus responsible for:

        * Setting the state of the calling agent
        * Providing the instructions to the delegate agent and calling its ``full_round_stream`` method
        * Buffering the delegate's response and returning it to the caller
        * Calling the appropriate cleanup methods of the delegate
        """
        return await self.agent.create_delegate_agent(instructions)
