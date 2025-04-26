"""
Mode switching tool for SerenaAgent.
"""

from serena.agent import Tool, ToolMarkerDoesNotRequireActiveProject
from serena.serena_agent_extended import SerenaAgent


class SetModesTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """
    Sets the modes for the SerenaAgent.
    """

    def apply(self, modes: list[str]) -> str:
        """
        Set the modes for the agent.

        :param modes: List of mode names or paths
        :return: Success message
        """
        try:
            # Cast agent to SerenaAgent to access set_modes
            agent = self.agent
            if isinstance(agent, SerenaAgent):
                agent.set_modes(modes)
                return f"Successfully set modes: {', '.join(modes)}"
            else:
                return "Error: Agent does not support mode switching"
        except Exception as e:
            return f"Error setting modes: {e!s}"
