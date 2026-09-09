"""Supervisor agent: simple coordinator for worker agents.

Provides registration, simple task dispatch, and status monitoring. This is
a lightweight in-process orchestrator used for demos and local testing.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Supervisor:
    def __init__(self):
        # map agent_name -> agent_instance
        self.agents: Dict[str, Any] = {}

    def register(self, name: str, agent):
        """Register an agent instance under a name."""
        self.agents[name] = agent
        logger.info("Registered agent %s", name)

    def unregister(self, name: str):
        if name in self.agents:
            del self.agents[name]
            logger.info("Unregistered agent %s", name)

    def status(self) -> Dict[str, Any]:
        return {"agents": list(self.agents.keys())}

    def dispatch(self, agent_name: str, task: Dict[str, Any]):
        """Send a task to a named agent and return its result.

        The agent is expected to implement a `handle_task(task)` method.
        """
        if agent_name not in self.agents:
            raise KeyError(f"Agent not found: {agent_name}")
        agent = self.agents[agent_name]
        logger.info("Dispatching task to %s: %s", agent_name, task)
        return agent.handle_task(task)

    def broadcast(self, task: Dict[str, Any]):
        """Send the task to all registered agents and collect results."""
        results = {}
        for name, agent in self.agents.items():
            try:
                results[name] = agent.handle_task(task)
            except Exception as e:
                logger.exception("Agent %s failed: %s", name, e)
                results[name] = {"error": str(e)}
        return results
