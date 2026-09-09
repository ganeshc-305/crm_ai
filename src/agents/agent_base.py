"""Agent base class for simple in-process agents."""
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class AgentBase:
    def __init__(self, name: str):
        self.name = name

    def handle_task(self, task: Dict[str, Any]) -> Any:
        """Handle a task dictionary and return a result. Override in subclasses."""
        logger.info("%s received task: %s", self.name, task)
        return {"status": "noop", "task": task}
