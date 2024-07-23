# framework.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class ControlledMechanic(ABC):
    @abstractmethod
    def get_options(self, state: Dict[str, Any]) -> List[str]:
        """Get the available options for this mechanic given the current state."""
        pass

    @abstractmethod
    def execute_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Execute the chosen action and return the updated state."""
        pass

class DecisionPoint:
    def __init__(self, state: Dict[str, Any], mechanic: ControlledMechanic):
        self.state = state
        self.mechanic = mechanic
        self.options = mechanic.get_options(state)

class AgentController(ABC):
    @abstractmethod
    def make_decision(self, decision_point: DecisionPoint) -> str:
        """Make a decision given the current decision point."""
        pass

class Environment(ABC):
    def __init__(self):
        self.controlled_mechanics: Dict[str, ControlledMechanic] = {}
        self.agent_controller: AgentController = None

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to its initial state."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment based on the given action."""
        pass

    def agent_step(self, mechanic_name: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step for an agent-controlled mechanic."""
        mechanic = self.controlled_mechanics[mechanic_name]
        state = self.get_state()
        decision_point = DecisionPoint(state, mechanic)
        action = self.agent_controller.make_decision(decision_point)
        new_state = mechanic.execute_action(state, action)
        return self.step(new_state)

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        pass

    @abstractmethod
    def is_over(self) -> bool:
        """Check if the environment is over."""
        pass

    @abstractmethod
    def get_result(self) -> Dict[str, Any]:
        """Get the final result of the environment."""
        pass