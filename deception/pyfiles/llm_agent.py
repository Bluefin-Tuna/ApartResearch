from abc import ABC, abstractmethod

class LLMAgent(ABC):
    @abstractmethod
    def get_action(self, state, prompt):
        pass