from typing import Dict

class FallbackNode:
    def __call__(self, state: Dict) -> Dict:
        state["answer"] = "На жаль, інформації з цього питання немає."
        state["sources"] = []
        return state