from typing import Dict

class InputNode:
    def __call__(self, state: Dict) -> Dict:
        state["query"] = state.get("input_query", "")
        return state