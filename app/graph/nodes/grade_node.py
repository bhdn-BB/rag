from typing import Dict


class GradeNode:
    def __call__(self, state: Dict) -> Dict:
        state["docs_count"] = len(state.get("docs", []))
        return state