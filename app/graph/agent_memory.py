# from typing import Dict, List
#
# class AgentMemory:
#     def __init__(self):
#         self.sessions: Dict[str, List[str]] = {}
#
#     def add_message(self, session_id: str, message: str):
#         self.sessions.setdefault(session_id, []).append(message)
#
#     def get_context(self, session_id: str) -> str:
#         return "\n".join(self.sessions.get(session_id, []))
#
#     def clear_session(self, session_id: str):
#         if session_id in self.sessions:
#             del self.sessions[session_id]
