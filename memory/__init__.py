"""
Memory module for session context and long-term company history.
"""
import json
import logging
import os
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SessionMemory:
    """
    Short-term conversation memory.
    """

    def __init__(self, session_id: str, max_size: int = 100):
        self.session_id = session_id
        self.max_size = max_size
        self._messages = deque(maxlen=max_size)
        self._context: Dict[str, Any] = {}
        self.created_at = datetime.now()

    async def add(self, message: Dict[str, Any]):
        payload = dict(message)
        payload["timestamp"] = datetime.now().isoformat()
        self._messages.append(payload)

    async def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return list(self._messages)[-n:]

    async def get_context(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    async def set_context(self, key: str, value: Any):
        self._context[key] = value

    async def clear(self):
        self._messages.clear()
        self._context.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": list(self._messages),
            "context": self._context,
            "created_at": self.created_at.isoformat(),
        }


class GlobalMemory:
    """
    Long-term memory persisted to disk.
    """

    def __init__(self, storage_path: str = "./memory_data"):
        self.storage_path = storage_path
        self._ensure_storage()

        self._task_progress: Dict[str, Dict[str, Any]] = {}
        self._long_term_goals: List[Dict[str, Any]] = []
        self._achievements: List[Dict[str, Any]] = []
        self._company_cache: Dict[str, Dict[str, Any]] = {}
        self._company_history_limit = 30

    def _ensure_storage(self):
        os.makedirs(self.storage_path, exist_ok=True)

    async def add(self, data: Dict[str, Any]):
        data_type = data.get("type", "general")
        timestamp = datetime.now().isoformat()
        record = {**data, "timestamp": timestamp}

        if data_type == "task_progress":
            task_id = data.get("task_id", "default")
            self._task_progress[task_id] = record

        elif data_type == "goal":
            self._long_term_goals.append(record)

        elif data_type == "achievement":
            self._achievements.append(record)

        elif data_type == "company_info":
            company = data.get("company")
            if company:
                existing = self._company_cache.get(company, {})
                history = list(existing.get("history", [])) if isinstance(existing, dict) else []
                content = data.get("content", {})
                history.append(
                    {
                        "timestamp": timestamp,
                        "generated_at": content.get("generated_at", timestamp),
                        "report": content.get("report", {}),
                        "risk_assessment": content.get("risk_assessment", {}),
                        "sentiment_distribution": content.get("sentiment_distribution", {}),
                        "content": content,
                    }
                )
                history = history[-self._company_history_limit :]
                record["history"] = history
                self._company_cache[company] = record

        await self._persist()

    async def get(self, data_type: str, **filters: Any) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if data_type == "task_progress":
            task_id = filters.get("task_id")
            if task_id:
                if task_id in self._task_progress:
                    results.append(self._task_progress[task_id])
            else:
                results = list(self._task_progress.values())

        elif data_type == "goal":
            results = list(self._long_term_goals)

        elif data_type == "achievement":
            results = list(self._achievements)

        elif data_type == "company_info":
            company = filters.get("company")
            if company and company in self._company_cache:
                results.append(self._company_cache[company])

        return results

    async def get_system_prompt_context(self) -> str:
        context_parts = []

        if self._task_progress:
            task_lines = "\n".join(
                f"- {task_id}: {task.get('status', 'unknown')}"
                for task_id, task in self._task_progress.items()
            )
            context_parts.append(f"## 任务进度\n{task_lines}")

        if self._long_term_goals:
            goal_lines = "\n".join(f"- {goal.get('content', '')}" for goal in self._long_term_goals)
            context_parts.append(f"## 长期目标\n{goal_lines}")

        return "\n\n".join(context_parts) if context_parts else "无特殊上下文"

    async def _persist(self):
        try:
            payload = {
                "task_progress": self._task_progress,
                "long_term_goals": self._long_term_goals,
                "achievements": self._achievements,
                "company_cache": self._company_cache,
                "updated_at": datetime.now().isoformat(),
            }
            path = os.path.join(self.storage_path, "global_memory.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("persist global memory failed: %s", exc)

    async def load(self):
        try:
            path = os.path.join(self.storage_path, "global_memory.json")
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._task_progress = payload.get("task_progress", {})
            self._long_term_goals = payload.get("long_term_goals", [])
            self._achievements = payload.get("achievements", [])
            self._company_cache = payload.get("company_cache", {})
        except Exception as exc:
            logger.error("load global memory failed: %s", exc)


class MemoryManager:
    """
    Unified memory manager for session + global memory.
    """

    def __init__(self, config: Any):
        self.config = config
        self.global_memory = GlobalMemory(config.memory_path)
        self.sessions: Dict[str, SessionMemory] = {}

    def get_session(self, session_id: str) -> SessionMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id)
        return self.sessions[session_id]

    async def initialize(self):
        await self.global_memory.load()
