"""
智能体模块 - 基础智能体架构
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """智能体状态"""
    task: str
    current_step: int = 0
    total_steps: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    output: Optional[Any] = None
    status: str = "idle"  # idle, running, completed, failed
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class BaseAgent(ABC):
    """
    智能体基类
    
    采用 ReAct 范式：Reason -> Act -> Observe 循环
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLanguageModel,
        tools: Optional[List[Any]] = None,
        memory: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.memory = memory
        self.config = config
        self.state = AgentState(task="")
        
        self._setup()
    
    def _setup(self):
        """初始化设置"""
        pass
    
    @abstractmethod
    async def execute(self, task: str, **kwargs) -> Any:
        """
        执行任务
        
        Args:
            task: 任务描述
            **kwargs: 额外参数
        
        Returns:
            执行结果
        """
        pass
    
    async def _reason(self, context: Dict[str, Any]) -> str:
        """
        推理：决定下一步行动
        
        Args:
            context: 当前上下文
        
        Returns:
            推理结果（下一步行动计划）
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_template(
            """你是一个智能助手。请分析当前情况并决定下一步行动。

当前任务：{task}
已有信息：{context}
可用工具：{tools}

请说明：
1. 当前进展
2. 下一步应该做什么
3. 需要调用什么工具（如果有）

推理结果："""
        )
        
        chain = prompt | self.llm
        
        response = await chain.ainvoke({
            "task": self.state.task,
            "context": str(context),
            "tools": [t.name for t in self.tools] if self.tools else [],
        })
        
        return response.content
    
    async def _act(self, action: str, **kwargs) -> Any:
        """
        行动：调用工具执行操作
        
        Args:
            action: 行动描述
            **kwargs: 工具参数
        
        Returns:
            行动结果
        """
        # 解析行动，决定调用哪个工具
        for tool in self.tools:
            if tool.name.lower() in action.lower():
                logger.info(f"调用工具：{tool.name}")
                return await tool.ainvoke(kwargs)
        
        # 如果没有匹配的工具，直接返回
        logger.warning(f"未找到匹配的工具：{action}")
        return None
    
    async def _observe(self, result: Any) -> Dict[str, Any]:
        """
        观察：处理工具返回结果
        
        Args:
            result: 工具返回结果
        
        Returns:
            观察结果（结构化信息）
        """
        return {
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
    
    async def _run_react_loop(
        self,
        task: str,
        max_iterations: int = 10,
    ) -> Any:
        """
        运行 ReAct 循环
        
        Args:
            task: 任务
            max_iterations: 最大迭代次数
        
        Returns:
            最终结果
        """
        self.state.task = task
        self.state.status = "running"
        self.state.current_step = 0
        self.state.total_steps = max_iterations
        
        context = {}
        
        for i in range(max_iterations):
            self.state.current_step = i + 1
            
            # Reason
            reasoning = await self._reason(context)
            logger.debug(f"[{self.name}] 推理：{reasoning}")
            
            # 检查是否完成
            if "完成" in reasoning or "无需" in reasoning:
                break
            
            # Act
            action_result = await self._act(reasoning)
            
            # Observe
            observation = await self._observe(action_result)
            context[f"step_{i}"] = {
                "reasoning": reasoning,
                "observation": observation,
            }
            
            # 记录到记忆
            if self.memory:
                await self.memory.add({
                    "agent": self.name,
                    "step": i,
                    "reasoning": reasoning,
                    "observation": observation,
                })
        
        self.state.status = "completed"
        self.state.output = context
        
        return context
    
    def get_state(self) -> AgentState:
        """获取当前状态"""
        return self.state
    
    def reset(self):
        """重置状态"""
        self.state = AgentState(task="")
