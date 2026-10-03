"""
Agent 工厂 - 统一创建和管理所有 Agent

提供统一的 Agent 创建接口，支持配置化
"""
import logging
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Agent 工厂类
    
    统一创建四种核心 Agent:
    1. DataCollectionAgent - 数据采集
    2. TextAnalysisAgent - 文本分析
    3. DecisionAgent - 决策分析
    4. PresentationAgent - 结果展示
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm
        self.tools = tools or []
        self.config = config or {}
        
        # 缓存已创建的 Agent
        self._agents: Dict[str, Any] = {}
    
    def create_agent(self, agent_type: str, **kwargs) -> Any:
        """
        创建指定类型的 Agent
        
        Args:
            agent_type: Agent 类型
                - "data_collector": 数据采集 Agent
                - "text_analyst": 文本分析 Agent
                - "decision_maker": 决策 Agent
                - "presenter": 展示 Agent
        
        Returns:
            对应类型的 Agent 实例
        """
        # 检查缓存
        if agent_type in self._agents:
            logger.info(f"使用缓存的 Agent: {agent_type}")
            return self._agents[agent_type]
        
        logger.info(f"创建 Agent: {agent_type}")
        
        # 创建 Agent
        if agent_type == "data_collector":
            agent = self._create_data_collector(**kwargs)
        elif agent_type == "text_analyst":
            agent = self._create_text_analyst(**kwargs)
        elif agent_type == "decision_maker":
            agent = self._create_decision_maker(**kwargs)
        elif agent_type == "presenter":
            agent = self._create_presenter(**kwargs)
        else:
            raise ValueError(f"未知的 Agent 类型：{agent_type}")
        
        # 缓存
        self._agents[agent_type] = agent
        
        return agent
    
    def _create_data_collector(self, **kwargs) -> Any:
        """创建数据采集 Agent"""
        from .data_collector_agent import DataCollectionAgent
        
        # 数据采集专用工具
        collector_tools = self._filter_tools([
            "news_search",
            "web_crawl",
            "stock_open_company_info",
            "entity_recognition",
        ])
        
        agent = DataCollectionAgent(
            llm=self.llm,
            tools=collector_tools,
            config=self.config.get("data_collector", {}),
            **kwargs,
        )
        
        logger.info("数据采集 Agent 创建成功")
        return agent
    
    def _create_text_analyst(self, **kwargs) -> Any:
        """创建文本分析 Agent"""
        from .text_analyst_agent import TextAnalysisAgent
        
        # 文本分析专用工具
        analyst_tools = self._filter_tools([
            "sentiment_analysis",
            "entity_recognition",
            "event_extraction",
        ])
        
        agent = TextAnalysisAgent(
            llm=self.llm,
            tools=analyst_tools,
            config=self.config.get("text_analyst", {}),
            **kwargs,
        )
        
        logger.info("文本分析 Agent 创建成功")
        return agent
    
    def _create_decision_maker(self, **kwargs) -> Any:
        """创建决策 Agent"""
        from .decision_agent import DecisionAgent
        
        # 决策 Agent 主要依靠 LLM，不需要太多工具
        decision_tools = self._filter_tools(["vector_search", "knowledge_query"])
        
        agent = DecisionAgent(
            llm=self.llm,
            tools=decision_tools,
            config=self.config.get("decision", {}),
            **kwargs,
        )
        
        logger.info("决策 Agent 创建成功")
        return agent
    
    def _create_presenter(self, **kwargs) -> Any:
        """创建展示 Agent"""
        from .presentation_agent import PresentationAgent
        
        agent = PresentationAgent(
            llm=self.llm,
            tools=[],  # 展示 Agent 不需要工具
            config=self.config.get("presentation", {}),
            **kwargs,
        )
        
        logger.info("展示 Agent 创建成功")
        return agent
    
    def _filter_tools(self, tool_names: List[str]) -> List[BaseTool]:
        """过滤出指定名称的工具"""
        filtered = []
        
        for tool in self.tools:
            if hasattr(tool, "name") and tool.name in tool_names:
                filtered.append(tool)
        
        return filtered
    
    def create_all_agents(self) -> Dict[str, Any]:
        """创建所有 Agent"""
        return {
            "data_collector": self.create_agent("data_collector"),
            "text_analyst": self.create_agent("text_analyst"),
            "decision_maker": self.create_agent("decision_maker"),
            "presenter": self.create_agent("presenter"),
        }
    
    def get_agent(self, agent_type: str) -> Optional[Any]:
        """获取已创建的 Agent"""
        return self._agents.get(agent_type)


# ========== 快捷创建函数 ==========

def create_sentiment_analysis_pipeline(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    创建完整的舆情分析流水线
    
    Returns:
        包含所有 Agent 的字典
    """
    factory = AgentFactory(llm=llm, tools=tools, config=config)
    return factory.create_all_agents()


# ========== 使用示例 ==========

if __name__ == "__main__":
    """
    使用示例:
    
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.tools import DuckDuckGoSearchRun
    
    # 初始化 LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # 初始化工具
    tools = [DuckDuckGoSearchRun()]
    
    # 创建工厂
    factory = AgentFactory(llm=llm, tools=tools)
    
    # 创建单个 Agent
    collector = factory.create_agent("data_collector")
    
    # 或创建所有 Agent
    agents = factory.create_all_agents()
    
    # 使用 Agent
    result = await collector.execute(
        company_name="贵州茅台",
        stock_code="600519",
    )
    """
    pass
