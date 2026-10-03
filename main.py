"""
Sentiment analysis system entrypoint.

This module orchestrates:
1. Data collection agent
2. Text analysis agent
3. Decision agent
4. Presentation agent
"""
import logging
import os
import inspect
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

try:
    from .agents.agent_factory import AgentFactory
    from .config import Config, config as global_config
    from .memory import MemoryManager
    from .tools import create_agent_caller_tool, get_default_tools
except ImportError:  # pragma: no cover - script mode fallback
    from agents.agent_factory import AgentFactory
    from config import Config, config as global_config
    from memory import MemoryManager
    from tools import create_agent_caller_tool, get_default_tools

logger = logging.getLogger(__name__)


class SentimentAnalysisSystem:
    """
    Four-layer listed-company sentiment analysis system.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.config = dict(config or {})

        self.factory = AgentFactory(llm=llm, tools=tools, config=self.config)

        self._collector = None
        self._analyst = None
        self._decision = None
        self._presenter = None
        self._agent_tool_cache: Dict[str, BaseTool] = {}

        memory_path = self.config.get("memory_path", "./memory_data")
        runtime_cfg = SimpleNamespace(memory_path=memory_path)
        self.memory_manager = MemoryManager(runtime_cfg)
        self._memory_initialized = False

    @property
    def collector(self):
        if self._collector is None:
            self._collector = self.factory.create_agent("data_collector")
        return self._collector

    @property
    def analyst(self):
        if self._analyst is None:
            self._analyst = self.factory.create_agent("text_analyst")
            self._attach_previous_agent_tool("text_analyst", self._analyst)
        return self._analyst

    @property
    def decision(self):
        if self._decision is None:
            self._decision = self.factory.create_agent("decision_maker")
            self._attach_previous_agent_tool("decision_maker", self._decision)
        return self._decision

    @property
    def presenter(self):
        if self._presenter is None:
            self._presenter = self.factory.create_agent("presenter")
            self._attach_previous_agent_tool("presenter", self._presenter)
        return self._presenter

    def _previous_agent_names(self, agent_name: str) -> List[str]:
        pipeline_order = [
            "data_collector",
            "text_analyst",
            "decision_maker",
            "presenter",
        ]
        if agent_name not in pipeline_order:
            return []
        return pipeline_order[:pipeline_order.index(agent_name)]

    def _get_agent_by_name(self, agent_name: str):
        if agent_name == "data_collector":
            return self.collector
        if agent_name == "text_analyst":
            return self.analyst
        if agent_name == "decision_maker":
            return self.decision
        if agent_name == "presenter":
            return self.presenter
        raise ValueError(f"Unknown agent: {agent_name}")

    def _attach_previous_agent_tool(self, agent_name: str, agent: Any) -> None:
        previous_agents = self._previous_agent_names(agent_name)
        if not previous_agents or agent is None:
            return

        cache_key = "|".join(previous_agents)
        tool = self._agent_tool_cache.get(cache_key)
        if tool is None:
            registry = {
                previous_name: self._get_agent_by_name(previous_name)
                for previous_name in previous_agents
            }
            tool = create_agent_caller_tool(
                agents_registry=registry,
                allowed_agents=previous_agents,
            )
            self._agent_tool_cache[cache_key] = tool

        existing_tools = list(getattr(agent, "tools", []) or [])
        existing_tools = [
            existing_tool
            for existing_tool in existing_tools
            if getattr(existing_tool, "name", None) != tool.name
        ]
        agent.tools = existing_tools + [tool]

    async def _ensure_memory(self):
        if self._memory_initialized:
            return
        try:
            await self.memory_manager.initialize()
            self._memory_initialized = True
        except Exception as exc:
            logger.warning("memory initialization failed: %s", exc)
            self._memory_initialized = True

    def _format_time_range(self, date_range: Optional[Any], collection_range: Dict[str, str]) -> str:
        if isinstance(date_range, str) and date_range.strip():
            return date_range.strip()
        if collection_range and collection_range.get("start") and collection_range.get("end"):
            return f"{collection_range['start']} to {collection_range['end']}"
        if isinstance(date_range, (list, tuple)) and len(date_range) >= 2:
            return f"{date_range[0]} to {date_range[1]}"
        return "Last 7 days"

    async def _build_memory_context(self, company_name: str) -> Dict[str, Any]:
        await self._ensure_memory()
        try:
            records = await self.memory_manager.global_memory.get("company_info", company=company_name)
            if not records:
                return {"recent_reports": []}
            record = records[0] if isinstance(records, list) and records else {}
            history = record.get("history", []) if isinstance(record, dict) else []
            recent_reports = [
                item.get("report", {})
                for item in history[-5:]
                if isinstance(item, dict) and isinstance(item.get("report"), dict)
            ]
            return {
                "recent_reports": recent_reports,
                "last_updated": record.get("timestamp"),
                "company": company_name,
            }
        except Exception as exc:
            logger.warning("build memory context failed: %s", exc)
            return {"recent_reports": []}

    async def _persist_memory(
        self,
        company_name: str,
        stock_code: Optional[str],
        report: Any,
        document_count: int,
        used_fallback_data: bool,
    ):
        await self._ensure_memory()
        try:
            content = {
                "company_name": company_name,
                "stock_code": stock_code,
                "generated_at": datetime.now().isoformat(),
                "used_fallback_data": used_fallback_data,
                "document_count": document_count,
                "report": report.to_dict() if hasattr(report, "to_dict") else {},
                "risk_assessment": {
                    "risk_level": report.risk_assessment.risk_level,
                    "risk_score": report.risk_assessment.risk_score,
                },
                "sentiment_distribution": getattr(report, "sentiment_distribution", {}),
            }
            await self.memory_manager.global_memory.add(
                {
                    "type": "company_info",
                    "company": company_name,
                    "content": content,
                }
            )
        except Exception as exc:
            logger.warning("persist memory failed: %s", exc)

    async def _emit_progress(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
        *,
        stage: str,
        status: str,
        task: str,
        output: Optional[Any] = None,
    ) -> None:
        if progress_callback is None:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "status": status,
            "task": task,
            "output": output,
        }
        try:
            callback_result = progress_callback(event)
            if inspect.isawaitable(callback_result):
                await callback_result
        except Exception as exc:
            logger.warning("progress callback failed: %s", exc)

    async def analyze(
        self,
        company_name: str,
        stock_code: Optional[str] = None,
        sources: Optional[List[str]] = None,
        date_range: Optional[Any] = None,
        max_effective_documents: Optional[int] = None,
        output_format: str = "markdown",
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        await self._ensure_memory()
        await self._emit_progress(
            progress_callback,
            stage="Pipeline",
            status="running",
            task="Starting sentiment analysis pipeline",
        )

        try:
            await self._emit_progress(
                progress_callback,
                stage="DataCollectionAgent",
                status="running",
                task=f"Collecting multi-source documents for {company_name}",
            )
            collection_result = await self.collector.execute(
                company_name=company_name,
                stock_code=stock_code,
                sources=sources,
                date_range=date_range,
            )
            await self._emit_progress(
                progress_callback,
                stage="DataCollectionAgent",
                status="completed",
                task="Data collection completed",
                output=collection_result,
            )

            documents = collection_result.get("documents", [])
            if not documents:
                empty_result = {
                    "report": f"No sentiment data found for {company_name}.",
                    "raw_report": None,
                    "charts": [],
                "metadata": {
                    "company_name": company_name,
                    "stock_code": stock_code,
                    "analyzed_at": datetime.now().isoformat(),
                    "document_count": 0,
                    "effective_text_limit": max_effective_documents,
                    "analysis_stopped_early": False,
                    "unprocessed_document_count": 0,
                },
            }
                await self._emit_progress(
                    progress_callback,
                    stage="Pipeline",
                    status="completed",
                    task="Pipeline finished with empty result",
                    output={"metadata": empty_result["metadata"]},
                )
                return empty_result

            resolved = collection_result.get("resolved_company", {}) or {}
            resolved_name = resolved.get("company_name") or company_name
            resolved_stock_code = resolved.get("stock_code") or stock_code
            memory_context = await self._build_memory_context(resolved_name)

            await self._emit_progress(
                progress_callback,
                stage="TextAnalysisAgent",
                status="running",
                task=f"Analyzing {len(documents)} documents",
            )
            analyzed_docs = await self.analyst.execute(
                documents=documents,
                target_company=resolved_name,
                max_effective_documents=max_effective_documents,
            )
            analysis_summary = {}
            try:
                if hasattr(self.analyst, "get_analysis_summary"):
                    analysis_summary = self.analyst.get_analysis_summary() or {}
            except Exception:
                analysis_summary = {}
            analyzed_count = len(analyzed_docs)
            skipped_count = int(analysis_summary.get("skipped_documents", max(0, len(documents) - analyzed_count)) or 0)
            await self._emit_progress(
                progress_callback,
                stage="TextAnalysisAgent",
                status="completed",
                task="Text analysis completed",
                output={
                    "analyzed_documents": analyzed_count,
                    "skipped_documents": skipped_count,
                    "summary": analysis_summary.get(
                        "overall_document_summary",
                        analysis_summary.get("executive_summary", ""),
                    ),
                    "document_summary_count": len(analysis_summary.get("document_summaries", []) or []),
                },
            )

            time_range_text = self._format_time_range(
                date_range=date_range,
                collection_range=collection_result.get("date_range", {}),
            )
            await self._emit_progress(
                progress_callback,
                stage="DecisionAgent",
                status="running",
                task="Aggregating sentiment and risk signals",
            )
            report = await self.decision.execute(
                company_name=resolved_name,
                stock_code=resolved_stock_code,
                analyzed_documents=analyzed_docs,
                time_range=time_range_text,
                memory_context=memory_context,
                analysis_summary=analysis_summary,
            )
            await self._emit_progress(
                progress_callback,
                stage="DecisionAgent",
                status="completed",
                task="Decision report generated",
                output=report,
            )

            await self._emit_progress(
                progress_callback,
                stage="PresentationAgent",
                status="running",
                task=f"Rendering report as {output_format}",
            )
            presentation = await self.presenter.execute(
                report=report,
                format=output_format,
            )
            await self._emit_progress(
                progress_callback,
                stage="PresentationAgent",
                status="completed",
                task="Presentation assets generated",
                output=presentation,
            )

            used_fallback_data = collection_result.get("used_fallback_data", False)
            await self._persist_memory(
                company_name=resolved_name,
                stock_code=resolved_stock_code,
                report=report,
                document_count=analyzed_count,
                used_fallback_data=used_fallback_data,
            )

            final_result = {
                "report": presentation.get("report", ""),
                "raw_report": report,
                "charts": presentation.get("charts", []),
                "metadata": {
                    "company_name": resolved_name,
                    "stock_code": resolved_stock_code,
                    "input_company_name": company_name,
                    "input_stock_code": stock_code,
                    "analyzed_at": datetime.now().isoformat(),
                    "date_range": collection_result.get("date_range", {}),
                    "raw_document_count": len(documents),
                    "document_count": analyzed_count,
                    "filtered_document_count": skipped_count,
                    "effective_text_limit": analysis_summary.get("effective_text_limit", max_effective_documents),
                    "analysis_stopped_early": bool(analysis_summary.get("analysis_stopped_early", False)),
                    "unprocessed_document_count": int(
                        analysis_summary.get("unprocessed_document_count", 0) or 0
                    ),
                    "text_analysis_summary": analysis_summary.get("executive_summary", ""),
                    "text_analysis_overall_summary": analysis_summary.get(
                        "overall_document_summary",
                        analysis_summary.get("executive_summary", ""),
                    ),
                    "document_summaries": analysis_summary.get("document_summaries", []),
                    "text_analysis_overview": analysis_summary,
                    "used_fallback_data": used_fallback_data,
                },
            }
            await self._emit_progress(
                progress_callback,
                stage="Pipeline",
                status="completed",
                task="Sentiment analysis pipeline completed",
                output={"metadata": final_result["metadata"]},
            )
            return final_result
        except Exception as exc:
            await self._emit_progress(
                progress_callback,
                stage="Pipeline",
                status="failed",
                task="Sentiment analysis pipeline failed",
                output={"error": str(exc)},
            )
            raise

    async def analyze_only_sentiment(self, text: str) -> Dict[str, Any]:
        try:
            from .tools import sentiment_analysis
        except ImportError:  # pragma: no cover - script mode fallback
            from tools import sentiment_analysis

        result = await sentiment_analysis.ainvoke({"text": text})
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": result,
        }

    @staticmethod
    def _has_gradio_frpc_binary() -> bool:
        if os.name != "nt":
            return True
        try:
            import gradio

            gradio_dir = Path(gradio.__file__).resolve().parent
            candidates = ("frpc_windows_amd64_v0.3", "frpc_windows_amd64.exe")
            return any((gradio_dir / name).exists() for name in candidates)
        except Exception:
            return False

    def launch_gradio(self, share: bool = False):
        demo = self.presenter.create_gradio_interface()
        self.presenter.analysis_handler = self.analyze
        if not demo:
            logger.error("failed to create gradio interface")
            return

        port = int(self.config.get("gradio_port", 7860))
        effective_share = bool(share)
        if effective_share and not self._has_gradio_frpc_binary():
            logger.warning(
                "gradio share requested but frpc binary is missing; fallback to local-only mode"
            )
            effective_share = False

        try:
            demo.launch(
                share=effective_share,
                server_name="0.0.0.0",
                server_port=port,
            )
        except Exception as exc:
            if effective_share:
                logger.warning("gradio share launch failed, retry with local-only mode: %s", exc)
                demo.launch(
                    share=False,
                    server_name="0.0.0.0",
                    server_port=port,
                )
                return
            raise


def build_llm(config: Config) -> BaseLanguageModel:
    """Build LLM instance from project config."""
    from langchain_openai import ChatOpenAI

    provider = (config.llm_provider or "openai").lower()
    api_key = (config.llm_api_key or "").strip()
    base_url = (config.llm_base_url or "").strip()

    if provider == "dashscope":
        base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif provider == "openai":
        base_url = base_url or None

    if not api_key:
        raise ValueError("LLM_API_KEY 未配置，无法创建 LLM。请复制 .env.example 为 .env 并填写密钥。")

    return ChatOpenAI(
        model=config.llm_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def build_default_system(config: Optional[Config] = None) -> SentimentAnalysisSystem:
    resolved_config = config or global_config
    llm = build_llm(resolved_config)
    tools = get_default_tools()
    return create_sentiment_system(
        llm=llm,
        tools=tools,
        config=resolved_config.__dict__,
    )


def create_sentiment_system(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    config: Optional[Dict[str, Any]] = None,
) -> SentimentAnalysisSystem:
    return SentimentAnalysisSystem(llm=llm, tools=tools, config=config)


async def run_sample_analysis():
    try:
        system = build_default_system(global_config)
    except ValueError as exc:
        logger.error(str(exc))
        print(str(exc))
        return

    result = await system.analyze(
        company_name="璐靛窞鑼呭彴",
        stock_code="600519",
    )
    print(result["report"])


def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    try:
        system = build_default_system(global_config)
    except ValueError as exc:
        logger.error(str(exc))
        print(str(exc))
        return

    share = bool(getattr(global_config, "gradio_share", False))
    port = int(getattr(global_config, "gradio_port", 7860))
    logger.info("launching gradio at http://127.0.0.1:%s (share=%s)", port, share)
    print(f"Launching Gradio at http://127.0.0.1:{port} (share={share})")
    system.launch_gradio(share=share)


if __name__ == "__main__":
    main()

