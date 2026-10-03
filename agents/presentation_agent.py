"""
Presentation agent for structured report generation and UI integration.
"""
import asyncio
import inspect
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from .base_agent import BaseAgent
from .decision_agent import DecisionReport

logger = logging.getLogger(__name__)

EVENT_TYPE_DISPLAY = {
    "penalty": "Penalty",
    "litigation": "Litigation",
    "investigation": "Investigation",
    "violation": "Violation",
    "case_filing": "Case Filing",
    "earnings": "Earnings",
    "financing": "Financing",
    "merger": "Merger",
    "cooperation": "Cooperation",
    "personnel": "Personnel",
    "unknown": "Unknown",
    "\u5904\u7f5a": "Penalty",
    "\u8bc9\u8bbc": "Litigation",
    "\u8c03\u67e5": "Investigation",
    "\u8fdd\u89c4": "Violation",
    "\u7acb\u6848": "Case Filing",
    "\u4e1a\u7ee9": "Earnings",
    "\u878d\u8d44": "Financing",
    "\u5e76\u8d2d": "Merger",
    "\u5408\u4f5c": "Cooperation",
    "\u4eba\u4e8b": "Personnel",
}


class PresentationAgent(BaseAgent):
    """
    Presentation agent.

    Generates markdown/html/json reports and chart payloads.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        super().__init__(
            name="presenter",
            llm=llm,
            tools=tools or [],
            memory=memory,
            config=config,
        )
        self.report_template = "default"
        self.analysis_handler = None

    async def execute(
        self,
        report: DecisionReport,
        format: str = "markdown",
        include_charts: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.state.task = f"generate {format} report"
        self.state.status = "running"

        if format == "html":
            report_content = self._generate_html_report(report)
        elif format == "json":
            report_content = self._generate_json_report(report)
        else:
            report_content = self._generate_markdown_report(report)

        charts = self._generate_chart_data(report) if include_charts else []
        self.state.status = "completed"
        self.state.output = {
            "report": report_content,
            "charts": charts,
            "generated_at": datetime.now().isoformat(),
        }
        return self.state.output

    def _generate_markdown_report(self, report: DecisionReport) -> str:
        risk_icon = {
            "low": "[LOW]",
            "medium": "[MEDIUM]",
            "high": "[HIGH]",
            "critical": "[CRITICAL]",
        }.get(report.risk_assessment.risk_level, "[UNKNOWN]")

        sentiment_icon = {
            "positive": "[+]",
            "neutral": "[=]",
            "negative": "[-]",
        }.get(report.overall_sentiment.get("label", "neutral"), "[=]")

        dist = report.sentiment_distribution or {}
        pos = dist.get("positive", 0)
        neu = dist.get("neutral", 0)
        neg = dist.get("negative", 0)
        total = max(1, pos + neu + neg)
        analysis_summary_text = str(report.data_overview.get("text_analysis_summary", "")).strip()
        overall_document_summary = str(
            report.data_overview.get("overall_document_summary", "") or analysis_summary_text
        ).strip()
        document_summaries = report.data_overview.get("document_summaries", []) or []
        raw_input_documents = int(
            report.data_overview.get("raw_input_documents", report.data_overview.get("total_documents", 0)) or 0
        )
        filtered_out_documents = int(report.data_overview.get("filtered_out_documents", 0) or 0)
        effective_text_limit = report.data_overview.get("effective_text_limit")
        processed_documents = int(report.data_overview.get("processed_documents", 0) or 0)
        unprocessed_document_count = int(report.data_overview.get("unprocessed_document_count", 0) or 0)
        analysis_stopped_early = bool(report.data_overview.get("analysis_stopped_early", False))
        if overall_document_summary:
            limit_text = effective_text_limit if effective_text_limit is not None else "No limit"
            analysis_summary_block = (
                f"{overall_document_summary}\n\n"
                f"- Raw input documents: **{raw_input_documents}**\n"
                f"- Filtered out as unrelated: **{filtered_out_documents}**\n"
                f"- Effective text limit: **{limit_text}**\n"
                f"- Processed documents: **{processed_documents}**\n"
                f"- Unprocessed documents: **{unprocessed_document_count}**\n"
                f"- Analysis stopped early: **{'Yes' if analysis_stopped_early else 'No'}**"
            )
        else:
            analysis_summary_block = "- Not available"
        document_summary_block = self._format_document_summaries(document_summaries)

        return f"""# Listed Company Sentiment Report

## 1. Basic Info

| Field | Value |
|------|------|
| Company | {report.company_name} |
| Stock Code | {report.stock_code or "N/A"} |
| Analysis Period | {report.analysis_period} |
| Generated At | {datetime.now().strftime("%Y-%m-%d %H:%M")} |
| Kept Documents | {report.data_overview.get("total_documents", 0)} |

## 2. Core Findings

### 1) Risk Assessment {risk_icon}

- Risk level: **{report.risk_assessment.risk_level.upper()}**
- Risk score: **{report.risk_assessment.risk_score:.2f} / 1.00**
- Confidence: **{report.risk_assessment.confidence:.0%}**

### 2) Sentiment Overview {sentiment_icon}

- Overall sentiment: **{report.overall_sentiment.get("label", "neutral")}**
- Sentiment score: **{report.overall_sentiment.get("score", 0):.2f}**
- Positive / Neutral / Negative: **{pos} / {neu} / {neg}**
- Negative ratio: **{(neg / total):.0%}**

### 3) Executive Summary

{report.summary}

### 4) Text-Analysis Summary

{analysis_summary_block}

### 5) Per-Document Summaries

{document_summary_block}

## 3. Risk Drivers And Key Events

### Risk Factors
{self._format_bullets(report.risk_assessment.risk_factors, empty_text="No significant risk factors.")}

### Negative Event Summary
{self._format_bullets(report.negative_event_summaries, empty_text="No high-risk negative events.")}

### Key Event List
{self._format_key_events(report.key_events)}

## 4. Recommendations
{self._format_numbered(report.recommendations, empty_text="No recommendation available.")}

## 5. Detailed Analysis
{report.detailed_analysis}

---
*Automatically generated by the sentiment analysis multi-agent system.*
"""

    def _generate_html_report(self, report: DecisionReport) -> str:
        markdown = self._generate_markdown_report(report)
        html_body = (
            markdown.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Sentiment Analysis Report - {report.company_name}</title>
  <style>
    body {{
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      max-width: 920px;
      margin: 24px auto;
      padding: 0 16px;
      line-height: 1.75;
      color: #1f2937;
      background: #f8fafc;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }}
  </style>
</head>
<body>
  <div class="card">{html_body}</div>
</body>
</html>
"""

    def _generate_json_report(self, report: DecisionReport) -> str:
        return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

    def _generate_chart_data(self, report: DecisionReport) -> List[Dict[str, Any]]:
        sentiment_distribution = report.sentiment_distribution or {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
        }
        timeline = report.sentiment_timeline or []
        event_counts = report.event_counts or {}

        top_events = sorted(event_counts.items(), key=lambda item: item[1], reverse=True)[:6]

        return [
            {
                "type": "pie",
                "title": "Sentiment Distribution",
                "data": [
                    {"name": "positive", "value": sentiment_distribution.get("positive", 0)},
                    {"name": "neutral", "value": sentiment_distribution.get("neutral", 0)},
                    {"name": "negative", "value": sentiment_distribution.get("negative", 0)},
                ],
            },
            {
                "type": "line",
                "title": "Sentiment Trend Timeline",
                "xAxis": [item.get("date", "") for item in timeline],
                "series": [
                    {"name": "positive", "data": [item.get("positive", 0) for item in timeline]},
                    {"name": "neutral", "data": [item.get("neutral", 0) for item in timeline]},
                    {"name": "negative", "data": [item.get("negative", 0) for item in timeline]},
                ],
            },
            {
                "type": "bar",
                "title": "Event Type Counts",
                "data": [{"name": self._format_event_label(name), "value": count} for name, count in top_events],
            },
            {
                "type": "gauge",
                "title": "Risk Score",
                "data": [{"value": report.risk_assessment.risk_score}],
            },
        ]

    def _format_bullets(self, items: List[str], empty_text: str) -> str:
        if not items:
            return f"- {empty_text}"
        return "\n".join(f"- {item}" for item in items if str(item).strip())

    def _format_numbered(self, items: List[str], empty_text: str) -> str:
        clean = [str(item).strip() for item in items if str(item).strip()]
        if not clean:
            return f"1. {empty_text}"
        return "\n".join(f"{idx}. {item}" for idx, item in enumerate(clean, start=1))

    def _format_key_events(self, events: List[Dict[str, Any]]) -> str:
        if not events:
            return "1. No key events."
        lines = []
        for idx, event in enumerate(events[:6], start=1):
            title = event.get("title") or event.get("trigger") or event.get("type") or "Unknown event"
            e_type = self._format_event_label(event.get("type", "unknown"))
            date = event.get("date") or ""
            impact = str(event.get("impact", "")).strip()
            impact = impact[:20000] + "..." if len(impact) > 20000 else impact
            impact_text = f" | {impact}" if impact else ""
            lines.append(f"{idx}. [{e_type}] {title} ({date}){impact_text}")
        return "\n".join(lines)

    def _format_document_summaries(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            return "1. No per-document summaries available."

        lines = []
        for idx, item in enumerate(items, start=1):
            title = str(item.get("title", "")).strip() or f"Document {idx}"
            summary = str(item.get("summary", "")).strip() or "No summary available."
            meta_parts = [
                str(item.get("publish_date", "")).strip(),
                str(item.get("source", "")).strip(),
                str(item.get("sentiment_label", "")).strip(),
            ]
            meta = ", ".join([part for part in meta_parts if part])
            if meta:
                lines.append(f"{idx}. {title} ({meta}): {summary}")
            else:
                lines.append(f"{idx}. {title}: {summary}")
        return "\n".join(lines)

    def create_gradio_interface(self) -> Any:
        """
        Create Gradio interface for end users.
        """
        try:
            import gradio as gr
        except ImportError:
            logger.error("Gradio is not installed. Run: pip install gradio")
            return None

        with gr.Blocks(title="Listed Company Sentiment Agent") as demo:
            default_date_from, default_date_to = self._default_date_range()
            gr.Markdown("# Listed Company Sentiment Agent")
            gr.Markdown(
                "Configure analysis inputs on the left, then review the multi-agent run "
                "trace and generated report on the right."
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    company_input = gr.Textbox(
                        label="Company Name",
                        placeholder="e.g. Kweichow Moutai / CATL / Ping An",
                    )
                    stock_input = gr.Textbox(
                        label="Stock Code (Optional)",
                        placeholder="e.g. 600519",
                    )
                    source_input = gr.CheckboxGroup(
                        label="Sources (Optional)",
                        choices=["news", "announcement", "social", "forum"],
                        value=["news", "announcement"],
                    )
                    date_from_input = gr.Textbox(
                        label="From Date",
                        placeholder="YYYY-MM-DD",
                        value=default_date_from,
                    )
                    date_to_input = gr.Textbox(
                        label="To Date",
                        placeholder="YYYY-MM-DD",
                        value=default_date_to,
                    )
                    max_effective_documents_input = gr.Number(
                        label="Effective Text Limit (Optional)",
                        minimum=1,
                        precision=0,
                        value=None,
                    )
                    output_format_input = gr.Radio(
                        label="Report Format",
                        choices=["markdown", "html", "json"],
                        value="markdown",
                    )
                    with gr.Row():
                        run_btn = gr.Button("Generate Report", variant="primary")
                        clear_btn = gr.Button("Clear")

                    gr.Examples(
                        examples=[
                            ["Kweichow Moutai", "600519", ["news", "announcement"], default_date_from, default_date_to, 5, "markdown"],
                            ["CATL", "300750", ["news", "social"], default_date_from, default_date_to, 8, "markdown"],
                            ["Ping An", "601318", ["news", "announcement", "forum"], default_date_from, default_date_to, 10, "json"],
                        ],
                        inputs=[
                            company_input,
                            stock_input,
                            source_input,
                            date_from_input,
                            date_to_input,
                            max_effective_documents_input,
                            output_format_input,
                        ],
                    )

                with gr.Column(scale=2):
                    agent_steps_output = gr.Chatbot(
                        label="Agent Trace",
                        type="messages",
                        height=340,
                        show_copy_button=True,
                    )
                    report_output = gr.Markdown(label="Report")
                    metadata_output = gr.JSON(label="Run Metadata")

            run_btn.click(
                fn=self._gradio_analyze,
                inputs=[
                    company_input,
                    stock_input,
                    source_input,
                    date_from_input,
                    date_to_input,
                    max_effective_documents_input,
                    output_format_input,
                ],
                outputs=[agent_steps_output, report_output, metadata_output],
            )
            clear_btn.click(
                fn=self._clear_gradio_outputs,
                outputs=[agent_steps_output, report_output, metadata_output],
            )

        demo.queue()
        return demo

    async def _gradio_analyze(
        self,
        company: str,
        stock: str,
        sources: List[str],
        date_from: Any,
        date_to: Any,
        max_effective_documents: Any,
        output_format: str,
    ):
        company = (company or "").strip()
        stock = (stock or "").strip() or None
        selected_sources = sources or ["news", "announcement", "social", "forum"]
        selected_date_range = self._resolve_date_range_inputs(date_from, date_to)
        normalized_limit = self._normalize_effective_limit_input(max_effective_documents)
        selected_format = (output_format or "markdown").strip().lower()
        started_at = datetime.now()

        if not company:
            messages = [
                self._new_chat_message(
                    role="assistant",
                    content="Please enter a company name first.",
                    title="Input Validation",
                    status="done",
                )
            ]
            yield messages, "Please enter a company name.", {"error": "company_name is required"}
            return

        if isinstance(selected_date_range, str):
            messages = [
                self._new_chat_message(
                    role="assistant",
                    content=selected_date_range,
                    title="Input Validation",
                    status="done",
                )
            ]
            yield messages, selected_date_range, {"error": "invalid_date_range"}
            return

        if isinstance(normalized_limit, str):
            messages = [
                self._new_chat_message(
                    role="assistant",
                    content=normalized_limit,
                    title="Input Validation",
                    status="done",
                )
            ]
            yield messages, normalized_limit, {"error": "invalid_effective_text_limit"}
            return

        if self.analysis_handler is None:
            messages = [
                self._new_chat_message(
                    role="assistant",
                    content="Analysis callback is not configured.",
                    title="System Error",
                    status="done",
                )
            ]
            yield messages, "Analysis callback is not configured.", {"error": "analysis_handler not configured"}
            return

        messages = [
            self._new_chat_message(
                role="user",
                content=f"Generate a sentiment risk report for {company}.",
            ),
            self._new_chat_message(
                role="assistant",
                content=(
                    f"Company: {company}\n"
                    f"Stock Code: {stock or 'N/A'}\n"
                    f"Sources: {', '.join(selected_sources)}\n"
                    f"From: {selected_date_range[0]}\n"
                    f"To: {selected_date_range[1]}\n"
                    f"Effective Text Limit: {normalized_limit if normalized_limit is not None else 'No limit'}\n"
                    f"Format: {selected_format}"
                ),
                title="Input Summary",
                status="done",
            ),
            self._new_chat_message(
                role="assistant",
                content="Starting multi-agent pipeline...",
                title="Pipeline Start",
                status="pending",
            ),
        ]
        yield messages, "Generating report...", {"status": "running"}

        try:
            pipeline_start_index = len(messages) - 1
            progress_queue: asyncio.Queue = asyncio.Queue()
            streamed_steps = 0

            async def _progress_callback(event: Dict[str, Any]) -> None:
                await progress_queue.put(event)

            request_payload = {
                "company_name": company,
                "stock_code": stock,
                "sources": selected_sources,
                "date_range": selected_date_range,
                "max_effective_documents": normalized_limit,
                "output_format": selected_format,
            }
            if self._analysis_handler_accepts_progress():
                request_payload["progress_callback"] = _progress_callback

            analysis_task = asyncio.create_task(self.analysis_handler(**request_payload))

            while True:
                if analysis_task.done():
                    break
                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                step_message = self._build_progress_message(event)
                if step_message:
                    messages.append(step_message)
                    streamed_steps += 1
                    yield messages, "Generating report...", {
                        "status": "running",
                        "latest_stage": event.get("stage", "Pipeline"),
                    }

            while not progress_queue.empty():
                event = progress_queue.get_nowait()
                step_message = self._build_progress_message(event)
                if step_message:
                    messages.append(step_message)
                    streamed_steps += 1

            result = await analysis_task

            messages[pipeline_start_index] = self._new_chat_message(
                role="assistant",
                content="Multi-agent pipeline completed.",
                title="Pipeline Start",
                status="done",
            )

            if streamed_steps == 0:
                for step in self._build_agent_trace_messages():
                    messages.append(step)

            elapsed = (datetime.now() - started_at).total_seconds()
            metadata = dict(result.get("metadata", {}))
            metadata.update(
                {
                    "sources": selected_sources,
                    "date_from": selected_date_range[0],
                    "date_to": selected_date_range[1],
                    "effective_text_limit": metadata.get("effective_text_limit", normalized_limit),
                    "output_format": selected_format,
                    "elapsed_seconds": round(elapsed, 2),
                }
            )

            report_text = result.get("report", "No report content generated.")
            messages.append(
                self._new_chat_message(
                    role="assistant",
                    content=(
                        f"Report generated in {elapsed:.2f}s, "
                        f"documents: {metadata.get('document_count', 0)} "
                        f"(filtered: {metadata.get('filtered_document_count', 0)})."
                    ),
                    title="Completed",
                    status="done",
                )
            )
            yield messages, self._format_report_for_display(report_text, selected_format), metadata
        except Exception as exc:  # pragma: no cover - interactive path
            logger.error("Gradio analysis failed: %s", exc)
            messages.append(
                self._new_chat_message(
                    role="assistant",
                    content=f"Execution failed: {exc}",
                    title="Pipeline Failed",
                    status="done",
                )
            )
            yield messages, f"Analysis failed: {exc}", {"error": str(exc)}

    def _build_agent_trace_messages(self) -> List[Dict[str, Any]]:
        if self.analysis_handler is None:
            return []

        system = getattr(self.analysis_handler, "__self__", None)
        if system is None:
            return []

        pipeline = [
            ("DataCollectionAgent", getattr(system, "collector", None)),
            ("TextAnalysisAgent", getattr(system, "analyst", None)),
            ("DecisionAgent", getattr(system, "decision", None)),
            ("PresentationAgent", getattr(system, "presenter", None)),
        ]

        messages = []
        for agent_name, agent in pipeline:
            state = getattr(agent, "state", None)
            if state is None:
                continue
            status = str(getattr(state, "status", "idle"))
            task = str(getattr(state, "task", "")).strip() or "No task description"
            output_hint = self._summarize_agent_output(getattr(state, "output", None))
            if output_hint:
                task = f"{task}\n{output_hint}"
            messages.append(
                self._new_chat_message(
                    role="assistant",
                    content=task,
                    title=agent_name,
                    status=self._state_to_gradio_status(status),
                )
            )
        return messages

    def _analysis_handler_accepts_progress(self) -> bool:
        if self.analysis_handler is None:
            return False
        try:
            signature = inspect.signature(self.analysis_handler)
        except Exception:
            return False

        if "progress_callback" in signature.parameters:
            return True
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _build_progress_message(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(event, dict):
            return None

        stage = str(event.get("stage", "Pipeline") or "Pipeline")
        status = str(event.get("status", "running") or "running")
        task = str(event.get("task", "")).strip()
        output_payload = event.get("output")

        lines = []
        if task:
            lines.append(task)

        if output_payload is not None:
            output_text = self._format_output_payload(output_payload)
            if output_text:
                lines.append(f"Output:\n```json\n{output_text}\n```")

        if not lines:
            lines.append(f"{stage} status: {status}")

        return self._new_chat_message(
            role="assistant",
            content="\n\n".join(lines),
            title=stage,
            status=self._state_to_gradio_status(status),
        )

    def _format_output_payload(self, payload: Any, max_chars: int = 50000) -> str:
        serializable = self._to_jsonable(payload)
        try:
            rendered = json.dumps(serializable, ensure_ascii=False, indent=2)
        except Exception:
            rendered = str(serializable)

        if len(rendered) > max_chars:
            remainder = len(rendered) - max_chars
            return f"{rendered[:max_chars]}\n... (truncated {remainder} chars)"
        return rendered

    def _to_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "to_dict"):
            try:
                return self._to_jsonable(value.to_dict())
            except Exception:
                pass
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(item) for item in value]
        if hasattr(value, "__dict__"):
            return {str(k): self._to_jsonable(v) for k, v in vars(value).items()}
        return str(value)

    def _state_to_gradio_status(self, status: str) -> str:
        normalized = (status or "").strip().lower()
        if normalized == "running":
            return "pending"
        return "done"

    def _summarize_agent_output(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, dict):
            if "total_count" in output:
                return f"Collected documents: {output.get('total_count', 0)}"
            if "analyzed_documents" in output:
                return (
                    f"Analyzed documents: {output.get('analyzed_documents', 0)}, "
                    f"filtered: {output.get('skipped_documents', 0)}"
                )
            if "charts" in output:
                return f"Chart count: {len(output.get('charts') or [])}"
            if "report" in output:
                report = str(output.get("report", ""))
                return f"Report length: {len(report)} chars"
        if isinstance(output, list):
            return f"Result entries: {len(output)}"

        risk_assessment = getattr(output, "risk_assessment", None)
        if risk_assessment is not None:
            level = getattr(risk_assessment, "risk_level", "unknown")
            score = getattr(risk_assessment, "risk_score", 0.0)
            return f"Risk level: {level}, score: {score:.2f}"
        return ""

    def _default_date_range(self) -> tuple[str, str]:
        end = datetime.now().date()
        start = end - timedelta(days=6)
        return (start.isoformat(), end.isoformat())

    def _normalize_date_input(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.date().isoformat()
        text = str(value).strip()
        if not text:
            return ""
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            return "__INVALID__"

    def _resolve_date_range_inputs(self, date_from: Any, date_to: Any) -> Any:
        normalized_from = self._normalize_date_input(date_from)
        normalized_to = self._normalize_date_input(date_to)
        if "__INVALID__" in {normalized_from, normalized_to}:
            return "Invalid date format. Please use YYYY-MM-DD."
        if not normalized_from or not normalized_to:
            return self._default_date_range()
        if normalized_from <= normalized_to:
            return (normalized_from, normalized_to)
        return (normalized_to, normalized_from)

    def _normalize_effective_limit_input(self, value: Any) -> Any:
        if value is None or value == "":
            return None
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return "Effective text limit must be a positive integer."
        if normalized < 1:
            return "Effective text limit must be a positive integer."
        return normalized

    def _format_report_for_display(self, report: str, output_format: str) -> str:
        if output_format == "json":
            return f"```json\n{report}\n```"
        if output_format == "html":
            return f"```html\n{report}\n```"
        return report

    def _new_chat_message(
        self,
        role: str,
        content: str,
        title: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": role, "content": content}
        metadata: Dict[str, Any] = {}
        if title:
            metadata["title"] = title
        if status:
            metadata["status"] = status
        if metadata:
            message["metadata"] = metadata
        return message

    def _format_event_label(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "Unknown"
        mapped = EVENT_TYPE_DISPLAY.get(text.lower())
        if mapped:
            return mapped
        mapped = EVENT_TYPE_DISPLAY.get(text)
        if mapped:
            return mapped
        return text.replace("_", " ").title()

    def _clear_gradio_outputs(self):
        return [], "", {}

def create_presentation_agent(
    llm: BaseLanguageModel,
    tools: Optional[List[BaseTool]] = None,
) -> Any:
    """
    Create a simplified ReAct presentation agent.
    """
    from langchain import hub

    try:
        from langchain.agents import create_react_agent
        try:
            from langchain.agents import AgentExecutor
        except ImportError:
            from langchain.agents.agent import AgentExecutor
    except ImportError as exc:  # pragma: no cover - optional path
        raise RuntimeError("Current langchain version does not support ReAct helper APIs.") from exc

    all_tools = tools or []
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=all_tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )
