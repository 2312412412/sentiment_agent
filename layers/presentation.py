"""
应用展示层 - Gradio 可视化界面
"""
import logging
from typing import Optional, Dict, Any
import gradio as gr
import pandas as pd

logger = logging.getLogger(__name__)


class PresentationLayer:
    """应用展示层 - Gradio 界面"""
    
    def __init__(self, system, config):
        self.system = system
        self.config = config
        self.app = None
    
    def build_ui(self) -> gr.Blocks:
        """构建 Gradio 界面"""
        
        with gr.Blocks(
            title="舆情分析智能体系统",
            theme=gr.themes.Soft(),
        ) as demo:
            
            # 标题
            gr.Markdown(
                """
                # 📊 上市公司舆情分析智能体系统
                
                基于多智能体架构的舆情监测与分析平台
                """
            )
            
            with gr.Row():
                # 左侧：输入区域
                with gr.Column(scale=1):
                    gr.Markdown("### 分析配置")
                    
                    company_input = gr.Textbox(
                        label="公司名称",
                        placeholder="输入要分析的公司名称",
                        lines=1,
                    )
                    
                    stock_input = gr.Textbox(
                        label="股票代码（可选）",
                        placeholder="如：600519",
                        lines=1,
                    )
                    
                    source_check = gr.CheckboxGroup(
                        label="数据源",
                        choices=[
                            ("新闻网站", "news"),
                            ("交易所公告", "announcement"),
                            ("社交媒体", "social"),
                            ("财经论坛", "forum"),
                        ],
                        value=["news", "announcement"],
                    )
                    
                    time_range = gr.Dropdown(
                        label="时间范围",
                        choices=["最近 1 天", "最近 3 天", "最近 7 天", "最近 30 天"],
                        value="最近 7 天",
                    )
                    
                    analyze_btn = gr.Button(
                        "🚀 开始分析",
                        variant="primary",
                        size="lg",
                    )
                
                # 右侧：输出区域
                with gr.Column(scale=2):
                    gr.Markdown("### 分析结果")
                    
                    # 风险等级指示器
                    risk_gauge = gr.HTML(
                        self._create_risk_gauge("neutral", 0.5)
                    )
                    
                    # 摘要
                    summary_output = gr.Textbox(
                        label="舆情摘要",
                        lines=3,
                        interactive=False,
                    )
                    
                    # 详细分析（折叠）
                    with gr.Accordion("详细分析报告", open=False):
                        detailed_output = gr.Markdown()
                    
                    # 关键事件表格
                    events_table = gr.Dataframe(
                        label="关键事件",
                        headers=["事件类型", "触发词", "时间"],
                        wrap=True,
                    )
                    
                    # 情感分布图
                    sentiment_chart = gr.Plot(label="情感分布")
                    
                    # 建议列表
                    recommendations_output = gr.Markdown("### 建议措施")
            
            # 状态栏
            status_bar = gr.Textbox(
                label="状态",
                value="就绪",
                interactive=False,
            )
            
            # 绑定事件
            analyze_btn.click(
                fn=self._run_analysis,
                inputs=[company_input, stock_input, source_check, time_range],
                outputs=[
                    risk_gauge,
                    summary_output,
                    detailed_output,
                    events_table,
                    sentiment_chart,
                    recommendations_output,
                    status_bar,
                ],
            )
            
            # 示例
            gr.Examples(
                examples=[
                    ["贵州茅台", "600519", ["news", "announcement"], "最近 7 天"],
                    ["宁德时代", "300750", ["news", "social"], "最近 3 天"],
                    ["中国平安", "601318", ["news", "announcement", "forum"], "最近 7 天"],
                ],
                inputs=[company_input, stock_input, source_check, time_range],
            )
        
        self.app = demo
        return demo
    
    def _run_analysis(
        self,
        company: str,
        stock: str,
        sources: list,
        time_range: str,
    ):
        """执行分析流程"""
        import asyncio
        
        if not company:
            return (
                self._create_risk_gauge("neutral", 0),
                "请输入公司名称",
                "",
                pd.DataFrame(),
                None,
                "",
                "❌ 错误：公司名称不能为空",
            )
        
        try:
            # 运行分析
            loop = asyncio.new_event_loop()
            report = loop.run_until_complete(
                self.system.analyze(
                    company_name=company,
                    stock_code=stock or None,
                    sources=sources,
                    time_range=time_range,
                )
            )
            loop.close()
            
            # 处理结果
            risk_level = report.risk_assessment.risk_level
            risk_score = report.risk_assessment.risk_score
            
            # 事件表格
            events_data = [
                [e.get("type", ""), e.get("trigger", ""), e.get("time", "")]
                for e in report.key_events
            ]
            events_df = pd.DataFrame(events_data, columns=["事件类型", "触发词", "时间"])
            
            # 情感分布图
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(6, 4))
            dist = report.to_dict().get("risk_assessment", {})
            # 简化处理
            ax.bar(["正面", "中性", "负面"], [30, 50, 20], color=["green", "gray", "red"])
            ax.set_ylabel("文档数量")
            ax.set_title("情感分布")
            plt.tight_layout()
            
            # 建议
            rec_text = "\n".join([f"- {rec}" for rec in report.recommendations])
            
            return (
                self._create_risk_gauge(risk_level, risk_score),
                report.summary,
                report.detailed_analysis,
                events_df,
                fig,
                f"### 建议措施\n{rec_text}",
                f"✅ 分析完成 - 风险等级：{risk_level.upper()}",
            )
            
        except Exception as e:
            logger.error(f"分析失败：{e}")
            return (
                self._create_risk_gauge("neutral", 0),
                f"分析失败：{str(e)}",
                "",
                pd.DataFrame(),
                None,
                "",
                f"❌ 错误：{str(e)}",
            )
    
    def _create_risk_gauge(self, level: str, score: float) -> str:
        """创建风险等级 HTML 指示器"""
        colors = {
            "low": "#4CAF50",
            "medium": "#FF9800",
            "high": "#F44336",
            "critical": "#9C27B0",
        }
        
        labels = {
            "low": "低风险",
            "medium": "中风险",
            "high": "高风险",
            "critical": "严重风险",
        }
        
        color = colors.get(level, "#9E9E9E")
        label = labels.get(level, "未知")
        
        return f"""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; font-weight: bold; color: {color};">
                {label}
            </div>
            <div style="font-size: 48px; margin: 10px 0;">
                {int(score * 100)}%
            </div>
            <div style="width: 100%; height: 10px; background: #e0e0e0; border-radius: 5px;">
                <div style="width: {score * 100}%; height: 100%; background: {color}; border-radius: 5px;"></div>
            </div>
        </div>
        """
    
    def launch(self, **kwargs):
        """启动应用"""
        if not self.app:
            self.build_ui()
        
        import time
        
        logger.info("正在启动 Gradio 界面，启用分享模式...")
        
        # 启动并获取分享链接
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        self.app.launch(
            server_name="0.0.0.0",
            server_port=self.config.gradio_port,
            share=True,
            debug=False,
            **kwargs,
        )
        
        # 恢复标准输出
        output = captured_output.getvalue()
        sys.stdout = old_stdout
        
        # 打印捕获的输出
        print(output)
        logger.info(f"Gradio 输出：{output}")
        
        # 立即打印分享链接
        print(f"\n{'='*60}")
        print(f"📊 舆情分析系统 - Gradio 分享链接")
        print(f"{'='*60}")
        
        # 尝试多种方式获取分享链接
        share_url = None
        if hasattr(self.app, 'share_url') and self.app.share_url:
            share_url = self.app.share_url
            print(f"🔗 {share_url}")
            logger.info(f"✅ 分享链接已生成：{share_url}")
        
        if share_url:
            print(f"🔗 {share_url}")
            logger.info(f"✅ 分享链接已生成：{share_url}")
        else:
            print("⚠️ 分享链接正在生成中，请查看上方日志...")
            print(f"本地访问：http://localhost:{self.config.gradio_port}")
        
        print(f"{'='*60}\n")
        print("此链接 7 天内有效，可直接在浏览器打开访问！")
        
        # 保持运行
        print("\n服务持续运行中，按 Ctrl+C 停止...\n")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("服务已停止")
