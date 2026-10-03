"""
使用示例 - 演示如何以编程方式使用舆情分析系统
"""
import asyncio

from config import config
from main import build_default_system


async def example_programmatic_usage():
    """示例：编程方式使用"""
    
    system = build_default_system(config)
    
    # 执行分析
    result = await system.analyze(
        company_name="贵州茅台",
        stock_code="600519",
        sources=["news", "announcement"],
    )
    
    # 查看结果
    print(f"\n{'='*50}")
    report = result["raw_report"]
    print(f"公司：{report.company_name}")
    print(f"风险等级：{report.risk_assessment.risk_level}")
    print(f"风险评分：{report.risk_assessment.risk_score:.2f}")
    print(f"整体情感：{report.overall_sentiment}")
    print(f"\n摘要：{report.summary}")
    print(f"\n建议：")
    for rec in report.recommendations:
        print(f"  - {rec}")
    print(f"{'='*50}\n")
    
    # 导出为字典
    return result


async def example_custom_analysis():
    """示例：自定义分析流程"""
    
    system = build_default_system(config)
    
    # 1. 单独使用数据采集层
    collection_result = await system.collector.execute(
        company_name="宁德时代",
        stock_code="300750",
        sources=["news"],
    )
    raw_docs = collection_result.get("documents", [])
    print(f"采集到 {len(raw_docs)} 篇文档")
    
    # 2. 单独使用文本分析层
    analyzed = await system.analyst.execute(
        documents=raw_docs,
        target_company="宁德时代",
    )
    print(f"分析完成 {len(analyzed)} 篇文档")
    
    # 3. 查看单篇分析结果
    if analyzed:
        doc = analyzed[0]
        print(f"\n首篇文档分析:")
        print(f"  实体：{len(doc.get('entities', []))} 个")
        print(f"  事件：{len(doc.get('events', []))} 个")
        print(f"  情感：{doc.get('sentiment', {}).get('label', 'N/A')}")
        print(f"  摘要：{doc.get('summary', '')[:100]}...")
    
    # 4. 使用决策层生成报告
    report = await system.decision.execute(
        company_name="宁德时代",
        stock_code="300750",
        analyzed_documents=analyzed,
        time_range="最近 7 天",
    )
    
    print(f"\n风险等级：{report.risk_assessment.risk_level}")
    


if __name__ == "__main__":
    print("舆情分析系统使用示例\n")
    print("运行：python example_usage.py\n")
    
    # 取消注释以运行示例
    # asyncio.run(example_programmatic_usage())
    # asyncio.run(example_custom_analysis())
