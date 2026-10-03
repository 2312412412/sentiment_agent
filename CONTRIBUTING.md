# Contributing

感谢你愿意改进这个项目。提交代码前请先确认以下事项。

## 本地开发

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

`.env` 只用于本地开发，不能提交到仓库。

## 代码规范

- 保持改动聚焦，避免无关重构。
- 新增外部 API 或数据源时，请在 README 或相关文档中补充配置说明。
- 涉及爬取逻辑时，请控制请求频率并尊重目标站点规则。
- 输出报告和测试数据不要包含真实密钥、私人数据或未授权内容。

## 提交前检查

```bash
python -m compileall .
```

如果添加了测试，请同时运行对应测试命令并在 PR 中说明。
