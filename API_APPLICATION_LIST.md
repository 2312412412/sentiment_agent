# 需要人工申请的 API/Tools 清单

## 📋 概览

| 工具名称 | 用途 | 申请方式 | 免费额度 | 优先级 |
|----------|------|----------|----------|--------|
| Tavily Search | 新闻搜索 | 官网注册 | 1000 次/月 | 🔴 高 |
| NewsAPI | 新闻聚合 | 官网注册 | 500 次/天 | 🟡 中 |
| 巨潮资讯 | 上市公司公告 | 官网/爬虫 | 免费 | 🔴 高 |
| 微博开放平台 | 社交媒体监控 | 开发者平台申请 | 需审批 | 🟡 中 |
| Hugging Face | 情感分析模型 | 官网注册 | 免费 | 🟢 低 |
| FAISS | 向量检索 | 本地部署 | 免费 | 🟢 低 |

---

## 1. 数据采集类

### 1.1 Tavily Search API 🔴 高优先级

**用途**: 新闻搜索、网络信息检索

**申请方式**:
- 官网: https://tavily.com/
- 注册后获取 API Key
- 环境变量: `TAVILY_API_KEY`

**免费额度**: 
- Developer 计划: 1000 次搜索/月
- 超出后: $0.01/次

**替代方案**: 
- SearXNG (自部署，免费)
- NewsAPI (500 次/天免费)


---

### 1.3 巨潮资讯 🔴 高优先级

**用途**: 中国上市公司官方公告

**申请方式**:
- 官网: http://www.cninfo.com.cn/
- 无需 API Key，可直接爬取
- 建议: 控制请求频率，避免被封 IP

**替代方案**:
- 使用第三方封装库 (如 `cninfo` Python 包)

---

### 1.4 微博开放平台 🟡 中优先级

**用途**: 社交媒体舆情监控

**申请方式**:
- 官网: https://open.weibo.com/
- 创建应用，获取 App Key/Secret
- 需要实名认证 + 应用审核

**免费额度**:
- 基础接口：免费但有限制
- 高级接口：需商业合作

**注意**: 审核周期较长 (1-2 周)

---

## 2. NLP/分析类

### 2.1 Hugging Face 🟢 低优先级

**用途**: 情感分析、实体识别等 NLP 任务

**申请方式**:
- 官网: https://huggingface.co/
- 注册获取 Token (可选)
- 环境变量: `HUGGINGFACEHUB_API_TOKEN`

**免费额度**:
- 公开模型：免费使用
- API 推理：3000 字符/月免费
- 本地部署：完全免费

**推荐模型**:
- 情感分析: `nlptown/bert-base-multilingual-uncased-sentiment`
- 中文 NER: `bert-base-chinese`

---

### 2.2 Google Cloud NLP (可选)

**用途**: 高级文本分析

**申请方式**:
- Google Cloud Console 创建项目
- 启用 Natural Language API
- 获取服务账号密钥

**免费额度**:
- 5000 单位/月 (1 单位=1000 字符)

---

## 3. 向量检索类

### 3.1 FAISS 🟢 低优先级

**用途**: 向量相似度检索

**申请方式**: 
- **无需申请** - Facebook 开源库
- 安装: `pip install faiss-cpu`

**部署方式**:
- 本地部署 (推荐)
- 索引文件存储在项目目录

---

### 3.2 Milvus/Qdrant (可选)

**用途**: 大规模向量检索

**申请方式**:
- Milvus: https://milvus.io/ (开源，自部署)
- Qdrant: https://qdrant.tech/ (开源，自部署)

**免费额度**: 
- 自部署完全免费
- 云服务有免费层级

---

## 4. 推荐配置方案

### 最小可行方案 (免费为主)

```yaml
tools:
  news_search:
    provider: searxng  # 自部署，免费
    enabled: true
  
  announcement_fetch:
    provider: cninfo  # 巨潮资讯，免费
    enabled: true
  
  sentiment_analysis:
    provider: huggingface  # 免费模型
    model: nlptown/bert-base-multilingual-uncased-sentiment
    enabled: true
  
  entity_recognition:
    provider: spacy  # 开源库
    model: zh_core_web_sm
    enabled: true
  
  vector_search:
    provider: faiss  # 本地部署
    enabled: true
  
  social_media_monitor:
    provider: manual  # 暂时手动采集
    enabled: false  # 先禁用
```

### 生产方案 (含付费 API)

```yaml
tools:
  news_search:
    provider: tavily  # 稳定可靠
    api_key_env: TAVILY_API_KEY
    enabled: true
  
  announcement_fetch:
    provider: cninfo
    enabled: true
  
  sentiment_analysis:
    provider: huggingface
    api_key_env: HUGGINGFACEHUB_API_TOKEN
    enabled: true
  
  social_media_monitor:
    provider: weibo
    app_key_env: WEIBO_APP_KEY
    app_secret_env: WEIBO_APP_SECRET
    enabled: true
```

---

## 5. 环境变量配置模板

```bash
# .env 文件

# Tavily Search (推荐)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx

# NewsAPI (备选)
NEWSAPI_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Hugging Face (可选，用于私有模型)
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx

# 微博开放平台 (可选)
WEIBO_APP_KEY=xxxxxxxxxx
WEIBO_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx

# SearXNG (自部署)
SEARXNG_BASE_URL=http://localhost:8080
```

---

## 6. 申请时间估算

| API | 申请时间 | 审核周期 | 建议申请顺序 |
|-----|----------|----------|--------------|
| Tavily | 5 分钟 | 即时 | 1️⃣ |
| Hugging Face | 5 分钟 | 即时 | 2️⃣ |
| NewsAPI | 5 分钟 | 即时 | 3️⃣ |
| 巨潮资讯 | 0 分钟 | 无需申请 | 4️⃣ |
| 微博开放平台 | 30 分钟 | 1-2 周 | 5️⃣ (可选) |

---

## 7. 快速开始建议

### 第 1 步：立即可用的免费工具
1. 安装 SearXNG (Docker 一键部署)
2. 安装 Hugging Face 模型 (本地运行)
3. 安装 FAISS (pip install)
4. 编写巨潮资讯爬虫

### 第 2 步：注册 Tavily (5 分钟)
1. 访问 https://tavily.com/
2. 注册账号
3. 获取 API Key
4. 添加到 `.env` 文件

### 第 3 步：可选扩展
- 微博 API (如需要社交媒体监控)
- 付费升级 (如免费额度不够)

---

## 8. 成本估算

### 开发阶段 (免费)
- Tavily: 1000 次/月 免费
- Hugging Face: 免费
- FAISS: 免费
- SearXNG: 免费 (自部署)
- **总计: ¥0/月**

### 生产阶段 (预估)
- Tavily: $10-50/月 (取决于使用量)
- Hugging Face: $0-9/月 (可选)
- 服务器成本: ¥50-200/月
- **总计: ¥100-500/月**

---

**文档更新时间**: 2026-03-17
**项目**: 舆情分析智能体系统
