# 灰袍 — AIGC 文本检测系统

>White Water: AI-Generated Content Text Detector

## 项目概述

**灰袍**（White Water）是一个基于深度学习的 AIGC 文本检测系统，能够识别文本内容是由人工撰写还是由 AI（如 ChatGPT、GPT-4 等大语言模型）生成。系统采用多层检测策略，结合多种方法进行综合判断，提供概率评分和详细的分析报告。

### 核心功能

- **文本粘贴检测**：支持直接在浏览器中粘贴文本进行分析
- **文件上传检测**：支持上传 `.txt`、`.docx`、`.pdf` 文件
- **多维度分析**：中文 RoBERTa 模型 + 9 维中文特征 + 困惑度 + 爆发度
- **分段可视化**：对长文本进行滑动窗口分段，逐段展示 AI 生成概率
- **中文界面**：完整的中文用户界面和交互体验

---

## 技术原理

### 1. 中文 RoBERTa 分类器

**模型：** `Hello-SimpleAI/chatgpt-detector-roberta-chinese`

系统使用专门针对中文 ChatGPT 生成文本进行微调的中文 RoBERTa 分类器（chinese-roberta-wwm-ext 底座）。

**原理：**
- 输入中文文本经过中文分词器编码为 token ID 序列
- 经过 12 层 Transformer encoder 提取语义特征
- 分类头将 [CLS] token 的表示映射为二分类概率（人类/AI）
- 输出 0~1 之间的置信度分数

**注意：** 该模型在短文本和带有明确连接词标记的文本上表现良好，但在学术论文风格的长文本上区分度有限。因此系统将其作为辅助参考信号，以中文语言特征分析为主要判断依据。

**滑动窗口分块：** 对于超过 512 token 的长文本，使用 stride=256 的滑动窗口将文本切分为多个重叠片段，分别打分后取均值。

### 2. 困惑度分析（辅助方法）

**模型：** `distilgpt2`（DistilGPT-2）

困惑度（Perplexity）是语言模型评估文本"自然程度"的核心指标。

**原理：**
- 使用一个轻量级因果语言模型（DistilGPT-2，8200 万参数）计算文本的交叉熵损失
- 困惑度 = exp(交叉熵损失)，数值越低表示模型认为文本越"可预测"
- AI 生成的文本往往具有更低的困惑度（更符合语言模型自身的生成分布）
- 通过将原始困惑度值映射到 0~1 分数，与其他方法进行集成

**分数映射规则：**
| 困惑度范围 | 含义 | AI 可能性评分 |
|-----------|------|-------------|
| < 10 | 极度可预测 | 0.90 |
| 10-30 | 非常可预测 | 0.80 |
| 30-60 | 较可预测 | 0.65 |
| 60-100 | 中等 | 0.50 |
| 100-200 | 较不可预测 | 0.35 |
| 200-400 | 不可预测 | 0.20 |
| > 400 | 极度不可预测 | 0.10 |

### 3. 爆发度分析（辅助方法）

爆发度（Burstiness）衡量文本中句子长度的变化程度。

**原理：**
- 将文本按句切分，统计每句的单词/词组数量
- 计算句子长度的变异系数（CV = 标准差 / 均值）
- 人类写作：句子长度波动大，CV 较高（~0.5-1.2）
- AI 写作：句子长度趋于均匀，CV 较低（~0.2-0.5）
- 将 CV 映射到 0~1 的 AI 可能性评分

### 4. 中文语言特征分析（核心方法）

针对中文论文场景，系统设计了 9 维语言特征分析：

**特征维度：**
| 特征 | 说明 | AI 写作表现 |
|------|------|-----------|
| 连接词密度 | 首先/其次/最后/此外/因此 等使用频率 | 显著偏高 |
| 平衡标记 | 一方面/另一方面/换言之 等 | 出现频率高 |
| 开篇模式词 | 以"在/从/通过/随着/基于"等开头的句子比例 | >30% |
| 术语密度 | 赋能/落地/维度/领域/显著 等高频词密度 | >4词/百字 |
| 论证标记 | 研究表明/数据表明/由此可见 等 | 频次高 |
| 连词密度 | 然而/因此/并且/同时 等 | 密度偏高 |
| 段落均句数 | 每段平均句子数 | 偏均匀（2-4句/段） |
| 段落长度CV | 段落间长度变异系数 | 偏低（较均匀） |
| 标点结构化 | 破折号、冒号使用密度 | 偏高 |

### 5. 集成评分

四种方法通过加权平均进行集成：

```
最终分数 = 0.15 × RoBERTa分数 + 0.05 × 困惑度分数 + 0.15 × 爆发度分数 + 0.65 × 中文特征分数
```

- 最终分数 > 0.5 → 判定为"AI 生成"
- 最终分数 ≤ 0.5 → 判定为"人类写作"
- 置信度 = |最终分数 - 0.5| × 2（归一化到 0-100%）

中文特征权重最高（0.65），因为在中文论文场景下，语言特征模式比单一分类器更可靠。当多维度特征同时触发时（≥3个子特征得分>0.5），系统会自动校准提升得分。RoBERTa（0.15）和爆发度（0.15）提供补充视角，困惑度（0.05）因使用英文模型仅作微弱参考。

---

## 系统架构

```
┌──────────────────────────────────────────────┐
│                    浏览器                      │
│           (Tailwind CSS + htmx + JS)          │
└──────────────────┬───────────────────────────┘
                   │ HTTP
┌──────────────────▼───────────────────────────┐
│                Nginx (:80)                    │
│     静态文件缓存 / 反向代理 / 速率限制          │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│            FastAPI (Uvicorn :8000)             │
│  ┌───────────┐  ┌────────────┐  ┌──────────┐ │
│  │ /api/detect│  │/api/detect │  │/api/health│ │
│  │   (文本)   │  │  /file     │  │  (健康)   │ │
│  └─────┬─────┘  └─────┬──────┘  └──────────┘ │
│        │               │                      │
│  ┌─────▼───────────────▼──────────────────┐   │
│  │         Detection Pipeline             │   │
│  │  ┌──────────┐ ┌─────────┐ ┌─────────┐  │   │
│  │  │ Text      │ │Clean &  │ │Analyze  │  │   │
│  │  │ Extraction│ │Chunk    │ │& Score  │  │   │
│  │  └──────────┘ └─────────┘ └─────────┘  │   │
│  └────────────────────────────────────────┘   │
│                                                │
│  ┌─────────────────────────────────────────┐  │
│  │            Model & Feature Layer            │  │
│  │  ┌──────────────────┐ ┌───────────────┐     │  │
│  │  │ RoBERTa Chinese  │ │ DistilGPT-2   │     │  │
│  │  │  (355M params)   │ │ (82M params)  │     │  │
│  │  └──────────────────┘ └───────────────┘     │  │
│  │  ┌──────────────────────────────────────┐   │  │
│  │  │  Chinese Feature Analyzer (9-dim)    │   │  │
│  │  └──────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
```

### 目录结构

```
aigc/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI 应用入口
│   │   ├── config.py                    # 配置管理 (Pydantic Settings)
│   │   ├── models/
│   │   │   └── schemas.py               # 请求/响应 Pydantic 模型
│   │   ├── routers/
│   │   │   ├── detect.py                # POST /api/detect
│   │   │   ├── file_detect.py           # POST /api/detect/file
│   │   │   └── health.py                # GET /api/health
│   │   ├── services/
│   │   │   ├── detector_service.py      # 模型加载/推理
│   │   │   ├── statistical_service.py   # 统计方法
│   │   │   └── text_processor.py        # 文本清洗/分块
│   │   ├── utils/
│   │   │   ├── file_parser.py           # 文件文本提取
│   │   │   └── scoring.py               # 评分配置
│   │   ├── templates/                   # Jinja2 模板
│   │   └── static/                      # CSS/JS 静态资源
│   ├── requirements.txt
│   └── Dockerfile
├── nginx/
│   └── nginx.conf                       # Nginx 配置
├── docker-compose.yml                   # 一键部署
└── README.md
```

---

## 技术栈

### 后端

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| Web 框架 | FastAPI 0.136+ | 基于 Starlette，支持异步，自动 OpenAPI 文档 |
| ASGI 服务器 | Uvicorn 0.46+ | 高性能，支持多 worker |
| 深度学习 | PyTorch 2.11 + Transformers 4.57 | 模型推理引擎 |
| 分类模型 | chatgpt-detector-roberta-chinese | 355M 参数，中文 ChatGPT 检测专用 |
| 统计方法 | 中文语言特征分析 (9维度) | 连接词、术语、句式结构等特征 |
| 语言模型 | DistilGPT-2 | 82M 参数，困惑度计算 |
| 配置管理 | Pydantic Settings | 类型安全的环境变量管理 |
| 文件解析 | python-docx + PyMuPDF | .docx / .pdf 文本提取 |

### 前端

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 模板引擎 | Jinja2 | 服务端渲染 |
| CSS 框架 | Tailwind CSS (CDN) | 零构建步骤，Utility-First |
| 动态交互 | htmx + Vanilla JS | 轻量级 AJAX |
| Logo | 内联 SVG | 矢量图形，无需额外资源 |

### 部署

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 容器化 | Docker + Docker Compose | 一键部署，环境隔离 |
| 反向代理 | Nginx | 静态缓存、速率限制、负载均衡 |
| SSL | Let's Encrypt / Certbot | 免费 HTTPS 证书 |

---

## 本地开发

### 环境要求

- Python 3.12+（推荐）或 3.14（已验证）
- 至少 4GB 可用内存（模型加载需要）
- 首次运行需下载约 2GB 模型文件

### 安装运行

```bash
# 1. 安装依赖
cd backend
pip install -r requirements.txt

# 2. 启动服务
uvicorn app.main:app --reload --port 8000

# 3. 打开浏览器
# http://localhost:8000
```

首次启动会从 HuggingFace Hub 下载模型文件（约需 2-5 分钟，取决于网络）。模型会被缓存到 `./models/` 目录，后续启动只需 2-3 秒。

---

## Docker 部署

### 开发环境

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f backend

# 停止
docker compose down
```

### 生产环境 VPS 部署

```bash
# 1. 服务器安装 Docker
curl -fsSL https://get.docker.com | sh

# 2. 克隆项目
git clone <your-repo-url> /opt/aigc
cd /opt/aigc

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 4. 启动服务
docker compose up -d

# 5. 配置 HTTPS（可选）
# 参考 nginx.conf 中的 SSL 配置
# 使用 certbot 获取 Let's Encrypt 证书
```

---

## API 文档

启动服务后访问 `http://localhost:8000/docs` 查看 Swagger UI 接口文档。

### POST /api/detect — 文本检测

**请求：**
```json
{
    "text": "要检测的文本内容..."
}
```

**响应：**
```json
{
    "status": "success",
    "data": {
        "overall_score": 0.8723,
        "classification": "AI-generated",
        "confidence": 0.7446,
        "breakdown": {
            "roberta": { "score": 0.9200 },
            "perplexity": { "perplexity": 12.45, "score": 0.8000 },
            "burstiness": { "cv": 0.2658, "score": 0.8228 },
            "chinese_features": {
                "marker_density": 0.83, "opener_ratio": 0.50,
                "buzzword_density": 5.2, "sentences_per_para": 2.8,
                "evidence_density": 0.33, "conj_density": 0.50,
                "score": 0.6500
            }
        },
        "segments": [
            {
                "start": 0,
                "end": 512,
                "text_preview": "文本前100字符预览...",
                "score": 0.9200,
                "label": "AI-generated"
            }
        ],
        "metadata": {
            "text_length": 1024,
            "chunks_analyzed": 2,
            "processing_time_ms": 1250.50
        }
    }
}
```

### POST /api/detect/file — 文件检测

**请求：** `multipart/form-data`，字段名 `file`

**支持格式：** `.txt` / `.docx` / `.pdf`

**响应格式：** 与 `/api/detect` 相同，额外返回 `metadata.source_file`

### GET /api/health — 健康检查

```json
{
    "status": "healthy",
    "version": "2.0.0",
    "models_loaded": ["Hello-SimpleAI/chatgpt-detector-roberta-chinese", "distilgpt2"],
    "gpu_available": false,
    "uptime_seconds": 3600.0
}
```

---

## 局限性说明

1. **概率性判断**：AI 检测本质上是概率推断，不应将其结果作为判定学术诚信或内容原创性的唯一依据。

2. **短文本困难**：对于极短文本（< 50 字），统计特征不足，模型判断可靠性下降。

3. **模型时效性**：检测模型的训练数据截止于 ChatGPT 早期版本。随着新模型（GPT-4、Claude 等）的演进，检测准确率可能下降。建议定期更新模型。

4. **对抗性文本**：经过改写、翻译或刻意伪装的 AI 文本可能绕过检测。这不是安全审计工具。

5. **语言偏向**：当前系统针对中文文本优化。对英文及其他语言的检测能力有限。

6. **误判可能**：
   - **假阳性**（False Positive）：将人类写作误判为 AI
   - **假阴性**（False Negative）：将 AI 写作误判为人类

7. **资源消耗**：模型推理需要较大内存（~2GB），CPU 推理速度较慢。生产环境建议使用 GPU 加速或 ONNX 导出优化。

---

## 未来规划

- [x] 中文专用模型 + 中文语言特征分析（已完成）
- [ ] GPU 推理支持 + ONNX 模型导出加速
- [ ] 用户账户系统 + 检测历史记录
- [ ] 批量检测 API + 异步任务队列
- [ ] PDF 标注输出（高亮可疑段落）
- [ ] Claude API / GPTZero API 集成作为第三方验证
- [ ] 仪表板 + 统计图表可视化
- [ ] 浏览器插件（Chrome Extension）

---

## License

MIT License
