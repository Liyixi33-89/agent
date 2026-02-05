# Transformers 微调项目 - 使用指南

## 📋 项目概述

这是一个基于 HuggingFace Transformers 的文本分类模型微调平台，支持：
- 使用 BERT 等预训练模型进行微调
- 支持 CSV 和 JSON 格式的数据集
- 提供 RESTful API 接口
- 集成 Gradio 演示界面
- 与 Ollama 集成进行模型推理

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
conda create -n transformers python=3.9
conda activate transformers

# 安装依赖
cd backend
pip install -r requirements.txt
```

### 2. 准备数据

支持两种数据格式：

**CSV 格式示例** (`data/train.csv`):
```csv
text,target
"这是一个正面评论",1
"这是一个负面评论",0
"这个产品很好用",1
```

**JSON 格式示例** (`data/train.json`):
```json
[
  {"text": "这是一个正面评论", "target": 1},
  {"text": "这是一个负面评论", "target": 0},
  {"text": "这个产品很好用", "target": 1}
]
```

### 3. 启动后端服务

```bash
cd backend
python main.py
```

服务将在 `http://localhost:8000` 启动

### 4. 启动前端界面

```bash
cd frontend
npm install
npm run dev
```

前端将在 `http://localhost:3000` 启动

### 5. 启动 Gradio 演示（可选）

```bash
cd backend
python gradio_demo.py
```

演示界面将在 `http://localhost:7860` 启动

## 🔧 API 接口文档

### 模型管理

- `GET /api/models` - 获取 Ollama 中的本地模型
- `POST /api/chat` - 与模型对话

### Agent 管理

- `POST /api/agents` - 创建新的 Agent 配置
- `GET /api/agents` - 获取所有 Agent

### 微调任务

- `POST /api/finetune` - 启动微调任务
- `GET /api/finetune/{task_id}` - 获取微调任务状态

### 微调请求示例

```json
POST /api/finetune
{
  "base_model": "bert-base-uncased",
  "dataset_path": "data/train.csv",
  "new_model_name": "my_custom_model",
  "epochs": 3,
  "learning_rate": 2e-5,
  "batch_size": 32,
  "max_length": 512,
  "text_column": "text",
  "label_column": "target"
}
```

## 📊 项目结构

```
agent/
├── backend/                 # Python 后端
│   ├── main.py             # FastAPI 主程序
│   ├── utils_data.py       # 数据加载和处理
│   ├── modeling_bert.py    # 模型结构定义
│   ├── trainer.py          # 训练逻辑
│   ├── gradio_demo.py      # Gradio 演示界面
│   └── requirements.txt    # Python 依赖
├── frontend/               # Next.js 前端
│   ├── src/app/            # 页面组件
│   ├── src/components/     # UI 组件
│   └── package.json        # Node.js 依赖
└── README.md               # 项目文档
```

## 🎯 微调方向建议

### 1. 虚假新闻检测
- **数据集**: Kaggle Fake News Dataset
- **模型**: BERT-base-uncased
- **应用**: 新闻真实性验证

### 2. 情感分析
- **数据集**: IMDB 电影评论
- **模型**: RoBERTa-base
- **应用**: 产品评论情感分类

### 3. 意图识别
- **数据集**: ATIS 航空旅行意图
- **模型**: DistilBERT
- **应用**: 智能客服系统

### 4. 文本分类
- **数据集**: AG News 新闻分类
- **模型**: BERT-large
- **应用**: 新闻自动分类

## 🔍 模型选型指南

| 模型 | 参数量 | 适用场景 | 优点 |
|------|--------|----------|------|
| BERT-base | 110M | 通用文本分类 | 平衡性能与速度 |
| BERT-large | 340M | 复杂分类任务 | 高精度 |
| RoBERTa | 125M | 专业领域分类 | 训练策略优化 |
| DistilBERT | 66M | 生产环境部署 | 轻量快速 |
| ALBERT | 12M | 资源受限环境 | 参数共享 |

## 📈 性能优化建议

### 1. 数据预处理
- 文本清洗和标准化
- 数据增强（同义词替换、回译）
- 类别平衡处理

### 2. 训练策略
- 学习率调度器（Linear Warmup）
- 早停机制（Early Stopping）
- 梯度裁剪（Gradient Clipping）

### 3. 模型优化
- 知识蒸馏（Knowledge Distillation）
- 量化压缩（Quantization）
- 剪枝（Pruning）

## 🛠️ 开发工具

- **Transformers**: HuggingFace 模型库
- **FastAPI**: 高性能 API 框架
- **Next.js**: React 前端框架
- **Gradio**: 快速构建演示界面
- **Ollama**: 本地模型推理

## 📚 学习资源

- [Transformers 官方文档](https://huggingface.co/docs/transformers)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Next.js 文档](https://nextjs.org/docs)
- [Gradio 文档](https://gradio.app/docs/)

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

MIT License

## 🆘 常见问题

### Q: 如何解决 CUDA 内存不足？
A: 减小 batch_size 或使用梯度累积

### Q: 如何提高模型准确率？
A: 增加训练数据、调整超参数、使用更大的模型

### Q: 如何部署到生产环境？
A: 使用 Docker 容器化部署，配合 Nginx 反向代理

## 📞 联系我们

如有问题，请提交 Issue 或联系项目维护者。