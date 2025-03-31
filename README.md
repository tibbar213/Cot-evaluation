# LLM评估项目

本项目旨在评估大型语言模型（LLM）的基础能力和思维链（Chain of Thought, CoT）能力。通过实现多种CoT策略，对比分析不同策略对模型性能的影响，从而深入了解LLM的推理能力和优化方向。

## 项目结构

```
llm-evaluation/
├── docs/                  # 文档
│   └── requirements.md    # 需求文档
├── src/                   # 源代码
│   ├── config.py          # 配置文件
│   ├── models.py          # 模型接口
│   ├── vector_db.py       # 向量数据库接口
│   ├── conversation_logger.py  # 对话日志记录器
│   ├── batch_evaluation.py     # 批量评估工具
│   ├── strategies/        # CoT策略实现
│   │   ├── __init__.py    # 策略导出
│   │   ├── base.py        # 策略基类
│   │   ├── baseline.py    # Baseline（无CoT）
│   │   ├── zero_shot.py   # Zero-shot CoT
│   │   ├── few_shot.py    # Few-shot CoT
│   │   ├── auto_cot.py    # Auto-CoT
│   │   ├── auto_reason.py # AutoReason
│   │   └── combined.py    # Auto-CoT + AutoReason
│   ├── evaluation.py      # 评估框架
│   └── main.py            # 主程序
├── data/                  # 测试数据
│   ├── questions.json     # 测试问题集
│   └── vector_store/      # 向量数据库存储
├── results/               # 评估结果
│   ├── eval_results.json  # 评估结果输出
│   └── conversation_logs/ # 对话日志存储
├── examples/              # 示例代码
│   └── simple_evaluation.py # 简单评估示例
├── requirements.txt       # 依赖库
├── .env.example           # 环境变量示例
└── README.md              # 项目说明
```

## 思维链（CoT）策略

本项目实现了以下CoT策略：

### 1. Zero-shot CoT

在提示的最后添加"Let's think step by step."，引导模型进行逐步推理。

**示例**：
```
Q: 2+2等于多少？
A: Let's think step by step.
```

### 2. Few-shot CoT

使用向量数据库存储示例问题及其答案。对于每个测试问题：
1. 使用BAAI/bge-m3向量模型将问题转换为向量
2. 在向量数据库中搜索k个最相似的问题
3. 将这些相似问题及其答案作为示例，添加到提示中

### 3. Auto-CoT

与Few-shot CoT类似，但为相似问题生成CoT推理过程：
1. 使用BAAI/bge-m3向量模型将问题转换为向量
2. 在向量数据库中搜索k个最相似的问题
3. 为这些相似问题生成CoT推理过程
4. 将这些相似问题及其生成的CoT推理过程作为示例，添加到提示中

### 4. AutoReason

对于每个测试问题，使用强模型生成详细的推理链，并将其作为提示的一部分。

### 5. Auto-CoT + AutoReason

结合Auto-CoT和AutoReason的优势：
1. 使用BAAI/bge-m3向量模型将问题转换为向量
2. 在向量数据库中搜索k个最相似的问题
3. 使用DeepSeek-R1为这些相似问题生成CoT推理过程
4. 将这些相似问题及其生成的CoT推理过程作为示例，添加到提示中

### 6. Baseline（无CoT）

作为基准实验，直接向模型提问，不添加任何CoT提示。

## 安装与配置

### 依赖库安装

```bash
pip install -r requirements.txt
```

### 环境变量配置

复制`.env.example`文件为`.env`，并填写您的OpenAI API密钥：

```bash
cp .env.example .env
```

编辑`.env`文件：

```
OPENAI_API_KEY=your_api_key_here
```

## 使用方法

### 初始化向量数据库

首次运行时，需要初始化向量数据库：

```bash
python src/main.py --rebuild-db
```

### 运行评估

运行所有策略的评估：

```bash
python src/main.py
```

运行特定策略的评估：

```bash
python src/main.py --strategies zero_shot few_shot
```

限制评估的问题数量：

```bash
python src/main.py --max-questions 5
```

评估特定问题：

```bash
python src/main.py --question-ids math_1 math_2 math_3
```

### 查看评估结果摘要

```bash
python src/main.py --summary-only
```

### 运行简单示例

```bash
python examples/simple_evaluation.py
```

## 新增功能：对话日志与批量评估

### 仅记录对话日志而不评估

使用`--log-only`参数可以只记录对话日志而不进行评估：

```bash
python src/main.py --log-only
```

您可以使用`--session-id`参数指定会话ID以便后续跟踪：

```bash
python src/main.py --log-only --session-id my-session-1
```

### 查看所有会话

使用batch_evaluation.py中的`--list-sessions`参数查看所有会话：

```bash
python src/batch_evaluation.py --list-sessions
```

### 批量评估存储的对话日志

评估所有未评估的对话日志：

```bash
python src/batch_evaluation.py
```

评估特定策略的对话日志：

```bash
python src/batch_evaluation.py --strategy zero_shot
```

评估特定会话的对话日志：

```bash
python src/batch_evaluation.py --session 1648795612
```

设置批处理大小：

```bash
python src/batch_evaluation.py --batch-size 20
```

### 生成会话报告

```bash
python src/batch_evaluation.py --report 1648795612
```

## 工作流程示例

1. 记录对话日志（不评估）：
```bash
python src/main.py --log-only --strategies zero_shot few_shot --max-questions 10
```

2. 查看所有会话：
```bash
python src/batch_evaluation.py --list-sessions
```

3. 批量评估特定会话：
```bash
python src/batch_evaluation.py --session 1648795612
```

4. 生成会话报告：
```bash
python src/batch_evaluation.py --report 1648795612
```

## 评估指标

本项目使用以下指标评估模型性能：

1. **准确率**：模型回答的正确率
2. **推理质量**：评估模型推理过程的合理性和逻辑性
3. **鲁棒性**：在不同类型问题上的表现一致性
4. **效率**：生成答案所需的时间和计算资源

## 注意事项

1. 需确保OpenAI API有足够的配额
2. 向量数据库可能需要较大存储空间
3. 评估过程可能耗费较多API调用，注意控制成本
4. 对于复杂问题，考虑设置较长的超时时间
5. 对话日志存储在`results/conversation_logs/`目录下，按策略名称分类
6. 批量评估可能需要较长时间，建议设置适当的批处理大小

# 数学问题解答系统

这是一个基于大语言模型的数学问题解答系统，支持多种解答策略和评估方法。

## 功能特点

- 支持多种解答策略（Baseline、CoT、RAG等）
- 自动评估答案准确性
- 支持对话日志记录和批量评估
- 支持数据集向量化和相似问题检索

## 安装

1. 克隆仓库：
```bash
git clone <repository_url>
cd <repository_name>
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
创建 `.env` 文件并设置以下变量：
```
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base_url
```

## 使用方法

### 1. 运行测试

使用以下命令运行测试：
```bash
python src/test_log_only.py --max-questions 3
```

### 2. 批量评估对话日志

使用以下命令评估特定会话的对话日志：
```bash
python src/batch_evaluation.py --session <session_id>
```

### 3. 生成会话报告

使用以下命令生成会话报告：
```bash
python src/batch_evaluation.py --report <session_id>
```

### 4. 向量化数据集

使用以下命令将问题集向量化：
```bash
python src/vectorize_dataset.py --questions data/questions.json --batch-size 10 --output data/vector_store
```

参数说明：
- `--questions`: 问题集文件路径
- `--batch-size`: 批处理大小，默认为10
- `--output`: 向量存储输出目录，默认为 `data/vector_store`

## 项目结构

```
.
├── data/
│   ├── questions.json      # 问题集
│   └── vector_store/      # 向量存储目录
├── results/
│   ├── conversation_logs/ # 对话日志
│   └── eval_results.json  # 评估结果
├── src/
│   ├── models.py         # 模型接口
│   ├── strategies.py     # 解答策略
│   ├── conversation_logger.py  # 对话日志记录器
│   ├── batch_evaluation.py    # 批量评估工具
│   ├── test_log_only.py      # 测试脚本
│   ├── vector_store.py       # 向量存储类
│   └── vectorize_dataset.py  # 数据集向量化工具
├── requirements.txt     # 项目依赖
└── README.md          # 项目说明
```

## 注意事项

1. 确保已正确配置API密钥和基础URL
2. 向量化过程可能需要较长时间，建议使用适当的批处理大小
3. 向量存储目录会自动创建，无需手动创建

## 新增功能：向量化模块

### 向量化模块结构

```
src/vectorization/
├── __init__.py           # 模块初始化文件
├── vector_store.py       # 向量存储类
├── vectorize_dataset.py  # 数据集向量化工具
└── search_similar.py     # 相似问题检索工具
```

### 向量化数据集

将问题集转换为向量形式并存储：

```bash
python src/vectorization/vectorize_dataset.py --questions data/questions.json --batch-size 10
```

参数说明：
- `--questions`: 问题集文件路径
- `--batch-size`: 批处理大小
- `--output`: 向量存储输出目录（默认：data/vector_store）

### 相似问题检索

搜索与给定问题最相似的问题：

```bash
python src/vectorization/search_similar.py --query "你的问题" --k 3
```

参数说明：
- `--query`: 查询问题
- `--k`: 返回的相似问题数量（默认：3）
- `--vector-store`: 向量存储目录（默认：data/vector_store）

### 功能特点

1. 使用BAAI/bge-m3模型进行文本向量化
2. 使用FAISS进行高效的向量索引和检索
3. 支持批量处理和进度显示
4. 保存完整的问题元数据（类别、难度等）
5. 相似度计算准确，结果展示清晰

### 使用场景

1. 构建Few-shot示例库
2. 相似问题推荐
3. 问题聚类分析
4. 难度分布分析
5. 评估数据集覆盖度
