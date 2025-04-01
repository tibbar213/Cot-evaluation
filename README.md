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
│   ├── dataset_loader.py       # 数据集加载工具
│   ├── test_livebench.py       # LiveBench数据集测试
│   ├── test_log_only.py        # 日志记录测试
│   ├── vectorization/          # 向量化相关工具
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
│   ├── processed_datasets/ # 处理后的数据集
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

复制`.env.example`文件为`.env`，并填写您的API密钥：

```bash
cp .env.example .env
```

编辑`.env`文件：

```
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1  # 或其他API端点

# 模型配置
LLM_MODEL=gpt-4  # 用于生成回答的主模型
EVALUATION_MODEL=gpt-4  # 用于评估回答的模型
EMBEDDING_MODEL=text-embedding-3-large  # 向量嵌入模型
REASONING_MODEL=deepseek-ai/DeepSeek-V3  # 用于生成推理链的模型

# 为每个模型配置不同的API密钥和基础URL（可选）
# LLM模型配置
LLM_API_KEY=your_llm_api_key  
LLM_API_BASE=https://api.openai.com/v1

# 评估模型配置
EVALUATION_API_KEY=your_evaluation_api_key
EVALUATION_API_BASE=https://api.openai.com/v1

# 嵌入模型配置
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_API_BASE=https://api.openai.com/v1

# 推理模型配置
REASONING_API_KEY=your_reasoning_api_key
REASONING_API_BASE=https://api.deepseek.com/v1
```

当您需要为不同模型使用不同的API密钥和端点时（例如使用多家供应商的API），可以在`.env`文件中设置特定模型的API配置。如果未设置特定模型的API配置，系统将使用默认的`OPENAI_API_KEY`和`OPENAI_API_BASE`。

您可以根据需要更改模型配置，系统支持以下模型：

1. **主要模型 (LLM_MODEL)**：
   - OpenAI：`gpt-3.5-turbo`、`gpt-4`、`gpt-4-turbo`等
   - DeepSeek：`deepseek-ai/DeepSeek-V3`
   - 通义千问：`dashscope.qwen/Qwen-Max`
   - 文心一言：`qianfan.ernie-bot-4`

2. **评估模型 (EVALUATION_MODEL)**：推荐使用`gpt-4`或其他强大的模型以获得更准确的评估结果

3. **嵌入模型 (EMBEDDING_MODEL)**：
   - OpenAI：`text-embedding-3-large`、`text-embedding-3-small`
   - 哈工大：`BAAI/bge-m3`

4. **推理链生成模型 (REASONING_MODEL)**：推荐使用`deepseek-ai/DeepSeek-V3`或`gpt-4`以获得高质量的推理链

## 使用方法

### 使用多个数据集

本项目支持使用多个数据集进行评估，包括LiveBench等Hugging Face数据集：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math livebench/reasoning livebench/data_analysis
```

限制每个数据集的样本数量：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --max-samples-per-dataset 10
```

指定数据集分割：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --hf-split train
```

指定缓存目录：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --cache-dir data/custom_cache
```

从本地JSON文件加载数据集：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --local-json-dir data/processed_datasets
```

保存数据集到本地JSON文件：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --save-datasets --save-dir data/processed_datasets
```

同时使用多个参数组合：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --max-samples-per-dataset 100 --strategies combined --result-prefix math_test --save-datasets --rebuild-db
```

仅记录对话日志（不评估）与Hugging Face数据集结合：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/reasoning --log-only --max-samples-per-dataset 10 --result-prefix reasoning_logs
```

### 数据集独立向量数据库

为每个数据集使用单独的向量数据库，这对于不同领域的数据集特别有用：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math livebench/reasoning --separate-db
```

指定向量数据库目录：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --vector-db-dir data/custom_vector_store
```

### 结果前缀功能

使用结果前缀来区分不同评估任务的结果：

```bash
python src/main.py --result-prefix math_evaluation
```

这将在结果目录中创建带有前缀的结果文件，例如：
- `results/math_evaluation_eval_results.json`
- `results/conversation_logs/math_evaluation/`

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

查看特定前缀的评估结果摘要：

```bash
python src/main.py --summary-only --result-prefix math_evaluation
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

使用特定结果前缀的对话日志：

```bash
python src/batch_evaluation.py --result-prefix math_evaluation
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

## 多数据集评估工作流

以下是使用多个数据集进行完整评估的示例工作流：

### 1. 下载和准备多个数据集

```bash
# 下载LiveBench数据集并保存到本地
python src/main.py --use-hf-dataset --hf-dataset livebench/math livebench/reasoning livebench/data_analysis --max-samples-per-dataset 100 --save-datasets --save-dir data/processed_datasets
```

### 2. 为每个数据集创建单独的向量数据库

```bash
# 创建独立向量数据库
python src/main.py --use-hf-dataset --hf-dataset livebench/math livebench/reasoning livebench/data_analysis --separate-db --rebuild-db --max-samples-per-dataset 10
```

### 3. 分别对每个数据集进行评估

```bash
# 数学能力评估
python src/main.py --use-hf-dataset --hf-dataset livebench/math --max-samples-per-dataset 20 --strategies auto_reason combined --result-prefix math_evaluation

# 推理能力评估  
python src/main.py --use-hf-dataset --hf-dataset livebench/reasoning --max-samples-per-dataset 20 --strategies auto_reason combined --result-prefix reasoning_evaluation

# 数据分析能力评估
python src/main.py --use-hf-dataset --hf-dataset livebench/data_analysis --max-samples-per-dataset 20 --strategies auto_reason combined --result-prefix data_analysis_evaluation
```

### 4. 从本地文件加载并评估（可选）

如果已经保存了数据集到本地JSON文件，可以使用以下命令从本地加载：

```bash
# 从本地加载数学数据集并评估
python src/main.py --use-hf-dataset --hf-dataset livebench/math --local-json-dir data/processed_datasets --strategies zero_shot few_shot --max-questions 10 --result-prefix math_local_test
```

### 5. 查看不同领域的评估结果

```bash
# 查看数学评估结果摘要
python src/main.py --summary-only --result-prefix math_evaluation

# 查看推理能力评估结果摘要
python src/main.py --summary-only --result-prefix reasoning_evaluation

# 查看数据分析能力评估结果摘要
python src/main.py --summary-only --result-prefix data_analysis_evaluation
```

### 6. 查看和分析对话日志

对话日志存储在以下目录，按策略和数据集组织：
- 数学：`results/conversation_logs/math_evaluation/`
- 推理：`results/conversation_logs/reasoning_evaluation/`
- 数据分析：`results/conversation_logs/data_analysis_evaluation/`

您可以使用以下命令查看日志内容：

```bash
# 列出数学评估日志文件
ls results/conversation_logs/math_evaluation/combined/

# 查看特定日志文件
cat results/conversation_logs/math_evaluation/combined/math_question_id-timestamp.json
```

每个日志文件包含以下内容：
- 问题和参考答案
- 模型回答和推理过程
- 评估结果（准确率和推理质量）
- 元数据（策略详情、相似问题等）

## 评估指标

本项目使用以下指标评估模型性能：

1. **准确率**：模型回答的正确率
2. **推理质量**：评估模型推理过程的合理性和逻辑性
3. **鲁棒性**：在不同类型问题上的表现一致性
4. **效率**：生成答案所需的时间和计算资源

## 日志和结果文件结构

### 对话日志结构

每个对话日志文件（JSON格式）包含以下字段：

```json
{
  "question_id": "math_1",                   // 问题ID
  "question": "问题文本",                     // 问题内容
  "reference_answer": "参考答案",             // 标准答案
  "model_answer": "模型生成的答案",           // 模型回答
  "full_response": "完整的模型输出",          // 完整的模型响应
  "has_reasoning": true,                     // 是否包含推理过程
  "reasoning": "模型生成的推理过程",          // 推理过程
  "strategy": "combined",                    // 使用的策略名称
  "category": "arithmetic",                  // 问题类别
  "difficulty": "medium",                    // 问题难度
  "timestamp": 1649123456.789,               // 记录时间戳
  "session_id": "1649123456",                // 会话ID
  "evaluated": true,                         // 是否已评估
  "metadata": {                              // 元数据
    "strategy_details": {                     // 策略详情
      "name": "Auto-CoT + AutoReason",
      "description": "结合Auto-CoT和AutoReason的优势",
      "reasoning_model": "deepseek-ai/DeepSeek-V3",
      "num_examples": 2
    },
    "similar_questions": [                    // 相似问题列表
      ["0", "相似问题1", "答案1", 0.95],
      ["1", "相似问题2", "答案2", 0.85]
    ],
    "example_reasoning_chains": [             // 示例推理链
      {
        "question_id": "0",
        "question": "相似问题1",
        "answer": "答案1",
        "similarity": 0.95,
        "reasoning_chain": "推理过程"
      }
    ]
  },
  "evaluation_result": {                     // 评估结果（仅在evaluated=true时存在）
    "accuracy": {                             // 准确率评估
      "score": 1,                              // 分数
      "explanation": "评估解释"                 // 解释
    },
    "reasoning_quality": {                     // 推理质量评估
      "score": 9,                              // 分数（1-10）
      "explanation": "评估解释"                 // 解释
    }
  },
  "evaluation_timestamp": 1649123556.789      // 评估时间戳
}
```

### 评估结果文件结构

评估结果保存在`results/{result_prefix}_eval_results.json`中，结构如下：

```json
{
  "combined": [                               // 策略名称
    {
      "id": "math_1",                          // 问题ID
      "question": "问题文本",                   // 问题内容
      "reference_answer": "参考答案",           // 标准答案
      "model_answer": "模型生成的答案",         // 模型回答
      "reasoning": "模型生成的推理过程",        // 推理过程
      "category": "arithmetic",                // 问题类别
      "difficulty": "medium",                  // 问题难度
      "metrics": {                             // 评估指标
        "accuracy": {                           // 准确率评估
          "score": 1,                            // 分数
          "explanation": "评估解释"               // 解释
        },
        "reasoning_quality": {                   // 推理质量评估
          "score": 9,                            // 分数（1-10）
          "explanation": "评估解释"               // 解释
        }
      },
      "timestamp": 1649123456.789              // 记录时间戳
    }
  ],
  "zero_shot": [ /* ... 其他策略的评估结果 ... */ ],
  "timestamp": 1649123556.789,                // 结果文件更新时间戳
  "overall_metrics": {                        // 总体指标统计
    "combined": {                              // 策略名称
      "total_questions": 50,                    // 总问题数
      "metrics": {                              // 平均指标
        "accuracy": {
          "average_score": 0.85,                // 平均准确率
          "count": 50                           // 评估问题数
        },
        "reasoning_quality": {
          "average_score": 8.5,                 // 平均推理质量
          "count": 50                           // 评估问题数
        }
      },
      "difficulty_breakdown": {                 // 按难度分析
        "easy": { "count": 15, "accuracy": 0.96 },
        "medium": { "count": 20, "accuracy": 0.85 },
        "hard": { "count": 15, "accuracy": 0.72 }
      },
      "category_breakdown": {                   // 按类别分析
        "arithmetic": { "count": 30, "accuracy": 0.92 },
        "algebra": { "count": 20, "accuracy": 0.75 }
      }
    }
  },
  "detailed_results": { /* ... 详细结果统计 ... */ }
}
```

### 在代码中访问结果和日志

您可以通过以下方式在代码中加载和分析结果：

```python
import json
from pathlib import Path

# 加载评估结果
def load_results(result_prefix=None):
    result_file = Path('results')
    if result_prefix:
        result_file = result_file / f"{result_prefix}_eval_results.json"
    else:
        result_file = result_file / "eval_results.json"
    
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载对话日志
def load_conversation_logs(strategy, result_prefix=None):
    log_dir = Path('results/conversation_logs')
    if result_prefix:
        log_dir = log_dir / result_prefix
    
    log_dir = log_dir / strategy
    
    logs = []
    for log_file in log_dir.glob('*.json'):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs.append(json.load(f))
    
    return logs

# 示例：分析特定策略的结果
results = load_results('math_evaluation')
combined_accuracy = results['overall_metrics']['combined']['metrics']['accuracy']['average_score']
print(f"Combined策略的平均准确率：{combined_accuracy:.2f}")

# 示例：分析对话日志
combined_logs = load_conversation_logs('combined', 'math_evaluation')
reasoning_examples = [log for log in combined_logs if log['has_reasoning']]
print(f"包含推理过程的日志数量：{len(reasoning_examples)}")
```

## 注意事项

1. 需确保OpenAI API有足够的配额
2. 向量数据库可能需要较大存储空间
3. 评估过程可能耗费较多API调用，注意控制成本
4. 对于复杂问题，考虑设置较长的超时时间
5. 对话日志存储在`results/conversation_logs/`目录下，按策略名称分类
6. 批量评估可能需要较长时间，建议设置适当的批处理大小
7. 使用`--separate-db`参数时，会为每个数据集创建独立的向量数据库，有助于提高相似问题检索质量
8. 使用`--result-prefix`参数可以将不同评估任务的结果分开存储，便于后续分析和比较

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
