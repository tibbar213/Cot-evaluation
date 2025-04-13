# LLM思维链(CoT)评估项目

本项目旨在评估大型语言模型（LLM）的思维链（Chain of Thought, CoT）能力。通过实现多种CoT策略，对比分析不同策略对模型推理性能的影响，从而深入了解LLM的推理能力和优化方向。

## 项目结构

```
llm-evaluation/
├── 毕业论文.md               # 毕业论文说明
├── README.md                 # 项目说明
├── requirements.txt          # 依赖库
├── .env.example              # 环境变量示例
├── src/                     # 源代码
│   ├── config.py
│   ├── models.py
│   ├── vector_db.py
│   ├── conversation_logger.py
│   ├── batch_evaluation.py
│   ├── dataset_loader.py
│   ├── evaluation.py
│   ├── main.py
│   ├── sqlite_backup.py
│   ├── backup_manager.py
│   └── strategies/          # CoT策略实现
│       ├── __init__.py
│       ├── base.py
│       ├── baseline.py       # Baseline（无CoT）
│       ├── zero_shot.py      # Zero-shot CoT
│       ├── few_shot.py       # Few-shot CoT
│       ├── auto_cot.py       # Auto-CoT
│       ├── auto_reason.py    # AutoReason
│       └── combined.py       # Auto-CoT + AutoReason
├── data/                    # 测试数据及备份数据库等
├── results/                 # 评估结果与对话日志
├── web/                     # 可视化评估界面和Web服务
└── venv/                    # 虚拟环境（通常忽略）
```

## Web界面使用指南
![image](https://github.com/user-attachments/assets/8d3c6cbc-e335-43fa-b1bc-d22b3c21c70d)
![image](https://github.com/user-attachments/assets/e66f9acd-c53e-4ab1-a924-c3d148b6dd36)
![image](https://github.com/user-attachments/assets/132b1c86-2c8e-4d7d-be1a-f00ab547b929)
![image](https://github.com/user-attachments/assets/617b0e11-8b7b-4b91-990e-3446f767b037)


### 启动Web服务

本项目提供了可视化Web界面，可通过不同方式加载评估数据：

1. 从SQLite数据库加载：
```bash
python web/server.py --use-sqlite --db-path data/backup.db
```

2. 从评估结果JSON文件加载：
```bash
python web/server.py --use-json --json-path results/eval_results.json
```

3. 从对话日志目录加载（推荐）：
```bash
python web/server.py --use-logs --logs-path results/conversation_logs
```

### 启动前端应用

```bash
cd web
npm install  # 首次使用时安装依赖
npm run dev
```

打开浏览器访问 http://localhost:3000 查看评估结果。

### Web界面功能

Web界面提供以下功能：
1. **数据集、模型和策略选择**：动态加载实际存在的选项
2. **评估统计**：显示总评估记录数和平均准确率
3. **策略对比**：通过图表和表格比较不同策略的性能
4. **模型对比**：比较不同模型在相同策略下的表现
5. **详细评估记录**：查看所有评估记录，包括问题、回答和评分
6. **详情查看**：点击详情按钮可查看完整的问题、答案和评估解释

## 思维链（CoT）策略

本项目实现了以下CoT策略：

### 1. Zero-shot CoT

在提示的最后添加预设引导语，引导模型进行逐步推理。

**示例**：
```bash
Q: 2+2等于多少？
A: 让我们按照步骤思考：首先计算2+2，得到4；因此，答案是4。
```

### 2. Few-shot CoT

使用向量数据库存储示例问题及其答案。对于每个测试问题：
1. 使用 BAAI/bge-m3 向量模型将问题转换为向量
2. 在向量数据库中搜索 k 个最相似的问题
3. 将这些相似问题及其答案作为示例，添加到提示中

### 3. Auto-CoT

与 Few-shot CoT 类似，但为相似问题生成CoT推理过程：
1. 使用 BAAI/bge-m3 向量模型将问题转换为向量
2. 在向量数据库中搜索 k 个最相似的问题
3. 为这些相似问题生成CoT推理过程
4. 将这些相似问题及其生成的CoT推理过程作为示例，添加到提示中

### 4. AutoReason

对于每个测试问题，使用强模型生成详细的推理链，并将其作为提示的一部分。

### 5. Auto-CoT + AutoReason

结合 Auto-CoT 和 AutoReason 的优势：
1. 使用 BAAI/bge-m3 向量模型将问题转换为向量
2. 在向量数据库中搜索 k 个最相似的问题
3. 使用 **deepseek-ai/DeepSeek-V3** 为这些相似问题生成CoT推理过程
4. 将这些相似问题及其生成的CoT推理过程作为示例，添加到提示中

### 6. Baseline（无CoT）

作为基准实验，直接向模型提问，不添加任何CoT提示。

## 安装与配置

### 依赖库安装

```bash
pip install -r requirements.txt
```

### 环境变量配置

复制 `.env.example` 文件为 `.env`，并填写您的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

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

当您需要为不同模型使用不同的 API 密钥和端点时（例如使用多家供应商的API），请设置相应配置。如果未设置特定模型的 API 配置，系统将使用默认的 `OPENAI_API_KEY` 和 `OPENAI_API_BASE`。

系统支持以下模型：
1. **主要模型 (LLM_MODEL)**：
   - OpenAI：`gpt-3.5-turbo`、`gpt-4`、`gpt-4-turbo`等
   - DeepSeek：`deepseek-ai/DeepSeek-V3`
2. **评估模型 (EVALUATION_MODEL)**：推荐使用 `gpt-4` 或其他强大模型以获得更准确的评估结果
3. **嵌入模型 (EMBEDDING_MODEL)**：
   - OpenAI：`text-embedding-3-large`、`text-embedding-3-small`
   - BAAI：`BAAI/bge-m3`
4. **推理链生成模型 (REASONING_MODEL)**：推荐使用 `deepseek-ai/DeepSeek-V3` 或 `gpt-4`以获得高质量的推理链

## 使用方法

### 使用多个数据集

本项目支持使用多个数据集进行评估，包括 LiveBench 等 Hugging Face 数据集：

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

从本地 JSON 文件加载数据集：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --local-json-dir data/processed_datasets
```

保存数据集到本地 JSON 文件：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --save-datasets --save-dir data/processed_datasets
```

同时使用多个参数组合：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math --max-samples-per-dataset 100 --strategies combined --result-prefix math_test --save-datasets --rebuild-db
```

仅记录对话日志（不评估）与 Hugging Face 数据集结合：

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

由于部分示例文件（例如位于 `examples/` 下的脚本）可能不存在，请使用以下命令运行主评估程序或启动 Web 界面以查看效果。

## 对话日志与批量评估

### 仅记录对话日志而不评估

使用 `--log-only` 参数可以只记录对话日志而不进行评估：

```bash
python src/main.py --log-only
```

您可以使用 `--session-id` 参数指定会话 ID 以便后续跟踪：

```bash
python src/main.py --log-only --session-id my-session-1
```

### 查看所有会话

使用 `batch_evaluation.py` 中的 `--list-sessions` 参数查看所有会话：

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

评估特定会话的对话日志（请将 <your_session_id> 替换为实际的会话ID）：

```bash
python src/batch_evaluation.py --session <your_session_id>
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
python src/batch_evaluation.py --report <your_session_id>
```

## SQLite备份功能

本项目新增了SQLite备份功能，可将评估结果和对话日志保存到SQLite数据库中，便于后续查询和分析。

### 使用SQLite备份

在运行评估时添加 `--sqlite-backup` 参数即可启用SQLite备份：

```bash
python src/main.py --sqlite-backup
```

默认情况下，SQLite数据库文件保存在 `data/backup.db` 中。您也可以通过 `--sqlite-db` 参数指定数据库文件路径：

```bash
python src/main.py --sqlite-backup --sqlite-db data/custom_backup.db
```

### 管理SQLite备份

项目提供了 `backup_manager.py` 工具用于管理SQLite备份：

#### 列出所有会话

```bash
python src/backup_manager.py list
```

#### 查看会话详情

```bash
python src/backup_manager.py detail <your_session_id>
```

#### 导出会话数据

将会话数据导出为JSON文件：

```bash
python src/backup_manager.py export <your_session_id> --output results/exported_session.json
```

### SQLite备份数据库结构

SQLite备份数据库包含以下表：
1. **evaluation_results**：评估结果表，保存每个问题的评估结果
2. **sessions**：会话元数据表，保存会话的基本信息
3. **strategy_metadata**：策略元数据表，保存策略的详细信息
4. **overall_metrics**：总体评估指标表，保存每个策略的总体评估指标

### 多数据集评估的SQLite备份

当使用多数据集评估时，所有数据集的评估结果和对话日志会保存在同一个SQLite数据库中，但会使用不同的会话ID，便于后续查询和分析：

```bash
python src/main.py --use-hf-dataset --hf-dataset livebench/math livebench/reasoning --separate-db --sqlite-backup
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
    python src/batch_evaluation.py --session <your_session_id>
    ```
4. 生成会话报告：
    ```bash
    python src/batch_evaluation.py --report <your_session_id>
    ```

### 使用SQLite备份的工作流示例

1. 启用SQLite备份运行评估：
    ```bash
    python src/main.py --strategies zero_shot few_shot --max-questions 10 --sqlite-backup
    ```
2. 查看所有会话：
    ```bash
    python src/backup_manager.py list
    ```
3. 查看特定会话详情：
    ```bash
    python src/backup_manager.py detail <your_session_id>
    ```
4. 导出特定会话数据：
    ```bash
    python src/backup_manager.py export <your_session_id> --output results/exported_session.json
    ```
5. 从Web服务器获取评估结果：
    ```bash
    python web/server.py --use-sqlite --db-path data/backup.db
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

## 查看和分析对话日志

对话日志存储在以下目录，按策略和数据集组织：
- 数学：`results/conversation_logs/math_evaluation/`
- 推理：`results/conversation_logs/reasoning_evaluation/`
- 数据分析：`results/conversation_logs/data_analysis_evaluation/`

您可以使用以下命令查看日志内容（*注意：Linux/macOS 下使用 `ls` 和 `cat` 命令；Windows 用户可使用 `dir` 和 `type` 命令*）：

```bash
# 列出数学评估日志文件（Linux/macOS: ls；Windows: dir）
ls results/conversation_logs/math_evaluation/combined/

# 查看特定日志文件（Linux/macOS: cat；Windows: type）
cat results/conversation_logs/math_evaluation/combined/math_question_id-timestamp.json
```

每个日志文件包含以下内容：
- 问题和参考答案
- 模型回答和推理过程
- 评估结果（准确率和推理质量）
- 元数据（策略详情、相似问题等）

> **备注**：在对话日志的 JSON 示例中，`similar_questions` 数组的格式为：
> ```json
> ["序号", "相似问题文本", "答案", 相似度评分]
> ```
> 其中第一个元素为内部序号或ID，后续分别为问题文本、答案和相似度数值。

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
  "question": "问题文本",                    // 问题内容
  "reference_answer": "参考答案",            // 标准答案
  "model_answer": "模型生成的答案",          // 模型回答
  "full_response": "完整的模型输出",         // 完整的模型响应
  "has_reasoning": true,                     // 是否包含推理过程
  "reasoning": "模型生成的推理过程",         // 推理过程
  "strategy": "combined",                    // 使用的策略名称
  "category": "arithmetic",                  // 问题类别
  "difficulty": "medium",                    // 问题难度
  "timestamp": 1649123456.789,               // 记录时间戳
  "session_id": "1649123456",                // 会话ID
  "evaluated": true,                         // 是否已评估
  "metadata": {                              // 元数据
    "strategy_details": {                    // 策略详情
      "name": "Auto-CoT + AutoReason",
      "description": "结合Auto-CoT和AutoReason的优势",
      "reasoning_model": "deepseek-ai/DeepSeek-V3",
      "num_examples": 2
    },
    "similar_questions": [                   
      ["0", "相似问题1", "答案1", 0.95],
      ["1", "相似问题2", "答案2", 0.85]
    ],
    "example_reasoning_chains": [            
      {
        "question_id": "0",
        "question": "相似问题1",
        "answer": "答案1",
        "similarity": 0.95,
        "reasoning_chain": "推理过程"
      }
    ]
  },
  "evaluation_result": {                     
    "accuracy": {
      "score": 1,
      "explanation": "评估解释"
    },
    "reasoning_quality": {
      "score": 9,
      "explanation": "评估解释"
    }
  },
  "evaluation_timestamp": 1649123556.789      
}
```

### 评估结果文件结构

评估结果保存在 `results/{result_prefix}_eval_results.json` 中，结构如下：

```json
{
  "combined": [
    {
      "id": "math_1",
      "question": "问题文本",
      "reference_answer": "参考答案",
      "model_answer": "模型生成的答案",
      "reasoning": "模型生成的推理过程",
      "category": "arithmetic",
      "difficulty": "medium",
      "metrics": {
        "accuracy": {
          "score": 1,
          "explanation": "评估解释"
        },
        "reasoning_quality": {
          "score": 9,
          "explanation": "评估解释"
        }
      },
      "timestamp": 1649123456.789
    }
  ],
  "zero_shot": [ /* ... 其他策略的评估结果 ... */ ],
  "timestamp": 1649123556.789,
  "overall_metrics": {
    "combined": {
      "total_questions": 50,
      "metrics": {
        "accuracy": {
          "average_score": 0.85,
          "count": 50
        },
        "reasoning_quality": {
          "average_score": 8.5,
          "count": 50
        }
      },
      "difficulty_breakdown": {
        "easy": { "count": 15, "accuracy": 0.96 },
        "medium": { "count": 20, "accuracy": 0.85 },
        "hard": { "count": 15, "accuracy": 0.72 }
      },
      "category_breakdown": {
        "arithmetic": { "count": 30, "accuracy": 0.92 },
        "algebra": { "count": 20, "accuracy": 0.75 }
      }
    }
  },
  "detailed_results": { /* ... 详细结果统计 ... */ }
}
```

## 多线程评估功能

本项目现支持多线程并行处理评估任务，可显著提高大规模问题的处理效率。

### 多线程功能特点

- 支持实时评估模式的多线程处理
- 支持批量评估模式的多线程处理
- 线程安全的评估结果记录和日志存储
- 可配置线程数量

### 使用方法

在命令行中，通过 `--threads` 参数指定线程数：

```bash
# 使用4个线程进行实时评估
python src/main.py --threads 4 --max-questions 10 --strategies zero_shot auto_cot

# 使用8个线程进行批量评估
python src/batch_evaluation.py --threads 8 --session <your_session_id>
```

### 示例脚本

提供了一个多线程评估示例脚本，位于 `examples/run_multithreaded.py`（如果该目录不存在，请忽略此部分或使用主评估程序）：

```bash
# 使用实时评估模式，4个线程
python examples/run_multithreaded.py --threads 4 --max-questions 10

# 使用批处理模式，8个线程
python examples/run_multithreaded.py --threads 8 --max-questions 20 --batch-mode
```

### 性能建议

- 对于需要大量API调用的场景，多线程能够显著提高并发处理能力
- 根据系统配置和API限制调整线程数
- 大型评估任务建议先收集对话日志，再采用批处理模式多线程评估

## 注意事项

1. 确保 OpenAI API 有足够的配额
2. 向量数据库可能需要较大存储空间
3. 评估过程可能耗费大量 API 调用，注意成本控制
4. 对于复杂问题，可设置较长超时时间
5. 对话日志存储在 `results/conversation_logs/` 目录下，按策略分类
6. 批量评估任务可能耗时较长，请适当设置批处理大小
7. 使用 `--separate-db` 参数时，会为每个数据集创建独立向量数据库，有助于提高相似问题检索效果
8. 使用 `--result-prefix` 参数可将不同评估任务的结果区分存储，便于后续比较分析

## 总结

本项目 README 文档提供了全面的评估流程说明，包括环境配置、各类评估策略、数据集处理、日志管理、SQLite备份以及多线程评估。请根据您的实际项目情况，适当调整配置与命令参数，以确保使用效果最佳。
