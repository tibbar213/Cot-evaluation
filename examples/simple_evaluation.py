"""
简单评估示例，展示如何使用评估框架
"""

import sys
import os
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models import generate_completion
from src.vector_db import VectorDatabase
from src.evaluation import Evaluator
from src.strategies import ZeroShotCoT, FewShotCoT, Baseline

def run_simple_evaluation():
    """运行简单评估示例"""
    print("===== LLM评估框架简单示例 =====")
    
    # 加载测试问题
    questions_file = project_root / "data" / "questions.json"
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"已加载 {len(questions)} 个问题")
    
    # 选择一小部分问题进行评估
    test_questions = questions[:3]
    
    # 初始化向量数据库
    print("初始化向量数据库...")
    vector_db = VectorDatabase()
    
    # 如果向量数据库为空，加载问题
    if len(vector_db.metadata) == 0:
        print("向量数据库为空，正在添加问题...")
        for q in questions:
            metadata = {k: v for k, v in q.items() if k != 'question'}
            vector_db.add_question(q['question'], metadata)
    
    print(f"向量数据库包含 {len(vector_db.metadata)} 个问题")
    
    # 初始化评估器
    evaluator = Evaluator()
    
    # 初始化策略
    strategies = {
        "baseline": Baseline(),
        "zero_shot": ZeroShotCoT(),
        "few_shot": FewShotCoT(vector_db=vector_db)
    }
    
    print(f"将使用 {len(strategies)} 个策略评估 {len(test_questions)} 个问题")
    
    # 运行评估
    for question in test_questions:
        question_id = question["id"]
        question_text = question["question"]
        reference_answer = question["answer"]
        category = question.get("category", "")
        difficulty = question.get("difficulty", "")
        
        print(f"\n问题: {question_text}")
        print(f"参考答案: {reference_answer}")
        
        for strategy_name, strategy in strategies.items():
            print(f"\n使用策略: {strategy_name}")
            
            # 生成提示
            prompt = strategy.generate_prompt(question_text)
            print(f"生成的提示:\n{prompt}")
            
            # 获取模型回答
            print("请求模型回答中...")
            response = generate_completion(prompt)
            print(f"模型回答:\n{response}")
            
            # 处理回答
            processed_response = strategy.process_response(response)
            print(f"处理后的答案: {processed_response['answer']}")
            
            # 评估回答
            eval_result = evaluator.evaluate_answer(
                question=question_text,
                reference_answer=reference_answer,
                model_response=processed_response,
                strategy_name=strategy_name,
                question_id=question_id,
                question_category=category,
                question_difficulty=difficulty
            )
            
            print(f"准确率: {eval_result['metrics']['accuracy']['score']}")
            print(f"评估说明: {eval_result['metrics']['accuracy']['explanation']}")
    
    # 保存结果
    result_file = evaluator.save_results("simple_eval_results.json")
    print(f"\n评估结果已保存到: {result_file}")
    
    # 打印摘要
    print("\n评估摘要:")
    evaluator.print_summary()

if __name__ == "__main__":
    # 确保存在 examples 目录
    os.makedirs(Path(__file__).parent, exist_ok=True)
    
    run_simple_evaluation()
