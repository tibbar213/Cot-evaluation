"""
模型接口，用于与OpenAI API交互
"""

import time
import logging
import re
import json
from typing import Dict, List, Any, Optional, Union
import openai
from openai import OpenAI
from config import (
    OPENAI_API_KEY, 
    OPENAI_API_BASE, 
    LLM_MODEL, 
    EVALUATION_MODEL, 
    EMBEDDING_MODEL,
    REASONING_MODEL,
    # 添加各模型专用API配置
    LLM_API_KEY,
    LLM_API_BASE,
    EVALUATION_API_KEY,
    EVALUATION_API_BASE,
    EMBEDDING_API_KEY,
    EMBEDDING_API_BASE,
    REASONING_API_KEY,
    REASONING_API_BASE
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 为不同模型创建不同的OpenAI客户端
default_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
evaluation_client = OpenAI(api_key=EVALUATION_API_KEY, base_url=EVALUATION_API_BASE)
embedding_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE)
reasoning_client = OpenAI(api_key=REASONING_API_KEY, base_url=REASONING_API_BASE)

logger.info(f"已初始化OpenAI客户端")
logger.info(f"使用的模型 - LLM: {LLM_MODEL}, 评估: {EVALUATION_MODEL}, 嵌入: {EMBEDDING_MODEL}, 推理: {REASONING_MODEL}")

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    获取文本的向量嵌入
    
    Args:
        text (str): 输入文本
        model (str): 使用的嵌入模型
        
    Returns:
        List[float]: 嵌入向量
    """
    try:
        # 对输入文本进行预处理
        text = text.replace("\n", " ")
        
        logger.info(f"正在获取文本的向量嵌入，使用模型: {model}")
        
        # 使用嵌入模型专用客户端
        # 调用OpenAI API获取嵌入
        response = embedding_client.embeddings.create(
            model=model,
            input=text
        )
        
        # 返回嵌入向量
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"获取嵌入时出错: {e}")
        # 记录更详细的错误信息
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        raise

def generate_completion(
    prompt: str, 
    model: str = LLM_MODEL, 
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retry_count: int = 3,
    retry_delay: int = 5
) -> str:
    """
    生成文本补全
    
    Args:
        prompt (str): 输入提示
        model (str): 使用的模型
        temperature (float): 温度参数，控制随机性
        max_tokens (int): 生成的最大令牌数
        retry_count (int): 重试次数
        retry_delay (int): 重试延迟（秒）
        
    Returns:
        str: 生成的文本
    """
    for attempt in range(retry_count):
        try:
            start_time = time.time()
            
            logger.info(f"正在生成文本补全，使用模型: {model}")
            
            # 根据模型类型选择相应的客户端
            if model == LLM_MODEL:
                client_to_use = llm_client
            elif model == EVALUATION_MODEL:
                client_to_use = evaluation_client
            elif model == REASONING_MODEL:
                client_to_use = reasoning_client
            else:
                client_to_use = default_client
            
            # 调用OpenAI API生成补全
            response = client_to_use.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            logger.info(f"生成补全耗时: {elapsed_time:.2f}秒")
            
            # 返回生成的文本
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"生成补全时出错 (尝试 {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"生成补全失败，已达到最大重试次数: {e}")
                # 记录更详细的错误信息
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
                raise

def clean_json_string(text: str) -> str:
    """
    清理JSON字符串，移除Markdown格式和其他可能导致解析错误的内容
    
    Args:
        text (str): 原始文本
        
    Returns:
        str: 清理后的JSON字符串
    """
    # 移除可能的Markdown JSON代码块
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    json_match = re.search(json_pattern, text)
    if json_match:
        return json_match.group(1).strip()
    
    # 如果没有匹配到Markdown格式，返回原始文本并移除前后的空白
    return text.strip()

def evaluate_response(
    question: str,
    reference_answer: str,
    model_response: str,
    metric: str = "accuracy",
    model: str = EVALUATION_MODEL
) -> Dict[str, Any]:
    """
    评估模型回答
    
    Args:
        question (str): 问题
        reference_answer (str): 参考答案
        model_response (str): 模型回答
        metric (str): 评估指标
        model (str): 使用的评估模型
        
    Returns:
        Dict[str, Any]: 评估结果
    """
    try:
        # 构建评估提示
        if metric == "accuracy":
            prompt = f"""
            请严格评估以下回答的准确性：
            
            问题: {question}
            参考答案: {reference_answer}
            模型回答: {model_response}
            
            请给出严格的二元评分：
            - 如果回答完全正确，给1分
            - 如果回答有任何错误或不完整，给0分
            
            不要给0到1之间的分数，必须是0或1。
            仅返回JSON格式：{{"score": 评分, "explanation": "解释"}}
            """
        elif metric == "reasoning_quality":
            prompt = f"""
            请评估以下回答的推理质量：
            
            问题: {question}
            模型回答: {model_response}
            
            考虑推理的清晰度、逻辑性和步骤的合理性。
            请给出评分（1-10之间的整数，其中1表示推理质量很差，10表示推理质量极佳）并简要解释原因。
            仅返回JSON格式：{{"score": 评分, "explanation": "解释"}}
            """
        else:
            raise ValueError(f"不支持的评估指标: {metric}")
        
        # 获取评估结果，使用评估模型专用客户端
        response = generate_completion(prompt, model=model, temperature=0.3)
        
        # 清理并解析评估结果
        try:
            cleaned_response = clean_json_string(response)
            logger.info(f"清理后的JSON: {cleaned_response}")
            result = json.loads(cleaned_response)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"无法解析评估结果JSON: {response}")
            logger.error(f"JSON解析错误: {e}")
            # 尝试一个更简单的解析方法，提取score
            try:
                score_match = re.search(r'"score"\s*:\s*([0-9\.]+)', response)
                if score_match:
                    score = float(score_match.group(1))
                    explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', response)
                    explanation = explanation_match.group(1) if explanation_match else "无法提取解释"
                    return {"score": score, "explanation": explanation}
                else:
                    logger.error("无法提取评分，返回默认评分0")
                    return {"score": 0, "explanation": "无法解析JSON评估结果"}
            except Exception as ex:
                logger.error(f"尝试提取评分时出错: {ex}")
                return {"score": 0, "explanation": "无法解析JSON评估结果"}
    except Exception as e:
        logger.error(f"评估回答时出错: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return {"score": 0, "explanation": f"评估过程出错: {e}"}

def generate_reasoning_chain(
    question: str,
    model: str = REASONING_MODEL,
    prompt_template: str = "您将获得一个问题，并使用该问题将其分解为一系列逻辑推理轨迹。仅写下推理过程，不要自己回答问题。\n\n问题: {question}"
) -> str:
    """
    为问题生成推理链
    
    Args:
        question (str): 问题
        model (str): 使用的模型
        prompt_template (str): 提示模板
        
    Returns:
        str: 生成的推理链
    """
    logger.info(f"使用模型 {model} 生成推理链")
    prompt = prompt_template.format(question=question)
    return generate_completion(prompt, model=model, temperature=0.3)
