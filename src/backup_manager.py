"""
SQLite备份管理工具，用于管理CoT评估的SQLite备份
"""

import argparse
import logging
from typing import List, Dict, Any, Optional
import sys
import json
from datetime import datetime
from pathlib import Path

from sqlite_backup import SQLiteBackup

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_sessions(backup: SQLiteBackup) -> None:
    """
    列出所有会话
    
    参数:
        backup: SQLite备份实例
    """
    sessions = backup.get_sessions()
    
    if not sessions:
        print("没有找到任何会话记录")
        return
    
    print(f"\n找到 {len(sessions)} 个会话:\n")
    print(f"{'会话ID':<15} {'前缀':<20} {'数据集':<20} {'模型':<15} {'开始时间':<25} {'问题数':<8}")
    print("-" * 100)
    
    for session in sessions:
        # 格式化时间戳
        start_time = session.get('start_time', 0)
        time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S') if start_time else "未知"
        
        # 显示会话信息
        print(f"{session.get('session_id', 'N/A'):<15} "
              f"{session.get('result_prefix', 'N/A'):<20} "
              f"{session.get('dataset', 'N/A'):<20} "
              f"{session.get('model', 'N/A'):<15} "
              f"{time_str:<25} "
              f"{session.get('total_questions', 0):<8}")
    
    print("\n")

def session_detail(backup: SQLiteBackup, session_id: str) -> None:
    """
    显示会话详情
    
    参数:
        backup: SQLite备份实例
        session_id: 会话ID
    """
    # 获取会话结果
    results = backup.get_session_results(session_id)
    
    if not results:
        print(f"未找到会话 {session_id} 的记录")
        return
    
    # 计算统计信息
    strategy_stats = {}
    for strategy, items in results.items():
        if strategy not in ['timestamp', 'overall_metrics']:
            strategy_stats[strategy] = {
                'count': len(items),
                'accuracy': 0,
                'evaluated': 0
            }
            
            # 计算平均准确率
            for item in items:
                if 'metrics' in item and 'accuracy' in item['metrics']:
                    strategy_stats[strategy]['accuracy'] += item['metrics']['accuracy'].get('score', 0)
                    strategy_stats[strategy]['evaluated'] += 1
            
            # 计算平均值
            if strategy_stats[strategy]['evaluated'] > 0:
                strategy_stats[strategy]['accuracy'] /= strategy_stats[strategy]['evaluated']
    
    # 打印会话信息
    print(f"\n会话 {session_id} 详情:\n")
    print(f"{'策略':<15} {'问题数':<8} {'已评估':<8} {'平均准确率':<10}")
    print("-" * 50)
    
    for strategy, stats in strategy_stats.items():
        print(f"{strategy:<15} {stats['count']:<8} {stats['evaluated']:<8} {stats['accuracy']:.4f}")
    
    print("\n")

def export_session(backup: SQLiteBackup, session_id: str, output_file: Optional[str] = None) -> None:
    """
    导出会话数据
    
    参数:
        backup: SQLite备份实例
        session_id: 会话ID
        output_file: 输出文件路径
    """
    # 导出结果
    result_file = backup.export_to_json(session_id, output_file)
    
    if result_file:
        print(f"\n会话 {session_id} 已导出到 {result_file}\n")
    else:
        print(f"\n导出会话 {session_id} 失败\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SQLite备份管理工具")
    parser.add_argument("--db-path", type=str, default="data/backup.db", help="SQLite数据库路径")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 列出会话
    list_parser = subparsers.add_parser("list", help="列出所有会话")
    
    # 显示会话详情
    detail_parser = subparsers.add_parser("detail", help="显示会话详情")
    detail_parser.add_argument("session_id", type=str, help="会话ID")
    
    # 导出会话
    export_parser = subparsers.add_parser("export", help="导出会话数据")
    export_parser.add_argument("session_id", type=str, help="会话ID")
    export_parser.add_argument("--output", type=str, help="输出文件路径")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 初始化SQLite备份实例
    try:
        backup = SQLiteBackup(db_path=args.db_path)
    except Exception as e:
        logger.error(f"初始化SQLite备份失败: {e}")
        sys.exit(1)
    
    # 执行命令
    try:
        if args.command == "list":
            list_sessions(backup)
        elif args.command == "detail":
            session_detail(backup, args.session_id)
        elif args.command == "export":
            export_session(backup, args.session_id, args.output)
    except Exception as e:
        logger.error(f"执行命令 {args.command} 时出错: {e}")
        sys.exit(1)
    finally:
        backup.close()

if __name__ == "__main__":
    main() 