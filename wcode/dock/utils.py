"""
Docking utilities for working with Schrodinger tools.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Optional

from wcode.utils.config import TPATH

# 设置日志
logger = logging.getLogger(__name__)


def split_mae_file(mae_file: str, output_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    使用Schrodinger的pv_convert.py脚本将MAE文件拆分为受体和配体部分。
    
    Args:
        mae_file: 输入的MAE文件路径
        output_dir: 输出目录，如果为None则使用输入文件所在目录
        
    Returns:
        Tuple[str, str]: (receptor_file, ligand_file) 受体和配体文件的路径
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(mae_file):
        raise FileNotFoundError(f"输入文件不存在: {mae_file}")
    
    # 获取Schrodinger路径
    schrodinger_path = TPATH.SCHRODINGER_PATH
    if not schrodinger_path:
        raise ValueError("Schrodinger路径未在config.py中配置")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(mae_file)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建pv_convert.py脚本路径
    pv_convert_script = os.path.join(
        schrodinger_path, 
        "mmshare-v7.0", 
        "python", 
        "common", 
        "pv_convert.py"
    )
    
    # 检查脚本是否存在
    if not os.path.exists(pv_convert_script):
        raise FileNotFoundError(f"pv_convert.py脚本不存在: {pv_convert_script}")
    
    # 构建run命令路径
    run_command = os.path.join(schrodinger_path, "run")
    if not os.path.exists(run_command):
        raise FileNotFoundError(f"Schrodinger run命令不存在: {run_command}")
    
    # 获取输入文件的基础名称（不含扩展名）
    input_basename = os.path.splitext(os.path.basename(mae_file))[0]
    
    # 构建输出文件路径
    receptor_file = os.path.join(output_dir, f"{input_basename}-out_recep.mae")
    ligand_file = os.path.join(output_dir, f"{input_basename}-out_lig.mae")
    
    try:
        # 提取受体部分
        logger.info(f"正在提取受体部分...")
        cmd_receptor = [
            run_command, 
            "python3", 
            pv_convert_script, 
            "-mode", "split_receptor", 
            mae_file
        ]
        
        result_receptor = subprocess.run(
            cmd_receptor,
            capture_output=True,
            text=True,
            cwd=output_dir
        )
        
        if result_receptor.returncode != 0:
            raise RuntimeError(f"提取受体失败: {result_receptor.stderr}")
        
        # 提取配体部分
        logger.info(f"正在提取配体部分...")
        cmd_ligand = [
            run_command, 
            "python3", 
            pv_convert_script, 
            "-mode", "split_ligand", 
            mae_file
        ]
        
        result_ligand = subprocess.run(
            cmd_ligand,
            capture_output=True,
            text=True,
            cwd=output_dir
        )
        
        if result_ligand.returncode != 0:
            raise RuntimeError(f"提取配体失败: {result_ligand.stderr}")
        
        # 检查输出文件是否生成
        if not os.path.exists(receptor_file):
            raise RuntimeError(f"受体文件未生成: {receptor_file}")
        
        if not os.path.exists(ligand_file):
            raise RuntimeError(f"配体文件未生成: {ligand_file}")
        
        logger.info(f"成功拆分MAE文件:")
        logger.info(f"  受体文件: {receptor_file}")
        logger.info(f"  配体文件: {ligand_file}")
        
        return receptor_file, ligand_file
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"执行pv_convert.py脚本失败: {e}")
    except Exception as e:
        raise RuntimeError(f"拆分MAE文件时发生错误: {e}")


if __name__ == "__main__":
    # 测试函数
    import sys
    
    if len(sys.argv) > 1:
        mae_file = sys.argv[1]
        try:
            receptor_file, ligand_file = split_mae_file(mae_file)
            print(f"成功拆分文件:")
            print(f"  受体: {receptor_file}")
            print(f"  配体: {ligand_file}")
        except Exception as e:
            print(f"错误: {e}")
    else:
        print("用法: python utils.py <mae_file>") 