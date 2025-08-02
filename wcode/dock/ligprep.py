import os
import subprocess
from typing import Optional, Literal
from wcode.utils.config import convert_wsl_to_windows_path, TPATH


def ligprep(input_file: str,
            output_file: str,
            input_format: Literal['smi', 'csv', 'mae', 'sd'] = 'sd',
            output_format: Literal['sd', 'mae', 'smi', 'csv'] = 'sd',
            ionization: Literal[0, 1, 2] = 1,
            use_epik: bool = True,
            epik_metal_binding: bool = False,
            ph: float = 7.0,
            ph_tolerance: Optional[float] = None,
            generate_stereoisomers: bool = False,
            respect_geometry: bool = False,
            max_stereoisomers: int = 32,
            force_field: Literal[14, 16] = 14,
            njobs: Optional[int] = None,
            nstructs: Optional[int] = None,
            wait: bool = True,
            local: bool = False,
            host: Optional[str] = None) -> None:
    """
    使用LigPrep进行配体预处理
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        input_format: 输入文件格式 ('smi', 'csv', 'mae', 'sd')
        output_format: 输出文件格式 ('sd', 'mae', 'smi', 'csv')
        ionization: 电离处理 (0-不处理, 1-仅中和, 2-中和和电离)
        use_epik: 是否使用Epik进行电离和互变异构化
        epik_metal_binding: 是否启用Epik金属结合选项
        ph: 有效pH值
        ph_tolerance: pH容差
        generate_stereoisomers: 是否生成立体异构体
        respect_geometry: 是否尊重输入几何构型
        max_stereoisomers: 每个输入结构生成的最大立体异构体数量
        force_field: 力场类型 (14=OPLS_2005, 16=S-OPLS)
        njobs: 将作业分成NJOBS个子作业
        nstructs: 每个子作业的最大结构数
        wait: 是否等待作业完成
        local: 是否在当前目录保存文件
        host: 远程主机名
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 转换WSL路径到Windows路径（仅在Linux/WSL环境中）
    if TPATH.IS_LINUX:
        input_file = convert_wsl_to_windows_path(input_file)
        output_file = convert_wsl_to_windows_path(output_file)
    
    # 构建命令
    tpath_instance = TPATH()
    cmd_parts = [tpath_instance.LIGPREP]
    
    # 输入格式
    input_flag = f'-i{input_format}'
    cmd_parts.extend([input_flag, input_file])
    
    # 输出格式
    output_flag = f'-o{output_format}'
    cmd_parts.extend([output_flag, output_file])
    
    # 电离选项
    if use_epik:
        if epik_metal_binding:
            cmd_parts.append('-emb')
        else:
            cmd_parts.append('-epik')
    else:
        cmd_parts.extend(['-i', str(ionization)])
    
    # pH设置
    cmd_parts.extend(['-ph', str(ph)])
    if ph_tolerance is not None:
        cmd_parts.extend(['-pht', str(ph_tolerance)])
    
    # 立体异构体选项
    if generate_stereoisomers:
        cmd_parts.append('-ac')
    elif respect_geometry:
        cmd_parts.append('-g')
    
    cmd_parts.extend(['-s', str(max_stereoisomers)])
    
    # 力场选项
    cmd_parts.extend(['-bff', str(force_field)])
    
    # 作业控制选项
    if njobs is not None:
        cmd_parts.extend(['-NJOBS', str(njobs)])
    if nstructs is not None:
        cmd_parts.extend(['-NSTRUCTS', str(nstructs)])
    if host is not None:
        cmd_parts.extend(['-HOST', host])
    if wait:
        cmd_parts.append('-WAIT')
    if local:
        cmd_parts.append('-LOCAL')
    
    # 执行命令
    cmd = ' '.join(cmd_parts)
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"LigPrep执行失败:")
            print(f"错误输出: {result.stderr}")
            raise RuntimeError(f"LigPrep执行失败，返回码: {result.returncode}")
        else:
            print(f"LigPrep执行成功")
            if result.stdout:
                print(f"输出: {result.stdout}")
    except Exception as e:
        print(f"执行LigPrep时发生错误: {e}")
        raise


def ligprep_simple(input_file: str,
                  output_dir: str,
                  input_format: str = 'sd',
                  output_format: str = 'sd',
                  use_epik: bool = True) -> str:
    """
    简化版LigPrep函数，保持向后兼容
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        input_format: 输入格式
        output_format: 输出格式
        use_epik: 是否使用Epik
    
    Returns:
        输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = f"{input_basename}_ligprep.{output_format}"
    output_file = os.path.join(output_dir, output_filename)
    
    # 调用主函数
    ligprep(
        input_file=input_file,
        output_file=output_file,
        input_format=input_format,
        output_format=output_format,
        use_epik=use_epik
    )
    
    return output_file


if __name__ == '__main__':
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 获取当前脚本的目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建sample_data的绝对路径
        sample_data_path = os.path.join(current_dir, '..', '..', 'sample_data', 'imatinib2d.sdf')
        input_file = os.path.abspath(sample_data_path)
    
    # 使用简化版函数
    output_file = ligprep_simple(
        input_file=input_file,
        output_dir='./ligprep_output',
        input_format='sd',
        output_format='sd',
        use_epik=True
    )
    print(f"输出文件: {output_file}")
    
    # 使用完整版函数的示例
    # ligprep(
    #     input_file=input_file,
    #     output_file='./ligprep_output/complex_output.sd',
    #     input_format='sd',
    #     output_format='sd',
    #     use_epik=True,
    #     generate_stereoisomers=True,
    #     max_stereoisomers=16,
    #     ph=7.4
    # )