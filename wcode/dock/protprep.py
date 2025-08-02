import os
import subprocess
from typing import Optional, Literal
from wcode.utils.config import convert_wsl_to_windows_path, TPATH


def protprep(input_file: str,
             output_file: str,
             input_format: Literal['pdb', 'mae'] = 'pdb',
             output_format: Literal['pdb', 'mae'] = 'mae',
             # 预处理选项
             no_preprocess: bool = False,
             reference_st_file: Optional[str] = None,
             reference_pdbid: Optional[str] = None,
             no_bond_orders: bool = False,
             no_ccd: bool = False,
             assign_all_residues: bool = True,
             no_htreat: bool = False,
             rehtreat: bool = True,
             no_metal_treat: bool = False,
             disulfides: bool = True,
             glycosylation: bool = False,
             palmitoylation: bool = False,
             antibody_cdr_scheme: Optional[Literal['Chothia', 'Kabat', 'IMGT', 'EnhancedChothia', 'AHo', 'None']] = 'Kabat',
             renumber_ab_residues: bool = False,
             tcr_cdr_scheme: Optional[Literal['IMGT', 'AHo', 'None']] = None,
             renumber_tcr_residues: bool = False,
             mse: bool = False,
             preprocess_watdist: Optional[float] = None,
             fill_loops: bool = False,
             fasta_file: Optional[str] = None,
             fill_sidechains: bool = True,
             add_oxt: bool = False,
             cap_termini: bool = False,
             cap_termini_min_atoms: Optional[int] = None,
             # Epik选项
             no_epik: bool = False,
             epik_ph: float = 7.4,
             epik_pht: float = 2.0,
             max_states: int = 4,
             no_idealize_htf: bool = False,
             # ProtAssign选项
             no_protassign: bool = False,
             sample_water: bool = True,
             include_epik_states: bool = False,
             xtal: bool = False,
             no_propka: bool = False,
             propka_ph: float = 7.4,
             label_pkas: bool = False,
             ph: Optional[Literal['very_low', 'low', 'neutral', 'high']] = None,
             force: Optional[str] = None,
             minimize_adj_h: bool = False,
             # Impref选项
             no_impref: bool = False,
             rmsd: float = 0.3,
             fix: bool = True,
             force_field: Literal['2005', '2.1', '3', 'S-OPLS', 'OPLS_2005'] = 'S-OPLS',
             keep_far_wat: bool = False,
             watdist: float = 5.0,
             delwater_hbond_cutoff: Optional[int] = None,
             # 其他选项
             preserve_st_titles: bool = False,
             use_pdb_ph: bool = False,
             # 作业控制
             host: Optional[str] = None,
             wait: bool = True,
             save: bool = False,
             no_jobid: bool = False,
             jobname: Optional[str] = None) -> None:
    """
    使用Protein Preparation Wizard进行蛋白质预处理
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        input_format: 输入文件格式 ('pdb', 'mae')
        output_format: 输出文件格式 ('pdb', 'mae')
        no_preprocess: 跳过预处理
        reference_st_file: 参考结构文件
        reference_pdbid: 参考PDB ID
        no_bond_orders: 不分配键序
        no_ccd: 不使用化学组分字典
        assign_all_residues: 为所有残基分配键序
        no_htreat: 不添加氢原子
        rehtreat: 删除并重新添加氢原子
        no_metal_treat: 不处理金属
        disulfides: 创建二硫键
        glycosylation: 创建糖基化键
        palmitoylation: 创建棕榈酰化键
        antibody_cdr_scheme: 抗体CDR注释方案
        renumber_ab_residues: 重新编号抗体残基
        tcr_cdr_scheme: TCR CDR注释方案
        renumber_tcr_residues: 重新编号TCR残基
        mse: 将硒代蛋氨酸转换为蛋氨酸
        preprocess_watdist: 预处理时删除水分子的距离阈值
        fill_loops: 用Prime填充缺失的环
        fasta_file: 自定义FASTA文件
        fill_sidechains: 用Prime填充缺失的侧链
        add_oxt: 添加末端氧原子
        cap_termini: 封端
        cap_termini_min_atoms: 封端的最小原子数
        no_epik: 不对het基团运行Epik
        epik_ph: Epik目标pH
        epik_pht: Epik pH范围
        max_states: 每个蛋白质复合物的最大状态数
        no_idealize_htf: 不理想化添加的氢原子温度因子
        no_protassign: 不运行ProtAssign
        sample_water: 采样水状态
        include_epik_states: 在优化中包含Epik嵌入状态
        xtal: 使用晶体对称性
        no_propka: 不使用PROPKA
        propka_ph: PROPKA pH值
        label_pkas: 用PROPKA pKa标记残基
        ph: 采样pH
        force: 强制特定残基状态
        minimize_adj_h: 能量最小化所有可调节氢原子
        no_impref: 不运行约束最小化作业
        rmsd: 最小化RMSD截止值
        fix: 固定重原子
        force_field: 力场版本
        keep_far_wat: 不删除远离het基团的水分子
        watdist: 远距离水分子的距离阈值
        delwater_hbond_cutoff: 删除水分子的氢键截止值
        preserve_st_titles: 保留结构标题
        use_pdb_ph: 使用PDB pH
        host: 远程主机名
        wait: 等待作业完成
        save: 作业完成时返回zip归档
        no_jobid: 直接运行作业
        jobname: 作业名称
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 转换WSL路径到Windows路径（仅在Linux/WSL环境中）
    if TPATH.IS_LINUX:
        input_file = convert_wsl_to_windows_path(input_file)
        output_file = convert_wsl_to_windows_path(output_file)
        if reference_st_file:
            reference_st_file = convert_wsl_to_windows_path(reference_st_file)
        if fasta_file:
            fasta_file = convert_wsl_to_windows_path(fasta_file)
    
    # 构建命令
    tpath_instance = TPATH()
    cmd_parts = [tpath_instance.PROTEINPREP, input_file, output_file]
    
    # 预处理选项
    if no_preprocess:
        cmd_parts.append('-nopreprocess')
    if reference_st_file:
        cmd_parts.extend(['-reference_st_file', reference_st_file])
    if reference_pdbid:
        cmd_parts.extend(['-reference_pdbid', reference_pdbid])
    if no_bond_orders:
        cmd_parts.append('-nobondorders')
    if no_ccd:
        cmd_parts.append('-noccd')
    if assign_all_residues:
        cmd_parts.append('-assign_all_residues')
    if no_htreat:
        cmd_parts.append('-nohtreat')
    if rehtreat:
        cmd_parts.append('-rehtreat')
    if no_metal_treat:
        cmd_parts.append('-nometaltreat')
    if disulfides:
        cmd_parts.append('-disulfides')
    if glycosylation:
        cmd_parts.append('-glycosylation')
    if palmitoylation:
        cmd_parts.append('-palmitoylation')
    if antibody_cdr_scheme:
        cmd_parts.extend(['-antibody_cdr_scheme', antibody_cdr_scheme])
    if renumber_ab_residues:
        cmd_parts.append('-renumber_ab_residues')
    if tcr_cdr_scheme:
        cmd_parts.extend(['-tcr_cdr_scheme', tcr_cdr_scheme])
    if renumber_tcr_residues:
        cmd_parts.append('-renumber_tcr_residues')
    if mse:
        cmd_parts.append('-mse')
    if preprocess_watdist is not None:
        cmd_parts.extend(['-preprocess_watdist', str(preprocess_watdist)])
    if fill_loops:
        cmd_parts.append('-fillloops')
    if fasta_file:
        cmd_parts.extend(['-fasta_file', fasta_file])
    if fill_sidechains:
        cmd_parts.append('-fillsidechains')
    if add_oxt:
        cmd_parts.append('-addOXT')
    if cap_termini:
        cmd_parts.append('-captermini')
    if cap_termini_min_atoms is not None:
        cmd_parts.extend(['-cap_termini_min_atoms', str(cap_termini_min_atoms)])
    
    # Epik选项
    if no_epik:
        cmd_parts.append('-noepik')
    cmd_parts.extend(['-epik_pH', str(epik_ph)])
    cmd_parts.extend(['-epik_pHt', str(epik_pht)])
    cmd_parts.extend(['-max_states', str(max_states)])
    if no_idealize_htf:
        cmd_parts.append('-noidealizehtf')
    
    # ProtAssign选项
    if no_protassign:
        cmd_parts.append('-noprotassign')
    if sample_water:
        cmd_parts.append('-samplewater')
    if include_epik_states:
        cmd_parts.append('-include_epik_states')
    if xtal:
        cmd_parts.append('-xtal')
    if no_propka:
        cmd_parts.append('-nopropka')
    cmd_parts.extend(['-propka_pH', str(propka_ph)])
    if label_pkas:
        cmd_parts.append('-label_pkas')
    if ph:
        cmd_parts.extend(['-pH', ph])
    if force:
        cmd_parts.extend(['-force', force])
    if minimize_adj_h:
        cmd_parts.append('-minimize_adj_h')
    
    # Impref选项
    if no_impref:
        cmd_parts.append('-noimpref')
    cmd_parts.extend(['-rmsd', str(rmsd)])
    if fix:
        cmd_parts.append('-fix')
    cmd_parts.extend(['-f', force_field])
    if keep_far_wat:
        cmd_parts.append('-keepfarwat')
    cmd_parts.extend(['-watdist', str(watdist)])
    if delwater_hbond_cutoff is not None:
        cmd_parts.extend(['-delwater_hbond_cutoff', str(delwater_hbond_cutoff)])
    
    # 其他选项
    if preserve_st_titles:
        cmd_parts.append('-preserve_st_titles')
    if use_pdb_ph:
        cmd_parts.append('-use_PDB_pH')
    
    # 作业控制选项
    if host:
        cmd_parts.extend(['-HOST', host])
    if wait:
        cmd_parts.append('-WAIT')
    if save:
        cmd_parts.append('-SAVE')
    if no_jobid:
        cmd_parts.append('-NOJOBID')
    if jobname:
        cmd_parts.extend(['-JOBNAME', jobname])
    
    # 执行命令
    cmd = ' '.join(cmd_parts)
    print(f"执行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Protein Preparation Wizard执行失败:")
            print(f"错误输出: {result.stderr}")
            raise RuntimeError(f"Protein Preparation Wizard执行失败，返回码: {result.returncode}")
        else:
            print(f"Protein Preparation Wizard执行成功")
            if result.stdout:
                print(f"输出: {result.stdout}")
    except Exception as e:
        print(f"执行Protein Preparation Wizard时发生错误: {e}")
        raise


def protprep_simple(input_file: str,
                   output_dir: str,
                   input_format: str = 'pdb',
                   output_format: str = 'mae',
                   fill_sidechains: bool = True,
                   disulfides: bool = True,
                   assign_all_residues: bool = True,
                   rehtreat: bool = True,
                   max_states: int = 1,
                   epik_ph: float = 7.4,
                   epik_pht: float = 2.0,
                   antibody_cdr_scheme: str = 'Kabat',
                   sample_water: bool = True,
                   propka_ph: float = 7.4,
                   fix: bool = True,
                   force_field: str = 'S-OPLS',
                   rmsd: float = 0.3,
                   watdist: float = 5.0,
                   host: str = 'localhost:16') -> str:
    """
    简化版Protein Preparation Wizard函数，保持向后兼容
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
        input_format: 输入格式
        output_format: 输出格式
        fill_sidechains: 填充侧链
        disulfides: 处理二硫键
        assign_all_residues: 分配所有残基
        rehtreat: 重新处理氢原子
        max_states: 最大状态数
        epik_ph: Epik pH
        epik_pht: Epik pH容差
        antibody_cdr_scheme: 抗体CDR方案
        sample_water: 采样水
        propka_ph: PROPKA pH
        fix: 固定重原子
        force_field: 力场
        rmsd: RMSD截止值
        watdist: 水距离
        host: 主机
    
    Returns:
        输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = f"{input_basename}_protprep.{output_format}"
    output_file = os.path.join(output_dir, output_filename)
    
    # 调用主函数
    protprep(
        input_file=input_file,
        output_file=output_file,
        input_format=input_format,
        output_format=output_format,
        fill_sidechains=fill_sidechains,
        disulfides=disulfides,
        assign_all_residues=assign_all_residues,
        rehtreat=rehtreat,
        max_states=max_states,
        epik_ph=epik_ph,
        epik_pht=epik_pht,
        antibody_cdr_scheme=antibody_cdr_scheme,
        sample_water=sample_water,
        propka_ph=propka_ph,
        fix=fix,
        force_field=force_field,
        rmsd=rmsd,
        watdist=watdist,
        host=host
    )
    
    return output_file


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 获取当前脚本的目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建sample_data的绝对路径
        sample_data_path = os.path.join(current_dir, '..', '..', 'sample_data', '1a0q_notprepare.pdb')
        input_file = os.path.abspath(sample_data_path)
    
    # 使用简化版函数
    output_file = protprep_simple(
        input_file=input_file,
        output_dir=os.path.join(current_dir, '..', '..', 'sample_data'),
        host='localhost:16'
    )
    print(f"输出文件: {output_file}")
    
    # 使用完整版函数的示例
    # protprep(
    #     input_file=input_file,
    #     output_file='./protprep_output/complex_output.mae',
    #     input_format='pdb',
    #     output_format='mae',
    #     fill_sidechains=True,
    #     disulfides=True,
    #     assign_all_residues=True,
    #     rehtreat=True,
    #     max_states=4,
    #     epik_ph=7.4,
    #     epik_pht=2.0,
    #     antibody_cdr_scheme='Kabat',
    #     sample_water=True,
    #     propka_ph=7.4,
    #     fix=True,
    #     force_field='S-OPLS',
    #     rmsd=0.3,
    #     watdist=5.0,
    #     host='localhost:16'
    # )