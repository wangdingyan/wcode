#!/usr/bin/env python


import os
from prody import *
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from io import StringIO
import pypdb
import gzip
import requests
import pandas as pd
import numpy as np

########################################################################################################################


##update 20231118-20
def get_pdb_components_by_pdbfile(pdbfile):
    """
    Split a protein-ligand pdb into protein and ligand components
    :param pdb_id: "8ahx.pdb"
    :return:
    """
    with open(pdbfile) as f:
        try:
            pdb = parsePDBStream(f)
            protein = pdb.select('protein')
            ligand = pdb.select('not protein and not water')
            return protein, ligand
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError occurred. Skipping file {pdbfile}.")

# get_pdb_components_by_pdbfile("8ahx.pdb")
# # @> 12687 atoms and 1 coordinate set(s) were parsed in 0.15s.
# # # (<Selection: 'protein' from Unknown (12008 atoms)>,
# # #  <Selection: 'not protein and not water' from Unknown (679 atoms)>)

##update 20231118-20
def get_pdb_components_by_file_gz(pdb_file_gz):
    """
    Split a protein-ligand pdb into protein and ligand components
    :param pdb_file_gz: "pdb8ahx.ent.gz" or "8ahx.pdb.gz"
    :return:
    """
    with gzip.open(pdb_file_gz, 'rt') as f:
        try:
            pdb = parsePDBStream(f)
            protein = pdb.select('protein')
            ligand = pdb.select('not protein and not water')
            return protein, ligand
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError occurred. Skipping file {pdb_file_gz}.")

##update from parse_complex_ligand.py
def read_ligand_expo():
    """
    # http://ligand-expo.rcsb.org/
    Read Ligand Expo data, try to find a file called
    Components-smiles-stereo-oe.smi in the current directory.
    If you can't find the file, grab it from the RCSB
    :return: Ligand Expo as a dictionary with ligand id as the key
    """
    file_name = "Components-smiles-stereo-oe.smi"
    try:
        df = pd.read_csv(file_name, sep=r"[\t]+",
                         header=None,
                         names=["SMILES", "ID", "Name"],
                         engine='python')
    except FileNotFoundError:
        url = f"http://ligand-expo.rcsb.org/dictionaries/{file_name}"
        print(url)
        r = requests.get(url, allow_redirects=True)
        open('Components-smiles-stereo-oe.smi', 'wb').write(r.content)
        df = pd.read_csv(file_name, sep="\t",
                         header=None,
                         names=["SMILES", "ID", "Name"],
                         na_filter=False)
    return df


def write_pdb(protein, pdb_name):
    """
    Write a prody protein to a pdb file
    :param protein: protein object from prody
    :param pdb_name: base name for the pdb file
    :return: None
    """
    output_pdb_name = f"{pdb_name}_protein.pdb"
    writePDB(f"{output_pdb_name}", protein)
    print(f"wrote {output_pdb_name}")

def write_sdf(new_mol, pdb_name, res_name):
    """
    Write an RDKit molecule to an SD file
    :param new_mol:
    :param pdb_name:
    :param res_name:
    :return:
    """
    outfile_name = f"{pdb_name}_{res_name}_ligand.sdf"
    writer = Chem.SDWriter(f"{outfile_name}")
    if new_mol is not None:
        writer.write(new_mol)
        print(f"wrote {outfile_name}")

def write_mol_to_sdf(input, output_file):
    """
    将分子（SMILES或RDKit的Mol对象）写入SDF文件
    参数:
    input (str or RDKit Mol): 输入的分子，可以是SMILES字符串或RDKit的Mol对象
    output_file (str): 输出SDF文件的路径
    返回:
    None
    """
    # 判断输入类型
    if isinstance(input, str):
        # 如果是字符串，尝试将其解析为RDKit的Mol对象
        mol = Chem.MolFromSmiles(input)
        if mol is None:
            print("输入的SMILES格式不符合规范，跳过保存为SDF。")
            return
    elif isinstance(input, Chem.Mol):
        mol = input
    else:
        print("输入的格式不符合规范，跳过保存为SDF。")
        return
    # 保存Mol对象为SDF文件
    try:
        writer = Chem.SDWriter(output_file)
        writer.write(mol)
        writer.close()
        print(f"分子已保存为SDF文件: {output_file}")
    except Exception as e:
        print(f"保存SDF文件时发生错误: {e}")

"""直接读取sdf保存为mol列表"""
def read_sdf_to_mol(input_file):
    """
    从SDF文件中读取并返回Mol对象列表
    参数:
    input_file (str): 输入SDF文件的路径
    返回:
    list: 包含Mol对象的列表
    """
    mol_list = []
    try:
        supplier = Chem.SDMolSupplier(input_file)
        for mol in supplier:
            if mol is not None:
                mol_list.append(mol)
        print(f"从SDF文件中读取了 {len(mol_list)} 个分子")
    except Exception as e:
        print(f"读取SDF文件时发生错误: {e}")
    return mol_list

##20231114update
def write_smiles_from_sdf_to_sdf(input_sdf, pdb_name, res_name):
    outfile_name = f"{pdb_name}_{res_name}_ligand_noxyz.sdf"
    writer = Chem.SDWriter(f"{outfile_name}")
    mol_suppier = Chem.SDMolSupplier(input_sdf)
    smiles_list = [Chem.MolToSmiles(mol) for mol in mol_suppier if mol is not None]
    new_mol = Chem.MolFromSmiles(smiles_list[0])
    if new_mol is not None:
        writer.write(new_mol)
        print(f"wrote {outfile_name}")

"""update 20231117：读取pdb文件，检查res和protein之间是否存在residue connectivity """
def find_bonds_from_pdb_file(pdb_name,res_name):
    # parsePDB(pdb_name)
    # pdb file
    file_name = f"{pdb_name}.pdb.gz"
    res = res_name
    hetatm_res_list = []
    conect_list = []
    atom_list = []
    if os.path.exists(file_name):
        with gzip.open(file_name, 'rt') as file:
            for line in file:
                ##get res atom id
                if line.startswith('HETATM') and res in line:
                    hetatm_id = line.strip()[6:6+5].strip()
                    hetatm_res_list.append(hetatm_id)
                    # elements = line.split()  # 默认使用空格分隔
                    # hetatm_res_list.append(elements)
                ##get res atom bonds conect
                if line.startswith('CONECT'):
                    res_id_num = (len(line)-6)//5
                    res_id_ls = res_id_num *[0]
                    for i in range(res_id_num):
                        res_id_ls[i] = line.strip()[6+5*i:6+5*(i+1)].strip()
                    conect_list.append(res_id_ls)
                if line.startswith("ATOM"):
                    atom_id = line.strip()[6:6+5].strip()
                    atom_list.append(atom_id)
        ls_res_set = set(hetatm_res_list)
        # print(ls_res_set)
        ls_con = []
        for item in conect_list:
            # print(item)
            if item[0] in ls_res_set:
                # print(item)
                ls_con.append(item[1:len(item)])
        ls_con_set = {element for sublist in ls_con for element in sublist}
        # print(ls_con_set)
        ls_atom_set = set(atom_list)
        if ls_res_set.issubset(ls_con_set) and ls_con_set.isdisjoint(ls_atom_set):
            conect_bond = False
            # print("the res is not bond to protein!")
        else:
            conect_bond = True
            # print("the res {} is bond to protein!".format(res_name))

        return conect_bond


"""过滤掉经常出现的75个complex+11个（SO4+FUC+MG+CA+NH2+NA+GOL+DGD+EDO+FMT+UND）共86个complex, uodate 20231112-17"""
# DGD C51H96O15=948
# https://huggingface.co/datasets/jglaser/pdb_protein_ligand_complexes/blob/main/parse_complexes.py
# Split a protein-ligand complex into protein and ligands and assign ligand bond orders using SMILES strings from Ligand Export
# Code requires Python 3.6
# filter out these common additives which occur in more than 75 complexes in the PDB
ubiquitous_ligands = ['SO4','FUC','MG','CA','NH2','NA','GOL','DGD','EDO','FMT','UND',
                      'PEG', 'ADP', 'FAD', 'NAD', 'ATP', 'MPD', 'NAP', 'GDP', 'MES',
                      'GTP', 'FMN', 'HEC', 'TRS', 'CIT', 'PGE', 'ANP', 'SAH', 'NDP',
                      'PG4', 'EPE', 'AMP', 'COA', 'MLI', 'FES', 'GNP', 'MRD', 'GSH',
                      'FLC', 'AGS', 'NAI', 'SAM', 'PCW', '1PE', 'TLA', 'BOG', 'CYC',
                      'UDP', 'PX4', 'NAG', 'IMP', 'POP', 'UMP', 'PLM', 'HEZ', 'TPP',
                      'ACP', 'LDA', 'ACO', 'CLR', 'BGC', 'P6G', 'LMT', 'OGA', 'DTT',
                      'POV', 'FBP', 'AKG', 'MLA', 'ADN', 'NHE', '7Q9', 'CMP', 'BTB',
                      'PLP', 'CAC', 'SIN', 'C2E', '2AN', 'OCT', '17F', 'TAR', 'BTN',
                      'XYP', 'MAN', '5GP', 'GAL', 'GLC', 'DTP', 'DGT', 'PEB', 'THP',
                      'BEZ', 'CTP', 'GSP', 'HED', 'ADE', 'TYD', 'TTP', 'BNG', 'IHP',
                      'FDA', 'PEP', 'ALF', 'APR', 'MTX', 'MLT', 'LU8', 'UTP', 'APC',
                      'BLA', 'C8E', 'D10', 'CHT', 'BO2', '3BV', 'ORO', 'MPO', 'Y01',
                      'OLC', 'B3P', 'G6P', 'PMP', 'D12', 'NDG', 'A3P', '78M', 'F6P',
                      'U5P', 'PRP', 'UPG', 'THM', 'SFG', 'MYR', 'FEO', 'PG0', 'CXS',
                      'AR6', 'CHD', 'WO4', 'C5P', 'UFP', 'GCP', 'HDD', 'SRT', 'STU',
                      'CDP', 'TCL', '04C', 'MYA', 'URA', 'PLG', 'MTA', 'BMP', 'SAL',
                      'TA1', 'UD1', 'OLA', 'BCN', 'LMR', 'BMA', 'OAA', 'TAM', 'MBO',
                      'MMA', 'SPD', 'MTE', 'AP5', 'TMP', 'PGA', 'GLA', '3PG', 'FUL',
                      'PQQ', '9TY', 'DUR', 'PPV', 'SPM', 'SIA', 'DUP', 'GTX', '1PG',
                      'GUN', 'ETF', 'FDP', 'MFU', 'G2P', 'PC', 'DST', 'INI']


def calculate_molecular_weight(input):
    """
    使用RDKit计算分子量的函数
    参数:
    input : (str or RDKit Mol)
    返回:
    float: 分子的分子量
    """
    # 判断输入类型
    if isinstance(input, str):
        # 如果是字符串，尝试将其解析为RDKit的Mol对象
        mol = Chem.MolFromSmiles(input)
        if mol is not None:
            return Descriptors.MolWt(mol)
        else:
            return None
    elif isinstance(input, Chem.Mol):
        mol = input
        if mol is not None:
            return Descriptors.MolWt(mol)
        else:
            return None
    else:
        return None


def filter_mol_by_elements(mol):
    """
    使用RDKit过滤掉包含金属元素的分子，并保留包含CHONPS和卤素的分子
    """
    elements_to_exclude = ['Fe']  # 要排除的金属元素
    # elements_to_exclude = ['Fe', 'Zn', 'Cu', 'Mg', 'Mn', 'Ca', 'Ni', 'K', 'Na']
    elements_to_include = ['C', 'H', 'O', 'N', 'P', 'S',  # CHONPS元素
                           'F', 'Cl', 'Br', 'I']  # 卤素元素
    if mol is not None:
        contains_metal = any(atom.GetSymbol() in elements_to_exclude for atom in mol.GetAtoms())
        contains_common_elements = all(atom.GetSymbol() in elements_to_include for atom in mol.GetAtoms())
        if not contains_metal and contains_common_elements:
            return mol
        else:
            return None
    else:
        return None

def get_sub_smiles(df_expo,res_name):
    df_res_smiles = df_expo[df_expo['ID'].values == res_name]['SMILES']
    if not df_res_smiles.empty:
        sub_smiles = df_res_smiles.values[0]
    else:
        chem_desc = pypdb.describe_chemical(f"{res_name}")
        ls_smiles = [item.get('descriptor') for item in chem_desc.get('pdbx_chem_comp_descriptor', []) if item.get('type') == 'SMILES']
        # ls_stereo_simles = chem_desc["rcsb_chem_comp_descriptor"]["smilesstereo"]
        if len(ls_smiles) == 1:
            sub_smiles = ls_smiles[0]
        elif len(ls_smiles) > 1:
            sub_smiles = ls_smiles[-1]
        # elif ls_stereo_simles is not None:
        #     sub_smiles = ls_stereo_simles
        else:
            sub_smiles = None
    return sub_smiles


##update 20231114-based on split_complex_v2.py
def process_ligand(ligand, res_name):
    """
    Add bond orders to a pdb ligand
    1. Select the ligand component with name "res_name"
    2. Get the corresponding SMILES from pypdb
    3. Create a template molecule from the SMILES in step 2
    4. Write the PDB file to a stream
    5. Read the stream into an RDKit molecule
    6. Assign the bond orders from the template from step 3
    :param ligand: ligand as generated by prody
    :param res_name: residue name of ligand to extract
    :return: molecule with bond orders assigned
    """
    output = StringIO()
    sub_mol = ligand.select(f"resname {res_name}")
    sub_smiles = get_sub_smiles(df_expo,res_name)
    if sub_smiles is not None:
        template = AllChem.MolFromSmiles(sub_smiles)
        writePDBStream(output, sub_mol)
        pdb_string = output.getvalue()
        rd_mol = AllChem.MolFromPDBBlock(pdb_string)
        try:
            new_mol = AllChem.AssignBondOrdersFromTemplate(template,rd_mol)
            return new_mol
        except:
            print("the bondorder is not correct!")
            return None
        # return rd_mol
    else:
        return None

def process_ligand_submol(sub_mol,ligand, res_name):
    """
    Add bond orders to a pdb ligand
    1. Select the ligand component with name "res_name"
    2. Get the corresponding SMILES from pypdb
    3. Create a template molecule from the SMILES in step 2
    4. Write the PDB file to a stream
    5. Read the stream into an RDKit molecule
    6. Assign the bond orders from the template from step 3
    :param ligand: ligand as generated by prody
    :param res_name: residue name of ligand to extract
    :return: molecule with bond orders assigned
    """
    output = StringIO()
    # sub_mol = ligand.select(f"resname {res_name}")
    sub_smiles = get_sub_smiles(df_expo, res_name)
    if sub_smiles is not None:
        template = AllChem.MolFromSmiles(sub_smiles)
        writePDBStream(output, sub_mol)
        pdb_string = output.getvalue()
        rd_mol = AllChem.MolFromPDBBlock(pdb_string)
        try:
            new_mol = AllChem.AssignBondOrdersFromTemplate(template,rd_mol)
            return new_mol
        except:
            print("the bondorder is not correct!")
            return None
            # return rd_mol
    else:
        return None


##update 20231112-14
# filter ligands by molecular weight,by elements
def process_ligand_filter_mw_element(ligand,res_name,mol_wt_cutoff_min,mol_wt_cutoff_max,filter_mw = True):
    rd_mol = process_ligand(ligand,res_name)
    # filter ligands by molecular weight,by elements
    molecule_weight = calculate_molecular_weight(rd_mol)
    # molecule_weight = chem_desc.get('chem_comp')['formula_weight']
    if filter_mw:
        # filter ligands by molecular weight
        if molecule_weight is not None:
            if mol_wt_cutoff_min <= molecule_weight <= mol_wt_cutoff_max:
                # filter ligands by by elements
                new_mol = filter_mol_by_elements(rd_mol)
            else:
                new_mol = None
        else:
            new_mol =None
    else:
        new_mol = filter_mol_by_elements(rd_mol)
    return new_mol

##update 20231112-14
# filter ligands by molecular weight,by elements
def process_ligand_filter_mw_element_rd_mol(rd_mol,mol_wt_cutoff_min,mol_wt_cutoff_max,filter_mw = True):
    # filter ligands by molecular weight,by elements
    molecule_weight = calculate_molecular_weight(rd_mol)
    # molecule_weight = chem_desc.get('chem_comp')['formula_weight']
    if filter_mw:
        if molecule_weight is not None:
            # filter ligands by molecular weight
            if mol_wt_cutoff_min <= molecule_weight <= mol_wt_cutoff_max:
                # filter ligands by by elements
                new_mol = filter_mol_by_elements(rd_mol)
            else:
                new_mol = None
        else:
            new_mol = None
    else:
        new_mol = filter_mol_by_elements(rd_mol)
    return new_mol

##获取xxx.ent_gz文件,update 20231117
def find_folders_with_ent_gz(root_folder,extension_name=".ent.gz"):
    result_paths = []
    for subdir, dirs, files in os.walk(root_folder):
        # check xxx.ent.gz
        pdb_gz_file = [f for f in files if f.endswith(extension_name)]
        # print(pdb_gz_file)
        if pdb_gz_file:
            for i in range(len(pdb_gz_file)):
                result_paths.append({
                'folder': subdir,
                'pdb_gz': os.path.join(subdir, pdb_gz_file[i]),
                })
    return result_paths

def delete_empty_subfolders(folder_path):
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path) and not os.listdir(subfolder_path):
            os.rmdir(subfolder_path)
            print(f"remove empty subfolder：{subfolder_path}")
        else:
            print(f"the subfolder is not empty,not remove：{subfolder_path}")

# update 20231120
def get_files_with_extensions_recursive(folder_path, valid_extensions):
    matching_files = []
    # 递归遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的绝对路径
            file_path = os.path.join(root, file)
            # 检查文件后缀是否在有效后缀列表中
            if any(file_path.lower().endswith(ext) for ext in valid_extensions):
                matching_files.append(file_path)
    return matching_files

def get_file_extension(file_name):
    # 使用字符串方法 rpartition 分离文件名和后缀
    _, _, extension = file_name.rpartition('.')
    return extension.lower()  # 将后缀名转换为小写，以便不区分大小写

def is_valid_extension(file_name, valid_extensions):
    # 获取文件后缀名
    extension = get_file_extension(file_name)
    # 检查后缀名是否在有效后缀列表中
    return extension in valid_extensions


##20231112udate-20231117
def process_main_by_pdb_gz_file(pdb_file_name,pdb_name):
    """
    Read Ligand Expo data, split pdb into protein and ligands,
    write protein pdb, write ligand sdf files
    :param pdb_name: "pdb8ahx.ent.gz" or "8ahx.pdb.gz" or "8ahx.pdb"
    :return:
    """
    main_folder = os.getcwd()
    print("the floder is:",main_folder)
    if is_valid_extension(pdb_file_name,".pdb") and get_pdb_components_by_pdbfile(pdb_file_name) is not None:
        protein, ligand = get_pdb_components_by_pdbfile(pdb_file_name)
        if protein is None:
            print("the {} is not protein and not process!".format(pdb_name))
        else:
            pdb_split_folder_name = "{}_split".format(pdb_name)
            if not os.path.exists(pdb_split_folder_name):
                os.makedirs(pdb_split_folder_name)
            os.chdir(pdb_split_folder_name)
            if ligand is not None:
                # filter ligands by ubiquitous ligands
                res_name_list = list(set(ligand.getResnames()))
                for res in res_name_list:
                    atom_bond = find_bonds_from_pdb_file(pdb_name, res)
                    if atom_bond == True:
                        print("the res {} is bond to protein!".format(res))
                        continue
                    elif res in ubiquitous_ligands:
                        print("the res {} is ubiquitous ligand!".format(res))
                        continue
                    else:
                        """同一蛋白包含多个相同配体的处理"""
                        allres = ligand.select(f"resname {res}")
                        res_id = np.unique(allres.getResindices())
                        # res_id = np.unique(ligand.getResindices())
                        if len(res_id) > 5:
                            break
                        elif len(res_id) > 1 and len(res_id) < 5:
                            search_break = False
                            for i in res_id:
                                sub_mol = ligand.select(f"resname {res} and resindex {i}")
                                # output = StringIO()
                                if sub_mol and search_break != True:
                                    # writePDBStream(output, sub_mol)
                                    # pdb_string = output.getvalue()
                                    # rd_mol = AllChem.MolFromPDBBlock(pdb_string)
                                    rd_mol = process_ligand_submol(sub_mol, ligand, res)
                                    new_mol = process_ligand_filter_mw_element_rd_mol(rd_mol, mol_wt_cutoff_min,
                                                                                      mol_wt_cutoff_max, filter_mw)
                                    if new_mol is not None:
                                        pdb_split_sdf_folder_name = "{}_{}_{}".format(pdb_name, str(i), res)
                                        if not os.path.exists(pdb_split_sdf_folder_name):
                                            os.makedirs(pdb_split_sdf_folder_name)
                                            os.chdir(os.path.join(os.getcwd(), pdb_split_sdf_folder_name))
                                        # os.chdir(pdb_split_sdf_folder_name)
                                        # os.chdir(os.path.join(os.getcwd(),pdb_split_sdf_folder_name))
                                        write_sdf(new_mol, pdb_name, str(i) + "_" + res)
                                        write_pdb(protein, pdb_name)
                                        ##no xyz ligand
                                        input_sdf = "{}_{}_{}_ligand.sdf".format(pdb_name, str(i), res)
                                        if os.path.isfile(input_sdf):
                                            write_smiles_from_sdf_to_sdf(input_sdf, pdb_name, str(i) + "_" + res)
                                        os.chdir(os.path.join(main_folder, pdb_split_folder_name))
                                        search_break = True
                        else:
                            new_mol = process_ligand_filter_mw_element(ligand, res, mol_wt_cutoff_min,
                                                                       mol_wt_cutoff_max, filter_mw)
                            if new_mol is not None:
                                pdb_split_sdf_folder_name = "{}_{}".format(pdb_name, res)
                                if not os.path.exists(pdb_split_sdf_folder_name):
                                    os.makedirs(pdb_split_sdf_folder_name)
                                    os.chdir(os.path.join(os.getcwd(), pdb_split_sdf_folder_name))
                                # os.chdir(pdb_split_sdf_folder_name)
                                # os.chdir(os.path.join(os.getcwd(), pdb_split_sdf_folder_name))
                                write_sdf(new_mol, pdb_name, res)
                                write_pdb(protein, pdb_name)
                                ##no xyz ligand
                                input_sdf = "{}_{}_ligand.sdf".format(pdb_name, res)
                                if os.path.isfile(input_sdf):
                                    write_smiles_from_sdf_to_sdf(input_sdf, pdb_name, res)
                                os.chdir(os.path.join(main_folder, pdb_split_folder_name))
            os.chdir(main_folder)
    # os.path.basename("pdb8q2e.pdb.gz").split('.')[-3][3:]
    # '8q2e'
    if is_valid_extension(pdb_file_name,".gz") and get_pdb_components_by_file_gz(pdb_file_name) is not None:
        protein, ligand = get_pdb_components_by_file_gz(pdb_file_name)
        # "pdb8ahx.ent.gz" or "8ahx.pdb.gz"
        # pdb_name = os.path.basename(pdb_file_name).split('.')[-3][3:]
        # # pdb_name = os.path.basename(pdb_file_name).split('.')[0]
        if protein is None:
            print("the {} is not protein and not process!".format(pdb_name))
        else:
            pdb_split_folder_name = "{}_split".format(pdb_name)
            if not os.path.exists(pdb_split_folder_name):
                os.makedirs(pdb_split_folder_name)
            os.chdir(pdb_split_folder_name)
            if ligand is not None:
                # filter ligands by ubiquitous ligands
                res_name_list = list(set(ligand.getResnames()))
                for res in res_name_list:
                    atom_bond = find_bonds_from_pdb_file(pdb_name,res)
                    if atom_bond == True:
                        print("the res {} is bond to protein!".format(res))
                        continue
                    elif res in ubiquitous_ligands:
                        print("the res {} is ubiquitous ligand!".format(res))
                        continue
                    else:
                        """同一蛋白包含多个相同配体的处理"""
                        allres = ligand.select(f"resname {res}")
                        res_id = np.unique(allres.getResindices())
                        # res_id = np.unique(ligand.getResindices())
                        if len(res_id) > 5:
                            break
                        elif len(res_id) > 1 and len(res_id) < 5:
                            search_break = False
                            for i in res_id:
                                sub_mol = ligand.select(f"resname {res} and resindex {i}")
                                # output = StringIO()
                                if sub_mol and search_break != True:
                                    # writePDBStream(output, sub_mol)
                                    # pdb_string = output.getvalue()
                                    # rd_mol = AllChem.MolFromPDBBlock(pdb_string)
                                    rd_mol = process_ligand_submol(sub_mol,ligand,res)
                                    new_mol = process_ligand_filter_mw_element_rd_mol(rd_mol,mol_wt_cutoff_min,mol_wt_cutoff_max,filter_mw)
                                    if new_mol is not None:
                                        pdb_split_sdf_folder_name = "{}_{}_{}".format(pdb_name,str(i),res)
                                        if not os.path.exists(pdb_split_sdf_folder_name):
                                            os.makedirs(pdb_split_sdf_folder_name)
                                            os.chdir(os.path.join(os.getcwd(), pdb_split_sdf_folder_name))
                                        # os.chdir(pdb_split_sdf_folder_name)
                                        # os.chdir(os.path.join(os.getcwd(),pdb_split_sdf_folder_name))
                                        write_sdf(new_mol, pdb_name, str(i)+"_"+res)
                                        write_pdb(protein, pdb_name)
                                        ##no xyz ligand
                                        input_sdf = "{}_{}_{}_ligand.sdf".format(pdb_name,str(i),res)
                                        if os.path.isfile(input_sdf):
                                            write_smiles_from_sdf_to_sdf(input_sdf,pdb_name, str(i)+"_"+res)
                                        os.chdir(os.path.join(main_folder, pdb_split_folder_name))
                                        search_break = True
                        else:
                            new_mol = process_ligand_filter_mw_element(ligand,res,mol_wt_cutoff_min,mol_wt_cutoff_max,filter_mw)
                            if new_mol is not None:
                                pdb_split_sdf_folder_name = "{}_{}".format(pdb_name,res)
                                if not os.path.exists(pdb_split_sdf_folder_name):
                                    os.makedirs(pdb_split_sdf_folder_name)
                                    os.chdir(os.path.join(os.getcwd(), pdb_split_sdf_folder_name))
                                # os.chdir(pdb_split_sdf_folder_name)
                                # os.chdir(os.path.join(os.getcwd(), pdb_split_sdf_folder_name))
                                write_sdf(new_mol, pdb_name, res)
                                write_pdb(protein, pdb_name)
                                ##no xyz ligand
                                input_sdf = "{}_{}_ligand.sdf".format(pdb_name, res)
                                if os.path.isfile(input_sdf):
                                    write_smiles_from_sdf_to_sdf(input_sdf, pdb_name, res)
                                os.chdir(os.path.join(main_folder,pdb_split_folder_name))
            os.chdir(main_folder)


if __name__ == "__main__":
    # minimum molecular weight to consider sth a ligand
    mol_wt_cutoff_min = 140
    mol_wt_cutoff_max = 800
    filter_mw = True
    # 定义有效的文件后缀列表
    valid_extensions = ['.pdb', '.ent.gz', '.pdb.gz']
    # "pdb8ahx.ent.gz" or "8ahx.pdb.gz" or "8ahx.pdb"
    df_expo = read_ligand_expo()
    # set the pdb file work dir
    pdb_file_dir = r"D:\pycharmprojects\pdbdataset\pdb-test"
    # set the result work dir
    main_folder = os.chdir(r"D:\pycharmprojects\pdbdataset\pdb-test-result")
    f_exist = [pdbs.split("_")[0] for pdbs in os.listdir(main_folder)]
    matching_files = get_files_with_extensions_recursive(pdb_file_dir, valid_extensions)
    # print(matching_files)
    for pdb_file_name in matching_files:
        # print(pdb_file_name)
        if is_valid_extension(pdb_file_name,".pdb"):
            pdb_name = os.path.basename(pdb_file_name).split(".")[0]
            if pdb_name not in f_exist:
                print(pdb_name)
                process_main_by_pdb_gz_file(pdb_file_name,pdb_name)
        elif is_valid_extension(pdb_file_name,".gz"):
            pdb_base_name = os.path.basename(pdb_file_name).split(".")[0]
            ##"pdb8ahx.ent.gz"
            if len(pdb_base_name) == 7:
                pdb_name = os.path.basename(pdb_file_name).split(".")[-3][3:]
                if pdb_name not in f_exist:
                    print(pdb_name)
                    process_main_by_pdb_gz_file(pdb_file_name, pdb_name)
            ##"8ahx.pdb.gz"
            if len(pdb_base_name) == 4:
                pdb_name = pdb_base_name
                if pdb_name not in f_exist:
                    print(pdb_name)
                    process_main_by_pdb_gz_file(pdb_file_name, pdb_name)
    main_folder = os.getcwd()
    print("the floder is:", main_folder)
    delete_empty_subfolders(main_folder)
    print("finshed!")


