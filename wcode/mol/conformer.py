from rdkit import Chem
from rdkit.Chem import AllChem


def generate_conformers(input_pdb_file, output_pdb_file):
    # 读取输入的PDB文件
    mol = Chem.MolFromPDBFile(input_pdb_file, sanitize=False)

    # 生成二维拓扑结构
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    mol = Chem.AddHs(mol)

    # 使用MM构象生成方法生成三维构象
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # 保存生成的PDB文件
    Chem.MolToPDBFile(mol, output_pdb_file)
    return None


if __name__ == '__main__':
    import os
    file_list = os.listdir('/mnt/d/nutshell/Official/AIPROJECT/CycPepModel/2017_ScienceTest')
    print(file_list)
    for f in file_list:
        print(f)
        if f.endswith('.pdb'):
            try:
                name = f.split('.')[0]
                output_name = name + '_random.pdb'
                input_name = name + '.pdb'
                generate_conformers('/mnt/d/nutshell/Official/AIPROJECT/CycPepModel/2017_ScienceTest/' + input_name,
                                    '/mnt/d/nutshell/Official/AIPROJECT/CycPepModel/2017_ScienceTest/' + output_name)
            except:
                pass

    # generate_conformers('/mnt/d/nutshell/Official/AIPROJECT/CycPepModel/2017_ScienceTest/7-1_6BE9.pdb',
    #                     '/mnt/d/nutshell/Official/AIPROJECT/CycPepModel/2017_ScienceTest/7-1_6BE9_random.pdb')
