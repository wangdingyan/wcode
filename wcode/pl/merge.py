from rdkit import Chem

########################################################################################################################

def merge_protein_ligand_file(protein_file,
                              ligand_file,
                              output_path=None):
    '''
    ligand_file: MUST be .sdf format with 3D coordinates.
    '''
    if output_path is None:
        output_path = protein_file.replace('.pdb', '_merge.pdb')
    ligand_mol = next(Chem.SDMolSupplier(ligand_file, sanitize=False))
    ligand_mol = Chem.RemoveHs(ligand_mol)
    ligand_smiles = Chem.MolToSmiles(ligand_mol)
    protein_mol = Chem.MolFromPDBFile(protein_file, sanitize=False)
    protein_mol = Chem.RemoveHs(protein_mol)
    merge_mol = Chem.CombineMols(protein_mol, ligand_mol)
    Chem.MolToPDBFile(merge_mol, output_path)
    with open(output_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('ATOM'):
                res_num = int(line[22:26])

        res_num += 1
        res_num = list(str(res_num).rjust(4))
        for i, line in enumerate(lines):
            if line.startswith('HETATM'):
                line = list(line)
                line[21] = 'Z'
                line[17], line[18], line[19] = 'L', 'I', 'G'
                line[22], line[23], line[24], line[25] = res_num
                line = "".join(line)
                lines[i] = line
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line)
    return output_path, ligand_smiles