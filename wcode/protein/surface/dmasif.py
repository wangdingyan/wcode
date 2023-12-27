import os
import glob
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import DataLoader
from Bio.PDB import StructureBuilder
from Bio.PDB import *
from Bio.SeqUtils import IUPACData
from torch_geometric.transforms import Compose
from wcode.protein.surface.data import NormalizeChemFeatures, CenterPairAtoms, RandomRotationPairAtoms, load_protein_pair

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

# Exclude disordered atoms.

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}

class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1"


def load_structure_np(fname, center):
    """Loads a .ply mesh to return a point cloud and connectivity."""
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set

def extractPDB(
    infilename, outfilename, chain_ids=None
):
    # extract the chain_ids from infilename and save in outfilename.
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if (
            chain_ids == None
            or chain.get_id() in chain_ids
        ):
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())

def convert_to_npy(target_pdb, chains_dir, npy_dir, chains = ['A']):
    pdb_id = target_pdb.split('\\')[-1]
    pdb_id = pdb_id[:-4]
    protonated_file = target_pdb
    pdb_filename = protonated_file

    # Extract chains of interest.
    for chain in chains:
        out_filename = chains_dir +"/"+ "{id}_{chain}.pdb".format(id=pdb_id,chain=chain)
        extractPDB(pdb_filename, str(out_filename), chain)
        protein = load_structure_np(out_filename,center=False)
        np.save(npy_dir +"/"+ "{id}_{chain}_atomxyz".format(id=pdb_id,chain=chain), protein["xyz"])
        np.save(npy_dir +"/"+ "{id}_{chain}_atomtypes".format(id=pdb_id,chain=chain), protein["types"])


def generate_descr(model_path, output_path, pdb_file, npy_directory, radius, resolution, supsampling):
    """Generat descriptors for a MaSIF site model"""
    parser = argparse.ArgumentParser(description="Network parameters")
    parser.add_argument("--experiment_name", type=str, default=model_path)
    parser.add_argument("--use_mesh", type=bool, default=False)
    parser.add_argument("--embedding_layer", type=str, default="dMaSIF")
    parser.add_argument("--curvature_scales", type=list, default=[1.0, 2.0, 3.0, 5.0, 10.0])
    parser.add_argument("--resolution", type=float, default=resolution)
    parser.add_argument("--distance", type=float, default=1.05)
    parser.add_argument("--variance", type=float, default=0.1)
    parser.add_argument("--sup_sampling", type=int, default=supsampling)
    parser.add_argument("--atom_dims", type=int, default=6)
    parser.add_argument("--emb_dims", type=int, default=16)
    parser.add_argument("--in_channels", type=int, default=16)
    parser.add_argument("--orientation_units", type=int, default=16)
    parser.add_argument("--unet_hidden_channels", type=int, default=8)
    parser.add_argument("--post_units", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--radius", type=float, default=radius)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--site", type=bool, default=True)  # set to true for site model
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--search", type=bool, default=False)  # Set to true for search model
    parser.add_argument("--single_pdb", type=str, default=pdb_file)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_rotation", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cpu")
    # parser.add_argument("--single_protein",type=bool,default=True)
    parser.add_argument("--single_protein", type=bool, default=True)  # set to false for site
    parser.add_argument("--no_chem", type=bool, default=False)
    parser.add_argument("--no_geom", type=bool, default=False)

    args = parser.parse_args("")

    model_path = args.experiment_name
    save_predictions_path = Path(output_path)

    # Ensure reproducability:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Load the train and test datasets:
    transformations = (
        Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
        if args.random_rotation
        else Compose([NormalizeChemFeatures()])
    )

    if args.single_pdb != "":
        single_data_dir = Path(npy_directory)
        test_dataset = [load_protein_pair(args.single_pdb, single_data_dir, single_pdb=True)]
        test_pdb_ids = [args.single_pdb]

    # PyTorch geometric expects an explicit list of "batched variables":
    batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, follow_batch=batch_vars
    )
    for d in test_loader:
        print(d)


    net = dMaSIF(args)
    # net.load_state_dict(torch.load(model_path, map_location=args.device))
    net.load_state_dict(torch.load(model_path, map_location=args.device)["model_state_dict"])
    net = net.to(args.device)

    # Perform one pass through the data:
    info = iterate(
        net,
        test_loader,
        None,
        args,
        test=True,
        save_path=save_predictions_path,
        pdb_ids=test_pdb_ids,
    )


if __name__ == '__main__':
    target_pdb = 'C:\\tmp\\dmasif\\1NPU.pdb'
    target_name = "1NPU"
    chains_dir = 'C:\\tmp\\dmasif\\chains'
    chain_name = 'A'
    model_path = '/content/MaSIF_colab/models/dMaSIF_site_3layer_16dims_12A_0.7res_150sup_epoch59'
    resolution = 0.7
    radius = 12
    supsampling = 100


    isExist = os.path.exists(chains_dir)
    if not isExist:
        os.makedirs(chains_dir)
    else:
        files = glob.glob(chains_dir + '/*')
        for f in files:
            os.remove(f)

    npy_dir = 'C:\\tmp\\dmasif\\npys'
    isExist = os.path.exists(npy_dir)
    if not isExist:
        os.makedirs(npy_dir)
    else:
        files = glob.glob(npy_dir + '/*')
        for f in files:
            os.remove(f)

    pred_dir = 'C:\\tmp\\dmasif\\preds'
    isExist = os.path.exists(pred_dir)
    if not isExist:
        os.makedirs(pred_dir)
    else:
        files = glob.glob(pred_dir + '/*')
        for f in files:
            os.remove(f)

    chains = ['A']

    # Generate the surface features
    convert_to_npy(target_pdb, chains_dir, npy_dir, chains)

    # Generate the embeddings
    pdb_name = "{n}_{c}_{c}".format(n=target_name, c=chain_name)
    info = generate_descr(model_path, pred_dir, pdb_name, npy_dir, radius, resolution, supsampling)
