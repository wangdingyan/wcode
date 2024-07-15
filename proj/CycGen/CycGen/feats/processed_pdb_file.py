# https://github.com/clauswilke/PeptideBuilder/blob/master/examples/evaluation.py
import os
import pickle
import json
import torch
import numpy as np
import random
from pathlib import Path
import math

from Bio.PDB import PDBIO
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
from Bio.PDB.vectors import calc_angle, calc_dihedral

from PeptideBuilder import Geometry
import PeptideBuilder

from rdkit import RDLogger
import networkx as nx
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')
import sys

from wcode.protein.constant import STANDARD_AMINO_ACID_MAPPING_3_TO_1

def preprocess_pdb_files(base_path):
    output_dic = {}
    file_names = os.listdir(base_path)

    for name in file_names:
        bonds = []
        angles = []
        dihedrals = []
        sequence = []

        sample_name = name.split('.')[0]
        output_dic[sample_name] = {}
        parser = PDBParser()
        structure = parser.get_structure("sample", os.path.join(base_path, sample_name+'.pdb'))

        model = structure[0]
        chain = next(model.get_chains())
        prev = "0"
        N_prev = "0"
        CA_prev = "0"

        rad = 180.0 / math.pi
        for res in chain:
            res_1 = STANDARD_AMINO_ACID_MAPPING_3_TO_1[res.get_resname()]
            sequence.append(res_1)

            if prev == "0":
                N_prev = res["N"]
                CA_prev = res["CA"]
                C_prev = res["C"]
                prev = "1"
            else:
                n1 = N_prev.get_vector()
                ca1 = CA_prev.get_vector()
                c1 = C_prev.get_vector()

                N_curr = res["N"]
                CA_curr = res["CA"]
                C_curr = res["C"]

                n = N_curr.get_vector()
                ca = CA_curr.get_vector()
                c = C_curr.get_vector()

                # length
                peptide_bond = N_curr - C_prev
                CA_N_length = CA_curr - N_curr
                CA_C_length = CA_curr - C_curr
                bonds.extend([peptide_bond, CA_N_length, CA_C_length])

                # angle
                CA_C_N_angle = calc_angle(ca1, c1, n) * rad
                C_N_CA_angle = calc_angle(c1, n, ca) * rad
                N_CA_C_angle = calc_angle(n, ca, c) * rad
                angles.extend([CA_C_N_angle,C_N_CA_angle, N_CA_C_angle])

                # dihedral
                psi = calc_dihedral(n1, ca1, c1, n)  ##goes to current res
                omega = calc_dihedral(ca1, c1, n, ca)  ##goes to current res
                phi = calc_dihedral(c1, n, ca, c)  ##goes to current res

                psi_im1 = psi * rad
                omega = omega * rad
                phi = phi * rad
                dihedrals.extend([psi_im1, omega, phi])

                N_prev = res["N"]
                CA_prev = res["CA"]
                C_prev = res["C"]

        sequence = ''.join(sequence)
        output_dic[sample_name]['seq'] = sequence
        output_dic[sample_name]['bond'] = bonds
        output_dic[sample_name]['angle'] = angles
        output_dic[sample_name]['dihedral'] = dihedrals

    return output_dic

def reconstruct_cyc_peptide(profile):
    model_structure_geo = []
    for i, res in enumerate(profile['seq']):
        geo = Geometry.geometry(res)
        if i == 0:
            pass
        else:
            # bond
            geo.peptide_bond = profile['bond'][(i-1)*3]
            geo.CA_N_length = profile['bond'][(i-1)*3+1]
            geo.CA_C_length = profile['bond'][(i-1)*3+2]

            # angle
            geo.CA_C_N_angle = profile['angle'][(i-1)*3]
            geo.C_N_CA_angle = profile['angle'][(i-1)*3+1]
            geo.N_CA_C_angle = profile['angle'][(i-1)*3+2]

            # dihedral
            geo.psi_im1 = profile['dihedral'][(i-1)*3]
            geo.omega = profile['dihedral'][(i-1)*3+1]
            geo.phi = profile['dihedral'][(i-1)*3+2]
        model_structure_geo.append(geo)
    return model_structure_geo

def make_pdb_file(struct, file_nom):
    outfile = PDBIO()
    outfile.set_structure(PeptideBuilder.make_structure_from_geos(struct))
    outfile.save(file_nom)
    return file_nom

if __name__ == '__main__':
    # d = preprocess_pdb_files('/mnt/c/tmp/peptide_cys_28')
    # struct = reconstruct_cyc_peptide(d['5f88-CP'])
    # make_pdb_file(struct, '/mnt/c/tmp/5f88-CP_rebuilt.pdb')
    d = preprocess_pdb_files('/mnt/c/tmp/peptide_cys_28')
    with open('/mnt/c/Official/WCODE/proj/CycGen/cys28.pkl', 'wb') as f:
        pickle.dump(d, f)


