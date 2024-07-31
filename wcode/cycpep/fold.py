import re
import os
import subprocess
from wcode.utils.config import TPATH


FOLD_PNEAR = '''mpirun -np {mpi_n} {simple_cycpep_predict}  \
                -sequence_file seq.txt  \
                -cyclic_peptide:cyclization_type "n_to_c_amide_bond"  \
                -nstruct {n_struct}  \
                -out:file:silent out.silent \
                -cyclic_peptide:MPI_batchsize_by_level 125  \
                -cyclic_peptide:MPI_auto_2level_distribution  \
                -cyclic_peptide:MPI_choose_highest false \
                -cyclic_peptide:MPI_sort_by energy \
                -score:symmetric_gly_tables true  \
                -cyclic_peptide:genkic_closure_attempts 250  \
                -cyclic_peptide:genkic_min_solution_count 1  \
                -cyclic_peptide:use_rama_filter true  \
                -cyclic_peptide:rama_cutoff 3.0  \
                -min_genkic_hbonds 0  \
                -min_final_hbonds 0  \
                -cyclic_peptide:MPI_pnear_lambda {lamd}  \
                -cyclic_peptide:MPI_pnear_kbt 0.62  \
                -cyclic_peptide:MPI_output_fraction {frac} \
                -mute all  \
                -unmute protocols.cyclic_peptide_predict.SimpleCycpepPredictApplication_MPI_summary  \
                -cyclic_peptide:compute_rmsd_to_lowest  >> run.log'''

SPLIT_SILENT_COMMAND = '''{extract_pdbs} -in::file::silent {silent_file} -in:auto_setup_metals'''

def fold_pnear(root,
               seq,
               mpi_n=10,
               lamd=1.0,
               frac=0.05,
               simple_cycpep_predict=TPATH.SIMPLEPEP,
               n_struct=10000):
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    seq_file = '{}/seq.txt'.format(root)
    fold_sh = '{}/fold.sh'.format(root)

    with open(seq_file, 'w') as fp:
        fp.write(seq)

    with open(fold_sh, 'w') as fp:
        fp.write(FOLD_PNEAR.format(mpi_n=mpi_n,
                                   lamd=lamd,
                                   frac=frac,
                                   simple_cycpep_predict=simple_cycpep_predict,
                                   n_struct=n_struct))

    subprocess.run('bash fold.sh', cwd=root, shell=True)


def extract_pnear(root):
    log_file ='{}/run.log'.format(root)

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("PNearLowest"):
                match = re.search(r'\d+\.\d+', line)
                if match:
                    number = float(match.group())
                    return number
    return None


def extract_pdb_files(silent_file):
    EXTRACT_PDBS = SPLIT_SILENT_COMMAND.format(extract_pdbs=TPATH.SILENT_SPLIT,
                                               silent_file=silent_file)
    subprocess.run(EXTRACT_PDBS, shell=True)


if __name__ == '__main__':
    # for ID, seq in [('7.1', 'DASP DTHR ASN PRO DTHR LYS DASN'),
    #                 ('7.2', 'DASP DGLN DSER DGLU PRO DHIS PRO'),
    #                 ('7.3', 'DGLN DASP DPRO PRO DLYS THR ASP'),
    #                 ('8.1', 'DASP DASP DPRO DTHR PRO ARG DGLN GLN'),
    #                 ('8.2', 'DARG GLN DPRO DGLN ARG DGLU PRO GLN'),
    #                 ('9.1', 'LYS ASP LEU DGLN DPRO PRO TYR DHIS PRO'),
    #                 ('10.1', 'PRO GLU ALA ALA ARG DVAL DPRO ARG DLEU DTHR'),
    #                 ('10.2', 'GLU DVAL ASP PRO DGLU DHIS DPRO ASN DALA DPRO')]:
    #     fold_pnear(f'/mnt/c/tmp/2017_Science2/{ID}',  mpi_n=8, seq=seq, n_struct=10000, lamd=0.5)
    # fold_pnear(f'/mnt/c/tmp/2017_Science2/3AVM', mpi_n=8, seq='SER ARG LYS ILE ASP ASN LEU ASP', n_struct=100000, lamd=0.5)
    extract_pdb_files('/mnt/c/tmp/2017_Science2/10.1/out.silent')


