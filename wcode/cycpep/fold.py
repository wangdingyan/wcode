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
                -cyclic_peptide:MPI_output_fraction 1  \
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
                -mute all  \
                -unmute protocols.cyclic_peptide_predict.SimpleCycpepPredictApplication_MPI_summary  \
                -cyclic_peptide:compute_rmsd_to_lowest  >> run.log'''


def fold_pnear(root,
               seq,
               mpi_n=10,
               lamd=1.0,
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


if __name__ == '__main__':
    for n_struct in [100, 1000, 10000, 100000]:
        for n in range(20):
            # fold_pnear(f'/mnt/c/tmp/validPNEAR/A1/{n_struct}_{n}',  mpi_n=8, seq='ASP THR DASN DPRO THR DLYS ASN', n_struct=n_struct, lamd=0.5)
            # fold_pnear(f'/mnt/c/tmp/validPNEAR/A2/{n_struct}_{n}',  mpi_n=8, seq='ASP THR DASN DHIS THR DLYS ASN', n_struct=n_struct, lamd=0.5)
            fold_pnear(f'/mnt/c/tmp/validPNEAR/A3/{n_struct}_{n}',  mpi_n=4, seq='ASP THR DASN PRO THR DLYS ASN', n_struct=n_struct, lamd=0.5)


