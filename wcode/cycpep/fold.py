import subprocess
import os
from wcode.utils.config import TPATH

FOLD_PNEAR = '''mpirun -np {mpi_n} {simple_cycpep_predict}  \
                -sequence_file seq.txt  \
                -nstruct {n_struct}  \
                -corrections::beta_nov16  \
                -beta_nov16  \
                -cyclic_peptide:genkic_closure_attempts 100  \
                -cyclic_peptide:rama_cutoff 5.0  \
                -fast_relax_rounds 3  \
                -cyclic_peptide:genkic_min_solution_count 1  \
                -cyclic_peptide:use_rama_filter true  \
                -symmetric_gly_tables true  \
                -cyclic_peptide:cyclization_type "n_to_c_amide_bond"  \
                -cyclic_peptide:MPI_auto_2level_distribution  \
                -cyclic_peptide:MPI_batchsize_by_level 125  \
                -cyclic_peptide:MPI_pnear_lambda 1.0  \
                -cyclic_peptide:MPI_pnear_kbt 0.62  \
                -cyclic_peptide:default_rama_sampling_table "flat_symm_dl_aa_ramatable"  \
                -min_genkic_hbonds 0  \
                -min_final_hbonds 0  \
                -mute all  \
                -unmute protocols.cyclic_peptide_predict.SimpleCycpepPredictApplication_MPI_summary  \
                -cyclic_peptide:MPI_output_fraction 0.002  \
                -cyclic_peptide:compute_rmsd_to_lowest  \
                -out:file:silent out.silent >> run.log'''


def fold_pnear(root,
               seq,
               mpi_n=10,
               simple_cycpep_predict=TPATH.SIMPLEPEP,
               n_struct=10000):
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    seq_file = '{}/seq.txt'.format(root)
    fold_sh = '{}/fold.sh'.format(root)

    with open(seq_file, 'w') as fp:
        fp.write(seq)

    with open(fold_sh, 'w') as fp:
        fp.write(FOLD_PNEAR.format(mpi_n=mpi_n, simple_cycpep_predict=simple_cycpep_predict, n_struct=n_struct))

    subprocess.run('bash fold.sh', cwd=root, shell=True)
