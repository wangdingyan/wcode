import os.path

import requests as r
from Bio import SeqIO
from io import StringIO

########################################################################################################################

def seq2file(uniprot_id, filepath):
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+uniprot_id+".fasta"
    response = r.post(currentUrl)
    cData=''.join(response.text)

    with open(os.path.abspath(filepath), 'w+') as f:
        for l in cData:
            f.write(l)

    # Seq=StringIO(cData)
    # pSeq=list(SeqIO.parse(Seq,'fasta'))
    return None

########################################################################################################################
