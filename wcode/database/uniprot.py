import os.path
import requests
import requests as r
import requests_cache
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


def get_uniprot_ids(pdb_id):
    # 定义PDBe API的URL
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"

    try:
        # 发送GET请求
        response = requests.get(url)
        # 检查响应状态码
        response.raise_for_status()

        # 解析JSON响应
        data = response.json()

        # 提取UniProt ID
        uniprot_ids = []
        if pdb_id in data and "UniProt" in data[pdb_id]:
            uniprot_data = data[pdb_id]["UniProt"]
            for uniprot_id in uniprot_data.keys():
                uniprot_ids.append(uniprot_id)

        return uniprot_ids

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []


########################################################################################################################


if __name__ == '__main__':
    pdb_id = "7eu8"
    uniprot_ids = get_uniprot_ids(pdb_id)
    print(uniprot_ids)