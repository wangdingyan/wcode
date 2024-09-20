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


def id2seq(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        fasta_data = response.text
        # 去掉FASTA格式的注释行(>开头的行)
        sequence = ''.join(fasta_data.splitlines()[1:])
        return sequence
    else:
        return f"Error: Unable to fetch data for {uniprot_id}, status code {response.status_code}"

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
    pdb_id = ["Q6ZNJ1",
              "Q96P70",
              "Q7Z6Z7",
              "Q70CQ2",
              "Q15386",
              "Q14151",
              "Q13838",
              "Q13535",
              "P78527",
              "P50748",
              "P21980",
              "P13010",
              "Q6YHU6",
              "O43149",
              "Q9ULT8",
              "Q70CQ2",
              "Q5JSL3",
              "Q9UBB4",
              "O94822",
              "P04150",
              "Q9HAV4",
              "Q13085",
              "O75691",
              "Q9Y2I7",
              "O14980",
              "P50748",
              "Q15386",
              "Q08AM6",
              "Q96T76",
              "Q8N3C0",
              "Q8N1F7",
              "P55060",
              "Q9Y6D5",
              "Q6PJG6",
              "Q92616",
              "Q9UIA9",
              "Q14192",
              "Q9NRW3",
              "Q96Q15",
              "P43487",
              "Q8IWV8",
              "O15287",
              "Q6P2Q9",
              "Q14CX7",
              "Q5VYK3",
              "Q7RTV0",
              "P31949",
              "P52306",
              "Q9HAB8",
              "Q7L576",
              "Q09472",
              "P42345",
              "O14981",
              "P61978",
              "Q9H000",
              "P50579",
              "P60228",
              "Q9BXI6",
              "Q9P265",
              "Q9UBB6",
              "Q9Y6R0"]
    seq_list = []
    for name in pdb_id:
        seq = id2seq(name)
        print(name, seq)
        seq_list.append(seq)
    import pandas as pd
    df = pd.DataFrame({'UniProtID': pdb_id, 'Sequence': seq_list})
    df.to_excel('C:\\tmp\\20240919_JHW.xlsx')