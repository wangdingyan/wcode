import json
import re
import warnings
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from pypdb.util import http_requests
from pypdb import get_info


def pdb_ligand_to_pubchem_RAW(pdb_ligand_id : str) -> Optional[str]:
    # Set the request URL
    request_url = f'https://www.rcsb.org/ligand/{pdb_ligand_id}'
    # Run the query
    parsed_response = None
    try:
        with urlopen(request_url) as response:
            parsed_response = response.read().decode("utf-8")
    # If the accession is not found in the PDB then we can stop here
    except HTTPError as error:
        if error.code == 404:
            print(f' PDB ligand {pdb_ligand_id} not found')
            return None
        else:
            print(error.msg)
            raise ValueError('Something went wrong with the PDB ligand request: ' + request_url)
    # Mine the pubchem id out of the whole response
    pattern = re.compile('pubchem.ncbi.nlm.nih.gov\/compound\/([0-9]*)\"')
    match = re.search(pattern, parsed_response)
    # If there is no pubchem id then return none
    # This is normal for some ligands such as counter ions (e.g. LI)
    if not match:
        return None
    pubchem_id = match[1]
    return pubchem_id


def pdb_id_to_pdb_ligand(pdb_id: str) -> Optional[str]:
    info = get_info(pdb_id)
    non_polymer_entity_ids = info['rcsb_entry_container_identifiers'].get('non_polymer_entity_ids', None)
    pdb_ligands = [get_pdb_ligand_from_npe_id(pdb_id, npe_id) for npe_id in non_polymer_entity_ids]
    return pdb_ligands


def get_pdb_ligand_from_npe_id(pdb_id: str, npe_id):
    url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{npe_id}"
    response = http_requests.request_limited(url)

    if response is None or response.status_code != 200:
        warnings.warn("Retrieval failed, returning None")
        return None

    result = str(response.text)
    out = json.loads(result)
    return out['pdbx_entity_nonpoly']['comp_id']


if __name__ == '__main__':
    print(pdb_id_to_pdb_ligand('5p21'))