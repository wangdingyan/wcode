import requests
import json
from bs4 import BeautifulSoup
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode


def smiles_to_pubchem_id(smiles : str) -> Optional[str]:
    # Set the request URL
    request_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/JSON'
    # Set the POST data
    data = urlencode({'smiles': smiles}).encode()
    try:
        with urlopen(request_url, data=data) as response:
            parsed_response = json.loads(response.read().decode("utf-8"))
    # If the smiles is not found in pubchem then we can stop here
    except HTTPError as error:
        if error.code == 404:
            print(f' Smiles {smiles} not found')
            return None
        else:
            raise ValueError('Something went wrong with the PubChem request: ' + request_url)
    # Get the PubChem id
    compounds = parsed_response.get('PC_Compounds', None)
    if not compounds:
        raise RuntimeError(f'Something went wrong when mining pubchem data for SMILES {smiles}')
    # Make sure there is only 1 compound
    # DANI: Esto algún día podría no ser así, ya lo gestionaremos
    if len(compounds) != 1:
        raise RuntimeError(f'There should be one and only one compound for SMILES {smiles}')
    # Keep mining
    compound = compounds[0]
    first_id = compound.get('id', None)
    # If there is no first id then it means there is no direct match with pubchem
    if not first_id:
        return None
    second_id = first_id.get('id', None)
    if not second_id:
        raise RuntimeError(f'Missing second id when mining pubchem data for SMILES {smiles}')
    pubchem_id = second_id.get('cid', None)
    if not pubchem_id:
        raise RuntimeError(f'Missing pubchem id when mining pubchem data for SMILES {smiles}')
    return str(pubchem_id)


def get_pubchem_data(id_pubchem: str) -> Optional[str]:
    # Request PUBChem
    parsed_response = None
    request_url = Request(
        url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{id_pubchem}/JSON/',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    try:
        with urlopen(request_url) as response:
            # parsed_response = json.loads(response.read().decode("windows-1252"))
            parsed_response = json.loads(response.read().decode("utf-8", errors='ignore'))
    # If the accession is not found in PUBChem then the id is not valid
    except HTTPError as error:
        if error.code == 404:
            print('WARNING: Cannot find PUBChem entry for accession ', id_pubchem)
            return None
        print('Error when requesting ', request_url)
        raise ValueError('Something went wrong with the PUBChem request (error ', str(error.code), ')')
    # If we have not a response at this point then it may mean we are trying to access an obsolete entry (e.g. P01607)
    if parsed_response == None:
        print('WARNING: Cannot find PUBChem entry for accession ' + id_pubchem)
        return None
    # Mine target data: SMILES
    record = parsed_response.get('Record', None)
    if record == None:
        raise RuntimeError('Wrong Pubchem data structure: no record')
    sections = record.get('Section', None)
    if sections == None:
        raise RuntimeError('Wrong Pubchem data structure: no sections')
    names_and_ids_section = next(
        (section for section in sections if section.get('TOCHeading', None) == 'Names and Identifiers'), None)
    if names_and_ids_section == None:
        raise RuntimeError('Wrong Pubchem data structure: no name and ids section')
    names_and_ids_subsections = names_and_ids_section.get('Section', None)
    if names_and_ids_subsections == None:
        raise RuntimeError('Wrong Pubchem data structure: no name and ids subsections')

    # Mine the name
    synonims = next((s for s in names_and_ids_subsections if s.get('TOCHeading', None) == 'Synonyms'), None)
    if synonims == None:
        descriptors = next(
            (s for s in names_and_ids_subsections if s.get('TOCHeading', None) == 'Computed Descriptors'), None)
        descriptors_subsections = descriptors.get('Section', None)
        if descriptors_subsections == None:
            raise RuntimeError('Wrong Pubchem data structure: no name and ids subsections')
        depositor_supplied_descriptors = next(
            (s for s in descriptors_subsections if s.get('TOCHeading', None) == 'IUPAC Name'), None)
        name_substance = \
        depositor_supplied_descriptors.get('Information', None)[0].get('Value', {}).get('StringWithMarkup', None)[
            0].get('String', None)
    else:
        synonims_subsections = synonims.get('Section', None)
        if synonims_subsections == None:
            raise RuntimeError('Wrong Pubchem data structure: no name and ids subsections')
        depositor_supplied_synonims = next(
            (s for s in synonims_subsections if s.get('TOCHeading', None) == 'Depositor-Supplied Synonyms'), None)
        name_substance = \
        depositor_supplied_synonims.get('Information', None)[0].get('Value', {}).get('StringWithMarkup', None)[0].get(
            'String', None)

    # Mine the SMILES
    computed_descriptors_subsection = next(
        (s for s in names_and_ids_subsections if s.get('TOCHeading', None) == 'Computed Descriptors'), None)
    if computed_descriptors_subsection == None:
        raise RuntimeError('Wrong Pubchem data structure: no computeed descriptors')
    canonical_smiles_section = computed_descriptors_subsection.get('Section', None)
    if canonical_smiles_section == None:
        raise RuntimeError('Wrong Pubchem data structure: no canonical SMILES section')
    canonical_smiles = next((s for s in canonical_smiles_section if s.get('TOCHeading', None) == 'Canonical SMILES'),
                            None)
    if canonical_smiles == None:
        raise RuntimeError('Wrong Pubchem data structure: no canonical SMILES')
    smiles = canonical_smiles.get('Information', None)[0].get('Value', {}).get('StringWithMarkup', None)[0].get(
        'String', None)
    if smiles == None:
        raise RuntimeError('Wrong Pubchem data structure: no SMILES')

    # Mine target data: MOLECULAR FORMULA
    molecular_formula_subsection = next(
        (s for s in names_and_ids_subsections if s.get('TOCHeading', None) == 'Molecular Formula'), None)
    if molecular_formula_subsection == None:
        raise RuntimeError('Wrong Pubchem data structure: no molecular formula section')
    molecular_formula = \
    molecular_formula_subsection.get('Information', None)[0].get('Value', {}).get('StringWithMarkup', None)[0].get(
        'String', None)
    if molecular_formula == None:
        raise RuntimeError('Wrong Pubchem data structure: no molecular formula')

    # Mine target data: ChEMBL ID / DrugBank ID
    other_identifiers_subsection = next(
        (s for s in names_and_ids_subsections if s.get('TOCHeading', None) == 'Other Identifiers'), None)
    if other_identifiers_subsection == None:
        raise RuntimeError('Wrong Pubchem data structure: no molecular formula section')
    other_identifiers_section = other_identifiers_subsection.get('Section', None)
    drugbank_subsection = next(
        (s for s in other_identifiers_section if s.get('TOCHeading', None) == 'DrugBank ID'), None)
    if drugbank_subsection is None:
        drugbank_id = None
    else:
        drugbank_id = drugbank_subsection.get('Information', None)[0].get('Value', {}).get('StringWithMarkup', None)[0].get(
        'String', None)

    other_identifiers_subsection = next(
        (s for s in names_and_ids_subsections if s.get('TOCHeading', None) == 'Other Identifiers'), None)
    if other_identifiers_subsection == None:
        raise RuntimeError('Wrong Pubchem data structure: no molecular formula section')
    other_identifiers_section = other_identifiers_subsection.get('Section', None)
    chembl_subsection = next(
        (s for s in other_identifiers_section if s.get('TOCHeading', None) == 'ChEMBL ID'), None)
    if chembl_subsection is None:
        chembl_id = None
    else:
        chembl_id = chembl_subsection.get('Information', None)[0].get('Value', {}).get('StringWithMarkup', None)[0].get(
            'String', None)

    return name_substance, smiles, molecular_formula, drugbank_id, chembl_id


if __name__ == '__main__':
    print(get_pubchem_data(smiles_to_pubchem_id("CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5")))
