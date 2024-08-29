import requests
import json
import re
from bs4 import BeautifulSoup
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode


def find_drugbank_pubchem(drugbank_id):
    # Request Drugbank
    request_url = Request(
        url=f'https://go.drugbank.com/drugs/{drugbank_id}',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    pubchem_id = None
    try:
        with urlopen(request_url) as response:
            content = response.read().decode("utf-8")
            print(str(content))
            pattern = re.compile("http\:\/\/pubchem.ncbi.nlm.nih.gov\/summary\/summary.cgi\?cid\=([0-9]*)")
            match = re.search(pattern, str(content))
            if not match:
                raise ValueError("No se encontró información sobre el Pubchem compound en esta página.")
            pubchem_id = match[1]
            if not pubchem_id:
                raise ValueError("No se encontró información sobre el Pubchem compound en esta página.")
    # If the accession is not found in the database then we stop here
    except HTTPError as error:
        # If the drugbank ID is not yet in the Drugbank references then return None
        raise ValueError(f'Wrong request. Code: {error.code}')
    # This error may occur if there is no internet connection
    except URLError as error:
        print('Error when requesting ' + request_url)
        raise ValueError('Something went wrong with the DrugBank request')

    return pubchem_id


def find_drugbank_groups(drugbank_id):
    # Request Drugbank
    request_url = Request(
        url=f'https://go.drugbank.com/drugs/{drugbank_id}',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    groups = None
    try:
        with urlopen(request_url) as response:
            content = response.read().decode("utf-8")

            pattern = r'<dt[^>]*id="groups"[^>]*>Groups<\/dt>\s*<dd[^>]*>([^<]+)'
            match = re.search(pattern, str(content))
            if not match:
                raise ValueError("No se encontró información sobre el Pubchem compound en esta página.")
            groups = match[1]
            if not groups:
                raise ValueError("No se encontró información sobre el Pubchem compound en esta página.")
    # If the accession is not found in the database then we stop here
    except HTTPError as error:
        # If the drugbank ID is not yet in the Drugbank references then return None
        raise ValueError(f'Wrong request. Code: {error.code}')
    # This error may occur if there is no internet connection
    except URLError as error:
        print('Error when requesting ' + request_url)
        raise ValueError('Something went wrong with the DrugBank request')

    return groups


if __name__ == '__main__':
    print(find_drugbank_groups('DB05000'))