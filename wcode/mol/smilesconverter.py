import copy
import json
from rdkit import Chem
from rdkit.Chem import AllChem, MolStandardize, Mol
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import quote


########################################################################################################################


"""Standardize molecule and convert between identifier."""


class SmilesConverter():
    """Converter class."""

    def __init__(self):
        """Initialize a Converter instance."""
        try:
            import rdkit.Chem as Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            self.Chem = Chem
            self.scaffold = MurckoScaffold
        except ImportError:
            raise ImportError("requires rdkit " +
                              "https://www.rdkit.org/")
        try:
            from chembl_structure_pipeline.standardizer import standardize_mol
            self.standardize = standardize_mol
        except ImportError:
            raise ImportError("requires chembl_structure_pipeline")
        try:
            import pubchempy as pcp
            self.pcp = pcp
        except ImportError:
            raise ImportError("requires pubchempy")

    def canonicalize_smiles(self, smiles, removechiral=False, kekulize=False):
        if type(smiles) is not str:
            print("Input is not String")
            return
        if len(smiles) == 0:
            return ''
        if removechiral:
            smiles = smiles.replace('@', '')
        mol = Chem.MolFromSmiles(smiles)
        lfc = MolStandardize.fragment.LargestFragmentChooser()

        if mol is not None:
            mol2 = lfc.choose(mol)
            smi2 = Chem.MolToSmiles(mol2, isomericSmiles=True, kekuleSmiles=kekulize)
            smi, _ = NeutraliseCharges(smi2)
            return smi
        else:
            return ''

    def smiles_to_scaffold(self, smiles, generic=False):
        """From SMILES to the SMILES of its scaffold."""
        scaffold_smiles = self.scaffold.MurckoScaffoldSmiles(smiles)
        if generic:
            scaffold_mol = self.scaffold.MakeScaffoldGeneric(
                self.Chem.MolFromSmiles(scaffold_smiles))
            scaffold_smiles = self.Chem.MolToSmiles(scaffold_mol)
        return scaffold_smiles

    def smiles_to_inchi(self, smiles):
        """From SMILES to InChIKey and InChI."""
        mol = self.Chem.MolFromSmiles(smiles)
        if not mol:
            raise ConversionError("MolFromSmiles returned None", smiles)
        try:
            mol = self.standardize(mol)
        except Exception as ex:
            raise ConversionError("'standardize' exception:", smiles)
        inchi = self.Chem.rdinchi.MolToInchi(mol)[0]
        if not inchi:
            raise ConversionError("'MolToInchi' returned None.", smiles)
        inchikey = self.Chem.rdinchi.InchiToInchiKey(inchi)
        if not inchikey:
            raise ConversionError("'InchiToInchiKey' returned None", smiles)
        try:
            mol = self.Chem.rdinchi.InchiToMol(inchi)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", smiles)
        return inchikey, inchi

    def inchi_to_smiles(self, inchi):
        """From InChI to SMILES."""
        try:
            inchi_ascii = inchi.encode('ascii', 'ignore')
            mol = self.Chem.rdinchi.InchiToMol(inchi_ascii)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", inchi)
        try:
            mol = self.standardize(mol)
        except Exception as ex:
            raise ConversionError("'standardize' exception:", inchi)
        return self.Chem.MolToSmiles(mol, isomericSmiles=True)

    def inchi_to_inchikey(self, inchi):
        """From InChI to InChIKey."""
        try:
            inchi_ascii = inchi.encode('ascii', 'ignore')
            inchikey = self.Chem.rdinchi.InchiToInchiKey(inchi_ascii)
        except Exception as ex:
            raise ConversionError("'InchiToInchiKey' exception:", inchi)
        return inchikey

    def inchi_to_mol(self, inchi):
        """From InChI to molecule."""
        try:
            inchi_ascii = inchi.encode("ascii", "ignore")
            mol = self.Chem.rdinchi.InchiToMol(inchi_ascii)[0]
        except Exception as ex:
            raise ConversionError("'InchiToMol' exception:", inchi)
        try:
            mol = self.standardize(mol)
        except Exception as ex:
            raise ConversionError("'standardize' exception:", inchi)
        return mol

    @staticmethod
    def ctd_to_smiles(ctdid):
        """From CTD identifier to SMILES."""
        # convert to pubchemcid
        try:
            url = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/' + \
                'sourceid/Comparative%20Toxicogenomics%20Database/' + \
                ctdid + '/cids/TXT/'
            pubchemcid = urlopen(url).read().rstrip().decode()
        except Exception as ex:
            SmilesConverter.__log.warning(str(ex))
            raise ConversionError("Cannot fetch PubChemID CID from CTD", ctdid)
        # get smiles
        try:
            url = 'http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/' + \
                'cid/%s/property/CanonicalSMILES/TXT/' % pubchemcid
            smiles = urlopen(url).read().rstrip().decode()
        except Exception as ex:
            SmilesConverter.__log.warning(str(ex))
            raise ConversionError(
                "Cannot fetch SMILES from PubChemID CID", pubchemcid)
        return smiles

    @staticmethod
    def chemical_name_to_smiles(chem_name):
        """From Chemical Name to SMILES via cactus.nci or pubchem."""
        smiles = None
        chem_name_quoted = quote(chem_name)
        smiles = SmilesConverter._chemical_name_to_smiles_cactus(chem_name_quoted)
        if smiles is not None:
            return smiles
        smiles = SmilesConverter._chemical_name_to_smiles_pubchem(chem_name)
        if smiles is None:
            raise ConversionError(
                "Cannot fetch SMILES from Chemical Name", chem_name)
        return smiles

    @staticmethod
    def chemical_name_to_inchi(chem_name):
        """From Chemical Name to InChI via cactus.nci or pubchem."""
        inchi = None
        chem_name_quoted = quote(chem_name)
        inchi = SmilesConverter._chemical_name_to_inchi_cactus(chem_name_quoted)
        if inchi is not None:
            return inchi
        inchi = SmilesConverter._chemical_name_to_inchi_pubchem(chem_name)
        if inchi is None:
            raise ConversionError(
                "Cannot fetch InChI from Chemical Name", chem_name)
        return inchi

    @staticmethod
    def inchikey_to_inchi(inchikey):
        """From InChIKey to InChI.

        Precedence is given to the local db that will be the fastest option.
        If it is not found locally several provider are contacted, and we
        possibly want to add the it to the Molecule table.
        """
        # if local_db:
        #     from lgdrugai.database.molecule import Molecule
        #     res = Molecule.get_inchikey_inchi_mapping([inchikey])
        #     if res[inchikey] is not None:
        #         return res[inchikey]

        resolve_fns = {
            'unichem': SmilesConverter._resove_inchikey_unichem,
            'cactus': SmilesConverter._resove_inchikey_cactus,
            'pubchem': SmilesConverter._resove_inchikey_pubchem,
        }
        inchi = None
        for provider, func in resolve_fns.items():
            try:
                inchi = func(inchikey)
                break
            except:
                SmilesConverter.__log.debug(
                    'InChIKey %s not found via %s' % (inchikey, provider))
                continue
        if inchi is None:
            raise ConversionError('Unable to resolve', inchikey)

        return inchi

########################################################################################################################

    @staticmethod
    def _chemical_name_to_smiles_cactus(chem_name):
        """From chemical name to SMILES."""
        try:
            url = 'http://cactus.nci.nih.gov/chemical/' + \
                'structure/%s/smiles' % chem_name
            smiles = urlopen(url).read().rstrip().decode()
            return smiles
        except Exception as ex:
            SmilesConverter.__log.warning(
                "Cannot convert Chemical Name "
                "to SMILES (cactus.nci): %s" % chem_name)
            return None

    @staticmethod
    def _chemical_name_to_inchi_cactus(chem_name):
        """From chemical name to InChI."""
        try:
            url = 'http://cactus.nci.nih.gov/chemical/' + \
                'structure/%s/stdinchi' % chem_name
            inchi = urlopen(url).read().rstrip().decode()
            return inchi
        except Exception as ex:
            SmilesConverter.__log.warning(
                "Cannot convert Chemical Name "
                "to InChI (cactus.nci): %s" % chem_name)
            return None


    @staticmethod
    def _resove_inchikey_unichem(inchikey):
        try:
            inchikey = quote(inchikey)
            url = 'https://www.ebi.ac.uk/unichem/rest/inchi/%s' % inchikey
            res = json.loads(urlopen(url).read().rstrip().decode())
        except Exception as ex:
            # Converter.__log.warning(str(ex))
            raise ConversionError(
                "No response from unichem: %s" % url, inchikey)

        if isinstance(res, dict):
            err_msg = '; '.join(['%s: %s' % (k, v)
                                 for k, v in res.items()])
            raise ConversionError(err_msg, inchikey)
        elif isinstance(res, list):
            if len(res) != 1:
                raise ConversionError(
                    'No results from unichem: %s' % str(res), inchikey)
            if 'standardinchi' not in res[0]:
                raise ConversionError(
                    'No results from unichem: %s' % str(res), inchikey)
            inchi = res[0]['standardinchi']
            return inchi

    @staticmethod
    def _resove_inchikey_cactus(inchikey):
        try:
            inchikey = quote(inchikey)
            url = ("https://cactus.nci.nih.gov/"
                   "chemical/structure/%s/stdinchi" % inchikey)
            res = urlopen(url).read().rstrip().decode()
            return res
        except Exception as ex:
            # Converter.__log.warning(str(ex))
            raise ConversionError(
                "No response from cactus: %s" % url, inchikey)


def _InitialiseNeutralisationReactions():
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


_reactions = None


def NeutraliseCharges(smiles, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions = _InitialiseNeutralisationReactions()
        reactions = _reactions
    mol = Chem.MolFromSmiles(smiles)
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return (Chem.MolToSmiles(mol, True), True)
    else:
        return (smiles, False)


def neutralize_atoms(mol):
    """
    https://www.rdkit.org/docs/Cookbook.html
    :param mol:
    :return:
    """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol



def standardize_mol(mol: Chem.Mol) -> Chem.Mol:
    mol = copy.deepcopy(mol)

    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol, sanitize=True)
    if mol is None:
        raise ValueError("Failed to standardize molecule.")

    return mol


class ConversionError(Exception):
    """Conversion error."""

    def __init__(self, message, idx):
        """Initialize a ConversionError."""
        message = "Cannot convert: %s Message: %s" % (idx, message)
        super(Exception, self).__init__(message)
