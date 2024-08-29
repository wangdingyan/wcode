# https://github.com/drorlab/combind/blob/be0f5bbf3d141a30c441345e7670a1481cf1941f/chembl/chembl.py#L35

import sqlite3
import pandas as pd
import numpy as np


# Macrocycles are hard to dock
MACROCYCLE_THRESH = 8

class CHEMBLDB:
    def __init__(self, chembldb):
        # CHEMBL database was downloaded from the chembl website in sql format.
        self.conn = sqlite3.connect('file:{}?mode=ro'.format(chembldb), uri=True)
        self.cur = self.conn.cursor()

    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        self.conn.close()

    def chembl_to_tid(self, chembl):
        self.cur.execute("SELECT tid FROM target_dictionary WHERE chembl_id=?", (chembl,))
        rows = self.cur.fetchall()
        assert len(rows) == 1, rows
        return rows[0][0]
    def tid_to_target_type(self, tid):
        self.cur.execute("SELECT target_type FROM target_dictionary WHERE tid=?", (tid,))
        rows = self.cur.fetchall()
        assert len(rows) == 1, rows
        return rows[0][0]
    def tid_to_assays(self, tid, protein_complex, homologous):
        if protein_complex and homologous:
            confidence = '(confidence_score=6 OR confidence_score=7)'
        elif protein_complex and not homologous:
            confidence = 'confidence_score=7'
        elif not protein_complex and homologous:
            confidence = '(confidence_score=8 OR confidence_score=9)'
        else:
            confidence = 'confidence_score=9'

        self.cur.execute("SELECT assay_id FROM assays WHERE tid=? AND "+confidence, (tid,))
        return [row[0] for row in self.cur.fetchall()]
    def assay_to_chemblid(self, assay):
        self.cur.execute("SELECT chembl_id FROM assays WHERE assay_id=?", (assay,))
        rows = self.cur.fetchall()
        assert len(rows) == 1, rows
        return rows[0][0]
    def assay_to_molregnos(self, assay):
        self.cur.execute("SELECT molregno FROM activities WHERE assay_id=?", (assay,))
        return [row[0] for row in self.cur.fetchall()]
    def molregno_to_molw(self, molregno):
        self.cur.execute("SELECT mw_freebase FROM compound_properties WHERE molregno=?", (molregno,))
        rows = self.cur.fetchall()
        if not len(rows):
            return 0
        return rows[0][0]
    def molregno_to_smiles(self, molregno):
        self.cur.execute("SELECT canonical_smiles FROM compound_structures WHERE molregno=?", (molregno,))
        rows = self.cur.fetchall()
        if not rows:
            return None
        return rows[0][0]
    def molregno_to_chemblid(self, molregno):
        self.cur.execute("SELECT chembl_id FROM molecule_dictionary WHERE molregno=?", (molregno,))
        rows = self.cur.fetchall()
        assert len(rows) == 1, rows
        return rows[0][0]
    def molregno_and_assay_to_activities(self, molregno, assay):
        self.cur.execute("SELECT standard_type, standard_value, standard_units, relation, activity_comment FROM activities WHERE molregno=? AND assay_id=?", (molregno, assay))
        return self.cur.fetchall()
    def chembl_to_activities(self, chembl, protein_complex, homologous):
        # chemblID, SMILES, MOLW, affinity
        activities = []
        tid = self.chembl_to_tid(chembl)
        for assay in self.tid_to_assays(tid, protein_complex, homologous):
            assay_chembl_id = self.assay_to_chemblid(assay)
            for molregno in self.assay_to_molregnos(assay):
                molw = self.molregno_to_molw(molregno)
                smiles = self.molregno_to_smiles(molregno)
                chembl_id = self.molregno_to_chemblid(molregno)
                for activity in self.molregno_and_assay_to_activities(molregno, assay):
                    activities += [[assay_chembl_id, chembl_id, molw, smiles] + list(activity)]
        return pd.DataFrame(activities,
                            columns=['assay_chembl_id', 'ligand_chembl_id', 'mw_freebase',
                                     'canonical_smiles', 'standard_type', 'standard_value',
                                     'standard_units', 'relation', 'comment'])
    def parse_database3(self):
        self.cur.execute('''
            SELECT md.chembl_id, cseq.accession, act.pchembl_value
            FROM molecule_dictionary md, activities act, assays ass, component_sequences cseq, target_components t, target_dictionary td
            WHERE
                act.molregno = md.molregno AND
                act.assay_id = ass.assay_id AND
                act.standard_relation = '=' AND
                act.standard_flag = 1 AND
                ass.assay_type = 'B' AND
                ass.tid = t.tid AND
                ass.tid = td.tid AND
                td.target_type = 'SINGLE PROTEIN' AND
                t.component_id = cseq.component_id AND
                cseq.accession IS NOT NULL AND
                act.pchembl_value >= 5
        ''')
        return self.cur.fetchall()

################################################################################

def standardize_nonbinders(activities,
                           affinity_thresh):
    # Set 'Not Active's to affinity_thresh
    activities.loc[activities['comment'].isin([None]), 'comment'] = ''
    duds = [('Not Active' in s) for s in activities['comment']]
    duds = np.array(duds)
    activities.loc[duds, 'standard_units'] = 'nM'
    activities.loc[duds, 'standard_value'] = affinity_thresh
    activities.loc[duds, 'relation'] = '='

    # Most nonbinders don't have equality relation.
    mask  = activities['standard_value'] >= affinity_thresh
    mask *= activities['relation'].isin(['>', '>='])
    activities.loc[mask, 'relation'] = '='

    # Cap affinity values.
    mask  = activities['standard_value'] >= affinity_thresh
    activities.loc[mask, 'standard_value'] = affinity_thresh
    return activities
def get_activities(chembl,
                   chembldb,
                   # uniprot_chembl,
                   protein_complex,
                   homologous,
                   affinity_thresh):

    # with CHEMBLDB(chembldb, uniprot_chembl) as chembldb:
    with CHEMBLDB(chembldb) as chembldb:
        activities = chembldb.chembl_to_activities(chembl, protein_complex, homologous)
    activities['target_chembl_id'] = chembl

    # Standardize units to nM
    m = {'M': 10**9, 'mM': 10**6, 'uM': 10**3, 'pM': 10**-3}
    for unit, relation in m.items():
        mask = activities['standard_units'] == unit
        activities.loc[mask, 'standard_value'] *= relation
        activities.loc[mask, 'standard_units'] = 'nM'

    activities = standardize_nonbinders(activities, affinity_thresh)

    return activities

########################     Filter Activities     ###########################

def filter_activities(activities, activity_type, molw_thresh):
    if activity_type == 'all':
        activity_types = ['IC50', 'Ki', 'Kd']
    else:
        activity_types = [activity_type]
    mask = activities['standard_type'].isin(activity_types)
    print('Removing {} rows b/c standard_type not in {}'.format(len(mask)-sum(mask),
                                                                 activity_types))
    print('Set of offending values is {}'.format(set(activities[~mask]['standard_type'])))
    activities = activities.loc[mask]

    mask = activities['standard_value'].notna()
    print('Removing {} rows b/c standard_value is na'.format(len(mask)-sum(mask)))
    activities = activities.loc[mask]

    mask = activities['standard_value'] != 0
    print('Removing {} rows b/c standard_value is 0'.format(len(mask)-sum(mask)))
    activities = activities.loc[mask]

    mask = ~activities['canonical_smiles'].isin([None])
    print('Removing {} rows b/c canonical_smiles is None'.format(len(mask)-sum(mask)))
    activities = activities.loc[mask]

    mask = activities['mw_freebase'] <= molw_thresh+100 # Not desalted yet.
    print('Removing {} rows b/c mw_freebase > {}'.format(len(mask)-sum(mask),
                                                         molw_thresh+100))
    activities = activities.loc[mask]

    mask = activities['standard_units'] == 'nM'
    print('Removing {} rows b/c standard_units != nM'.format(len(mask)-sum(mask)))
    print('Set of offending values is {}'.format(set(activities[~mask]['standard_units'])))
    activities = activities.loc[mask]

    mask = activities['relation'] == '='
    print('Removing {} rows b/c relation != ='.format(len(mask)-sum(mask)))
    print('Set of offending values is {}'.format(set(activities[~mask]['relation'])))
    activities = activities.loc[mask]

    return activities


def filter_properties(activities, ambiguous_stereo, molw_thresh):
    mask = activities['MOLW'] <= molw_thresh
    print('Removing {} rows b/c molw > {}'.format(len(mask)-sum(mask),
                                                  molw_thresh))
    activities = activities.loc[mask]

    mask = ~activities['MACROCYCLE']
    print('Removing {} rows b/c macrocycle'.format(len(mask)-sum(mask)))
    activities = activities.loc[mask]

    if not ambiguous_stereo:
        mask = activities['STEREO']
        print('Removing {} rows b/c ambiguous stereochemistry'.format(len(mask)-sum(mask)))
        activities = activities.loc[mask]
    return activities

def collapse_duplicates(activities, seperate_activity_types):
    if seperate_activity_types:
        keys = ['SMILES', 'standard_type']
    else:
        keys = ['SMILES']

    averages = activities.loc[:, keys+['standard_value']].groupby(keys).mean()
    activities = activities.groupby(keys, as_index=False).first()
    activities['standard_value'] = [averages.loc[tuple([row[key] for key in keys])]['standard_value']
                                    for _, row in activities.iterrows()]
    return activities

################################################################################


if __name__ == '__main__':
    # with CHEMBLDB("C:\\database\\chembl_33\\chembl_33_sqlite\\chembl_33.db") as db:
        # df = db.chembl_to_activities('CHEMBL4036',
        #                               False,
        #                               False)
        # print(df.loc[df['comment'].isin([None]), 'comment'])
        df = get_activities('CHEMBL224',
                             "C:\\database\\chembl_33\\chembl_33_sqlite\\chembl_33.db",
                             False,
                             False,
                             1000000)
        df = filter_activities(df, activity_type='all', molw_thresh=700)
        df.to_excel('C:\\tmp\\CHEMBL224.xlsx')

