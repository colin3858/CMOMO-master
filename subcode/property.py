# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:45:28 2022

@author: 86136
"""

from functools import lru_cache
import torch
import numpy as np
import pandas as pd
import rdkit
from rdkit.Chem import AllChem
#from DRD2.DRD2_predictor2 import *
#from drd.drd_model import *
from tdc import Oracle
from rdkit.Chem import Descriptors

from rdkit import Chem, DataStructs
    
def morgan_fingerprint(mol):
    """Molecular fingerprint using Morgan algorithm.

    Uses ``radius=2, nBits=2048``.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the fingerprint.

    Returns:
        np.ndarray: Fingerprint vector.
    """
    #mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def tanimoto_similarity(seq, fp_0):
    """Tanimoto similarity between two molecules.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.
        fp_0 (array): Fingerprint vector of original molecule.

    Returns:
        float: Tanimoto similarity.
    """
    #mol = Chem.MolFromSmiles(seq)
    fp = morgan_fingerprint(seq)
    if fp is None:
        return 0
    return rdkit.DataStructs.TanimotoSimilarity(fp_0, fp)


def sim_2(mol1, mol2):
    """Tanimoto similarity between two molecules.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.
        fp_0 (array): Fingerprint vector of original molecule.

    Returns:
        float: Tanimoto similarity.
    """
    #mol = Chem.MolFromSmiles(mol)
    fp_1 = morgan_fingerprint(mol1)
    fp_2 = morgan_fingerprint(mol2)
    if fp_1 is None:
        return 0
    return rdkit.DataStructs.TanimotoSimilarity(fp_1, fp_2)



qed_ = Oracle('qed')
sa_ = Oracle('sa')
jnk_ = Oracle('JNK3')
gsk_ = Oracle('GSK3B')
logp_ = Oracle('logp')
drd2_ = Oracle('drd2')
fexofenadine_ = Oracle('fexofenadine_mpo')
osimertinib_ = Oracle('osimertinib_mpo')

def QED(mol):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    #mol = Chem.MolFromSmiles(mol)
    try:
        return qed_(mol)
    except:
        return np.nan

def penalized_logP(mol):
    """Penalized logP.

    Computed as logP(mol) - SA(mol) as in JT-VAE.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: Penalized logP or NaN if mol is None.
    """
    # mol = Chem.MolFromSmiles(mol)
    try:
        return logp_(mol) - sa_(mol)
    except:
        return np.nan

def normalize_sa(smiles):
    try:
        sa_score = sa_(smiles)
        normalized_sa = (10. - sa_score) / 9.
        return normalized_sa
    except:
        return np.nan


def gsk(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return gsk_(smi)
    except:
        return np.nan

def drd2(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return drd2_(smi)
    except:
        return np.nan

smarts = pd.read_csv(('./data/archive/sure_chembl_alerts.txt'), header=None, sep='\t')[1].tolist()
alert_mols = [Chem.MolFromSmarts(smart) for smart in smarts if Chem.MolFromSmarts(smart) is not None]
def cv(mol, c):
    try:
        ri = mol.GetRingInfo()#分子中的环
        len_ring = [len(r) for r in ri.AtomRings()]#多个环的原子数
        #计算环原子数的违反度
        if len(len_ring)==0:
            cv_ring = 0
        else:
            cv_ring = [max([x-6,0])+max([5-x,0]) for x in len_ring]
            cv_ring = sum(cv_ring)
        #计算分子量的违反度
        if c==2:
            #wt = Descriptors.MolWt(mol)  # 分子量
            #cv_wt = max([wt-500,0])
            ls_tox = [mol.HasSubstructMatch(alert) for alert in alert_mols]
            ls_tox = np.array(ls_tox)
            cv_tox = np.sum(ls_tox != 0)
            return cv_ring, cv_tox
        else:
            return cv_ring
    except:
        if c==2:
            return 100, 10000
        else:
            return 100

def fexo(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return fexofenadine_(smi)
    except:
        return np.nan

def osim(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return osimertinib_(smi)
    except:
        return np.nan

'''
smi='COc1cccc(O[C@@H]2CC[C@H]([NH3+])C2)n1'
smi2 = 'COc1ccc(C(C)=O)c(OCC(=O)N2[C@@H](C)CCC[C@H]2C)c1'
print(qed_(smi))
print(logp_(smi))
mol_0 = Chem.MolFromSmiles(smi)
print(penalized_logP(mol_0))

print(logp_(smi2))
mol_2 = Chem.MolFromSmiles(smi2)
print(penalized_logP(mol_2))

print(jnk(smi))
print(gsk(smi))
print(drd2(smi))
print(fexo(smi))
print(osim(smi))
'''

'''
smi = 'CN1C(=O)CC(N2c3ccccc3SC(c3ccco3)C2c2ccco2)c2ccccc21'#,'CN1C(=O)CC(N2c3ccccc3SC(c3ccco3)C2c2ccco2)c2ccccc21']
test_mol = Chem.MolFromSmiles(smi)
HA = test_mol.GetNumHeavyAtoms()
print(HA)
#cv_mol = cv(test_mol,2)
#print(cv_mol)
'''





