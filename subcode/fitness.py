# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:42 2022

@author: 86136
"""
from property import *
from nonDominationSort import *
import torch
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import calc_no as dock
from rdkit.Chem import Descriptors




"""
calculate fitness and CV
"""
def fitness_qed(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维 
    nPop = len(mol)
    fits = np.array([ff_qed(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits 

def ff_qed(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, sim#pen_logP,



def fitness_plogp(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_plogp(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_plogp(seq, mol, fp_0):
    pen_logP = penalized_logP(mol)#需改为种群
    #qed = QED(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return pen_logP, sim#pen_logP,


#docking
def fitness_4lde(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)

    fits = np.array([ff_4lde(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_4lde(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    sim = tanimoto_similarity(mol, fp_0)
    lde4 = -cal_4lde(seq)#返回正数，越大越好
    print('lde4:',lde4)
    return qed, sim, lde4#pen_logP,

def cal_4lde(seq):
    mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return 10 ** 4
    else:
        lde4 = dock.perform_calc_single(seq, '4lde', docking_program='qvina')
        return lde4

##多目标QED,SIM,Plogp
def fitness_qedlogp(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_qedlogp(seq[i], mol[i], fp_0) for i in range(nPop)])

    return fits

def ff_qedlogp(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    pen_logP = penalized_logP(mol)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, pen_logP, sim#pen_logP,

#多目标qed,drd2,sim
def fitness_qeddrd(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qeddrd(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qeddrd(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)  # 需改为种群
    drd = drd2(seq)
    #sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, sim, drd  # pen_logP

#多目标qed,gskb,sa_nom,sim
def fitness_qedgsksa(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qedgsksa(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qedgsksa(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)  # 需改为种群
    gskb = gsk(seq)
    sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, gskb, sa_nom, sim  # pen_logP


#计算约束违反度
def CV(mol, c):
    # 计算种群的违反度
    CV =  np.array([cv(x,c) for x in mol])#CV
    #print(CV)
    if c==1:
        return CV
    else:
        #自适应归一化
        #max_cv1 = max(CV[:,0])
        #max_cv1 = max([max_cv1,1])
        #max_cv2 = max(CV[:, 1])
        #max_cv2 = max([max_cv2, 1])
        #CV_all = CV[:, 0]/max_cv1+CV[:, 1]/max_cv2
        '''
        全局归一化
        '''
        CV_all = CV[:, 0] / 45 + CV[:, 1] / 5
        return CV_all

#calculate the violation of molecules
def CV_globel(mol, c):
    CV =  np.array([cv(x,c) for x in mol])#CV
    #print(CV)
    if c==1:
        return CV
    else:
        max_cv1 = max(CV[:,0])
        max_cv2 = max(CV[:, 1])
        return max_cv1, max_cv2
