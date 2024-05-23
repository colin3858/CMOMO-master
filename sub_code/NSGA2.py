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
种群初始化 
"""
#def initPops(seqs):
#    pops = model.encode(seqs)
#    return pops
"""
选择算子 
"""
def select1(pool, pops, fits, ranks, distances):
    # 一对一锦标赛选择 
    # pool: 新生成的种群大小 
    nPop, nChr = pops.shape 
    nF = fits.shape[1] 
    newPops = np.zeros((pool, nChr)) 
    newFits = np.zeros((pool, nF))  

    indices = np.arange(nPop).tolist()
    i = 0 
    while i < pool: 
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体 
        idx = compare(idx1, idx2, ranks, distances)

        newPops[i] = pops[idx] 
        newFits[i] = fits[idx] 
        i += 1 
    return newPops, newFits

def select1_unique(pool, pops, fits, ranks, distances):
    # 一对一锦标赛选择
    # pool: 新生成的种群大小
    fits_uni, indices = np.unique(fits, axis=0, return_index=True)
    pops_uni = pops[indices]
    nPop, nChr = pops_uni.shape
    nF = fits_uni.shape[1]
    newPops = np.zeros((pool, nChr))
    newFits = np.zeros((pool, nF))

    indices = np.arange(nPop).tolist()
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体
        idx = compare(idx1, idx2, ranks, distances)

        newPops[i] = pops_uni[idx]
        newFits[i] = fits_uni[idx]
        i += 1
    return newPops, newFits

def compare(idx1, idx2, ranks, distances): 
    # return: 更优的 idx 
    if ranks[idx1] < ranks[idx2]: 
        idx = idx1 
    elif ranks[idx1] > ranks[idx2]:
        idx = idx2 
    else:
        if distances[idx1] <= distances[idx2]:
            idx = idx2 
        else:
            idx = idx1 
    return idx  
"""交叉算子 
混合线性交叉 
"""

def crossover(pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    chrPops = pops
    nPop = chrPops.shape[0]
    for i in range(0, nPop): 
        if np.random.rand() < pc: 
            mother = chrPops[np.random.randint(nPop)]
            alpha1=np.random.rand()#生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1=(-d)+(1+2*d)*alpha1
            #r2=(-d)+(1+2*d)*alpha2
            chrPops[i] = chrPops[i]+r1*(mother-chrPops[i])#混合线性交叉
            chrPops[i][chrPops[i]<lb] = lb 
            chrPops[i][chrPops[i]>rb] = rb 
    return chrPops

def crossover_2(pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = pops.shape[0]
    chrPops = np.zeros((nPop * 2, pops.shape[1]))
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother1 = pops[np.random.randint(nPop)]
            mother2 = pops[np.random.randint(nPop)]
            [alpha1, alpha2] = np.random.rand(2)  # 生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1 = (-d) + (1 + 2 * d) * alpha1
            r2 = (-d) + (1 + 2 * d) * alpha2
            chrPops[2 * i] = pops[i]+r1*(mother1-pops[i])  # 混合线性交叉
            chrPops[2 * i + 1] = pops[i]+r2*(mother2-pops[i])  # 混合线性交叉
        chrPops[chrPops < lb] = lb
        chrPops[chrPops > rb] = rb
    return chrPops
'''
def crossover_4(pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = pops.shape[0]
    chrPops = np.zeros((nPop * 4, pops.shape[1]))
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother1 = pops[np.random.randint(nPop)]
            mother2 = pops[np.random.randint(nPop)]
            [alpha1, alpha2] = np.random.rand(2)  # 生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1 = (-d) + (1 + 2 * d) * alpha1
            r2 = (-d) + (1 + 2 * d) * alpha2
            chrPops[4 * i] = pops[i]+r1*(mother1-pops[i])  # 混合线性交叉
            chrPops[4 * i + 1] = pops[i]+r2*(mother1-pops[i])  # 混合线性交叉
            chrPops[4 * i + 2] = pops[i] + r1 * (mother2 - pops[i])  # 混合线性交叉
            chrPops[4 * i + 3] = pops[i] + r2 * (mother2 - pops[i])  # 混合线性交叉
        chrPops[chrPops < lb] = lb
        chrPops[chrPops > rb] = rb
    return chrPops
'''
def crossover_z0(z_0, archive1_emb, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    chrPops = archive1_emb
    nPop = chrPops.shape[0]
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = z_0
            alpha1=np.random.rand()#生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1=(-d)+(1+2*d)*alpha1
            #r2=(-d)+(1+2*d)*alpha2
            chrPops[i] = mother  + r1*(archive1_emb[i]-mother)#0.8*
    chrPops[i][chrPops[i]<lb] = lb
    chrPops[i][chrPops[i]>rb] = rb
    return chrPops

def crossover_z0_2(z_0, archive1_emb, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = archive1_emb.shape[0]
    chrPops = np.zeros((nPop*2, z_0.shape[1]))

    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = z_0
            [alpha1,alpha2]=np.random.rand(2)#生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1=(-d)+(1+2*d)*alpha1
            r2=(-d)+(1+2*d)*alpha2
            chrPops[2*i] = mother+0.8*r1*(archive1_emb[i]-mother)#混合线性交叉
            chrPops[2*i+1] = mother + 0.8*r2* (archive1_emb[i]-mother)  # 混合线性交叉
    chrPops[chrPops < lb] = lb
    chrPops[chrPops > rb] = rb
    return chrPops

def crossover_co_2(arc, pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = arc.shape[0]
    chrPops = np.zeros((nPop * 2, pops.shape[1]))
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother1 = pops[np.random.randint(nPop)]
            mother2 = pops[np.random.randint(nPop)]
            [alpha1, alpha2] = np.random.rand(2)  # 生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1 = (-d) + (1 + 2 * d) * alpha1
            r2 = (-d) + (1 + 2 * d) * alpha2
            chrPops[2 * i] = arc[i]+r1*(mother1-arc[i])  # 混合线性交叉
            chrPops[2 * i + 1] = arc[i]+r2*(mother2-arc[i])  # 混合线性交叉
        chrPops[chrPops < lb] = lb
        chrPops[chrPops > rb] = rb
    return chrPops
'''#离散片段交叉
def crossover(pops, pc, nChr, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    chrPops = pops.copy()
    nPop = chrPops.shape[0]
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = chrPops[np.random.randint(nPop)]
            pos = np.random.randint(0, nChr-1, 1)
            chrPops[i][pos[0]:] = mother[pos[0]:]
            chrPops[i][chrPops[i]<lb] = lb
            chrPops[i][chrPops[i]>rb] = rb
    return chrPops

#模拟二进制交叉
def crossover_SBX(pops, pc, etaC, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构
    """
        :param pc: the probabilities of doing crossover
        :param etaC: the distribution index of simulated binary crossover，设20
        lb：下界
        rb：上界
    """
    chrPops = pops.copy()
    nPop = chrPops.shape[0]
    #for i in range(0, nPop, 2):
    #    if np.random.rand() < pc:
    #        SBX(chrPops[i], chrPops[i+1], etaC, lb, rb)  # 交叉
    #return chrPops
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = chrPops[np.random.randint(nPop)]
            SBX(chrPops[i], mother, etaC, lb, rb)  # 交叉
    return chrPops

def SBX(chr1, chr2, etaC, lb, rb):

    # 模拟二进制交叉
    pos1, pos2 = np.sort(np.random.randint(0,len(chr1),2)) #随机产生两个位置
    pos2 += 1
    u = np.random.rand()
    if u <= 0.5:
        gamma = (2*u) ** (1/(etaC+1))
    else:
        gamma = (1/(2*(1-u))) ** (1/(etaC+1))
    x1 = chr1[pos1:pos2]
    x2 = chr2[pos1:pos2]
    chr1[pos1:pos2], chr2[pos1:pos2] = 0.5*((1+gamma)*x1+(1-gamma)*x2), \
        0.5*((1-gamma)*x1+(1+gamma)*x2)
    # 检查是否符合约束
    chr1[chr1<lb] = lb
    chr1[chr1>rb] = rb
    chr2[chr2<lb] = lb
    chr2[chr2<rb] = rb
'''
"""变异算子 
单点
"""
def mutate(pops, pm, nChr,m):
    nPop = pops.shape[0] 
    for i in range(nPop):
        if np.random.rand() < pm:
            zz=np.random.rand(m)
            zz = torch.tensor(zz).to(torch.float32)
            #zzz = 4*zz-2#-2到2
            #pos = np.random.randint(0,nChr,1)
            pos = np.random.randint(0, nChr, m)#变异2个位置
            pops[i][pos] = zz
    return pops
#分组线性随机变异
def group_mut(pops, lb, rb, pm,numberOfGroups):
    nPop, nChr = pops.shape
    outIndexarray = creategroup(nPop, nChr, numberOfGroups)
    chosengroups = np.random.randint(1, numberOfGroups + 1, size=outIndexarray.shape[0])  # size=outIndexList.shape[0]
    Site = np.zeros((nPop, nChr))
    for i in range(len(chosengroups)):
        Site[i, :] = (outIndexarray[i, :] == chosengroups[i]).astype(int)
    # 生成随机数判断是否变异
    mu = np.random.rand(nPop, 1)
    mu = np.tile(mu, (1, nChr))
    # 选中的组且小于变异概率，p*emb
    temp = np.where((Site == 1) & (mu < pm), 1, 0)
    pops[np.where(temp == 1)] = (np.random.rand(len(np.where(temp==1)[0]))-0.5)*2
    pops = np.minimum(np.maximum(pops, lb), rb)
    return pops

#分组线性多项式变异
def group_polyMutation(pops, lb, rb,disM, pm,numberOfGroups):
    nPop, nChr = pops.shape
    outIndexarray = creategroup(nPop, nChr, numberOfGroups)
    chosengroups = np.random.randint(1, numberOfGroups + 1, size=outIndexarray.shape[0])  # size=outIndexList.shape[0]
    Site = np.zeros((nPop, nChr))
    for i in range(len(chosengroups)):
        Site[i, :] = (outIndexarray[i, :] == chosengroups[i]).astype(int)
    # 生成随机数判断是否变异
    mu = np.random.rand(nPop, 1)
    mu = np.tile(mu, (1, nChr))
    # 选中的组且小于变异概率，p*emb
    temp = np.where((Site == 1) & (mu < pm), 1, 0)
    pops = np.minimum(np.maximum(pops, lb), rb)
    Upper = np.full((nPop, nChr), rb)
    Lower = np.full((nPop, nChr), lb)
    # 种群中对应位置计算变化
    pops[np.where(temp == 1)] = pops[np.where(temp == 1)] + (Upper[np.where(temp == 1)] - Lower[np.where(temp == 1)]) * (
                                             (2 * mu[np.where(temp == 1)] + (1 - 2 * mu[np.where(temp == 1)]) *
                                              (1 - (pops[np.where(temp == 1)] - Lower[np.where(temp == 1)]) / (
                                                          Upper[np.where(temp == 1)] - Lower[np.where(temp == 1)])) ** (
                                                          disM + 1)) ** (1 / (disM + 1)) - 1)
    temp = np.where((Site == 1) & (mu > pm), 1, 0)
    pops[np.where(temp == 1)] = pops[np.where(temp == 1)] + (
                Upper[np.where(temp == 1)] - Lower[np.where(temp == 1)]) * (
                                             1 - (2 * (1 - mu[np.where(temp == 1)]) + 2 * (
                                                 mu[np.where(temp == 1)] - 0.5) *
                                                  (1 - (Upper[np.where(temp == 1)] - pops[np.where(temp == 1)]) / (
                                                              Upper[np.where(temp == 1)] - Lower[
                                                          np.where(temp == 1)])) ** (disM + 1)) ** (1 / (disM + 1)))
    return pops


#生成随机分组索引
def creategroup(nPop,nChr,numberOfGroups):
    outIndexarray = []
    for i in range(nPop):
        # 初始化索引列表
        outIndexList = []
        varsPerGroup = nChr // numberOfGroups
        # 循环生成索引列表
        for i in range(1, numberOfGroups):
            outIndexList.extend(np.ones(varsPerGroup) * i)
        # 补足长度以确保所有变量都被分组
        outIndexList.extend(np.ones(nChr - len(outIndexList)) * numberOfGroups)
        # 对索引列表进行随机排列
        np.random.shuffle(outIndexList)
        outIndexarray.append(outIndexList)
    outIndexarray = np.array(outIndexarray)
    return outIndexarray

'''
#多项式变异
def mutate_mutpol(pops, pm, etaM, lb, rb):
    """
            :param pm: the probabilities of doing mutate
            :param etaM: the distribution index of mutate，设20
            lb：下界
            rb：上界
    """
    nPop = pops.shape[0]
    for i in range(nPop):
        if np.random.rand() < pm:
            polyMutation(pops[i], etaM, lb, rb)
    return pops


def polyMutation(chr, etaM, lb, rb):
    # 多项式变异
    pos1, pos2 = np.sort(np.random.randint(0,len(chr),2))
    pos2 += 1
    u = np.random.rand()
    if u < 0.5:
        delta = (2*u) ** (1/(etaM+1)) - 1
    else:
        delta = 1-(2*(1-u)) ** (1/(etaM+1))
    chr[pos1:pos2] += delta
    chr[chr<lb] = lb
    chr[chr>rb] = rb
'''
"""扰动产生新的子代 
多点
"""
def Disturb(pops, nChr,m,lb, rb):
    '''
    nPop = pops.shape[0]
    dis_pop = np.zeros((nPop, nChr))
    gauss = np.random.normal(0, 1, (nPop, nChr))
    dis_pop[:nPop] = gauss*0.5 + pops
    '''
    nPop = pops.shape[0]
    for i in range(nPop):
        pos = np.random.randint(0, nChr, m)  # 变异m个位置
        gauss = np.random.normal(0, 1, m)
        pops[i][pos] = pops[i][pos]+gauss#*0.5
        pops[i][pops[i] < lb] = lb
        pops[i][pops[i] > rb] = rb
    return pops

"""
种群或个体的适应度 
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

#多目标qed,jnk3,sa_nom,sim
def fitness_qedjnksa(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qedjnksa(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qedjnksa(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    #print('seq:', seq)
    qed = QED(mol)  # 需改为种群
    jnk3 = jnk(seq)
    sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, jnk3, sa_nom, sim  # pen_logP

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

#多目标qed,drd2,sa_nom,sim
def fitness_qeddrdsa(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qeddrdsa(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qeddrdsa(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)  # 需改为种群
    drd = drd2(seq)
    sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, drd, sa_nom, sim  # pen_logP

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

#计算约束违反度
def CV_globel(mol, c):
    # 计算种群的违反度
    CV =  np.array([cv(x,c) for x in mol])#CV
    #print(CV)
    if c==1:
        return CV
    else:
        max_cv1 = max(CV[:,0])
        max_cv2 = max(CV[:, 1])
        return max_cv1, max_cv2
'''
#计算余弦相似度
def cos(z_0,Z):
    sim_cos=[]
    zz = z_0.numpy().reshape(512)
    for i in range(Z.shape[0]):
        z_1 = Z[i].numpy().reshape(512)
        dot_sim = dot(zz, z_1) / (norm(zz) * norm(z_1))
        sim_cos.append(dot_sim)
    return sim_cos  # pen_logP
'''
"""
种群的合并和优选 
"""
def optSelect(pops, fits, chrPops, chrFits):
    """种群合并与优选 
    Return: 
        newPops, newFits 
    """
    nPop, nChr = pops.shape 
    nF = fits.shape[1] 
    newPops = np.zeros((nPop, nChr)) 
    newFits = np.zeros((nPop, nF)) 
    # 合并父代种群和子代种群构成一个新种群 
    MergePops = np.concatenate((pops,chrPops), axis=0) 
    MergeFits = np.concatenate((fits,chrFits), axis=0) 
    MergeRanks = nonDominationSort(MergePops, MergeFits) 
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks) 

    indices = np.arange(MergePops.shape[0]) 
    r = 0 
    i = 0 
    rIndices = indices[MergeRanks==r]  # 当前等级为r的索引 
    while i + len(rIndices)  <= nPop:
        newPops[i:i+len(rIndices)] = MergePops[rIndices] 
        newFits[i:i+len(rIndices)] = MergeFits[rIndices] 
        r += 1  # 当前等级+1 
        i += len(rIndices) 
        rIndices = indices[MergeRanks==r]  # 当前等级为r的索引 
    
    if i < nPop: 
        rDistances = MergeDistances[rIndices]   # 当前等级个体的拥挤度 
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小 
        surIndices = rIndices[rSortedIdx[:(nPop-i)]]  
        newPops[i:] = MergePops[surIndices] 
        newFits[i:] = MergeFits[surIndices] 
    return (newPops, newFits)



#种群的CV需更新,种群SMILES需更新
def optSelect_uni(pops, fits, CV_pops, smiles, chrPops, chrFits, CV_offspring, chrsmiles, nPop):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nChr = pops.shape[1]
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    newCV = [0]*nPop
    newsmiles = [0] * nPop
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeCV = np.concatenate((CV_pops, CV_offspring), axis=0)
    Mergesmiles = np.concatenate((smiles, chrsmiles), axis=0)
    #首先去除重复
    MergeFits_uni, indices = np.unique(MergeFits, axis=0, return_index=True)
    MergePops_uni = MergePops[indices]
    MergeCV_uni = MergeCV[indices]
    Mergesmiles_uni = Mergesmiles[indices]
    MergeRanks = nonDominationSort(MergePops_uni, MergeFits_uni)
    MergeDistances = crowdingDistanceSort(MergePops_uni, MergeFits_uni, MergeRanks)

    indices = np.arange(MergePops_uni.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) < nPop:
        newPops[i:i + len(rIndices)] = MergePops_uni[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits_uni[rIndices]
        newCV[i:i + len(rIndices)] = MergeCV_uni[rIndices]
        newsmiles[i:i + len(rIndices)] = Mergesmiles_uni[rIndices]
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        # 若加到最后一个等级仍不足种群数，随机采分子
        # 不够的分子采样补全
        if r == max(MergeRanks) + 1:
            IID = indices.tolist()
            for j in range(i, nPop):
                idx1, idx2 = random.sample(IID, 2)  # 随机挑选两个个体
                idx = compare(idx1, idx2, MergeRanks, MergeDistances)
                newPops[j] = MergePops_uni[idx]
                newFits[j] = MergeFits_uni[idx]
                newCV[j] = MergeCV_uni[idx]
                newsmiles[j] = Mergesmiles_uni[idx]
                j += 1
                i += 1

    if i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops_uni[surIndices]
        newFits[i:] = MergeFits_uni[surIndices]
        newCV[i:] = MergeCV_uni[surIndices]
        newsmiles[i:] = Mergesmiles_uni[surIndices]
    return (newPops, newFits, newCV, newsmiles)

def select1_uni(pool, pops, fits, CV_pops, smiles):
    # 一对一锦标赛选择
    # pool: 新生成的种群大小
    fits_uni, indices = np.unique(fits, axis=0, return_index=True)
    pops_uni = pops[indices]
    CV_uni = CV_pops[indices]
    smi_uni = smiles[indices]

    ranks = nonDominationSort(pops_uni, fits_uni)
    distances = crowdingDistanceSort(pops_uni, fits_uni, ranks)
    nPop, nChr = pops_uni.shape
    nF = fits_uni.shape[1]
    newPops = np.zeros((pool, nChr))
    newFits = np.zeros((pool, nF))
    newCV = [0] * pool
    newsmiles = [0] * pool
    indices = np.arange(nPop).tolist()
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体
        idx = compare(idx1, idx2, ranks, distances)

        newPops[i] = pops_uni[idx]
        newFits[i] = fits_uni[idx]
        newCV[i] = CV_uni[idx]
        newsmiles[i] = smi_uni[idx]
        i += 1
    return newPops, newFits, newCV, newsmiles

def optSelect_id(pops, fits, chrPops, chrFits):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)
    optse_id = []
    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) <= nPop:
        newPops[i:i + len(rIndices)] = MergePops[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits[rIndices]
        for j in rIndices:
            optse_id.append(j)
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引

    if i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
        for s in surIndices:
            optse_id.append(s)
    return newPops, newFits, optse_id

#接收概率，按照NSGA2排序，分子仍有一定概率不被接收
def optSelect_ap(pops, fits, chrPops, chrFits, ap):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    apIndices = rIndices[np.random.rand(len(rIndices)) <= ap]  # 当前等级中通过概率筛选的索引

    while i + len(apIndices) <= nPop:
        newPops[i:i + len(apIndices)] = MergePops[apIndices]#添加满足概率的当前等级的解
        newFits[i:i + len(apIndices)] = MergeFits[apIndices]
        r += 1  # 当前等级+1
        i += len(apIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        apIndices = rIndices[np.random.rand(len(rIndices)) <= ap]  # 当前等级中通过概率筛选的索引

    if i < nPop:
        rDistances = MergeDistances[apIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
    return (newPops, newFits)


'''
smi = ['C=c1c2c(nc3c1cc(C(=O)N1COCC1(C)C)n3C)C(CN)=CC=2','CN1C(=O)CC(N2c3ccccc3SC(c3ccco3)C2c2ccco2)c2ccccc21']
test_mol = [Chem.MolFromSmiles(x) for x in smi]
cv_mol = CV(test_mol)
#print(Descriptors.MolWt(mol))


pops = np.random.rand(3, 6)
new_pops = group_mut(pops, -1, 1, 0.5,3)
'''