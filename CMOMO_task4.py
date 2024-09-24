import argparse
import os

import numpy as np
from tensorboardX import SummaryWriter
# import tensorflow as tf
import time

from sub_code.models import CDDDModel
from sub_code.fitness import *
from sub_code.property import *
from warnings import simplefilter
from sub_code.generation_rule import *
from sub_code.selection_rule import *
simplefilter(action="ignore", category=FutureWarning)
import pygmo as pg
import math
# Suppress warnings
# tf.logging.set_verbosity(tf.logging.ERROR)
# RDLogger.logger().setLevel(RDLogger.CRITICAL)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default='qedsagsk',
                        choices=['qed', 'logP', 'qedplogp', 'qeddrd', 'qedsajnk', 'qedsagsk', '4lde'])
    args = parser.parse_args()
    if args.opt == 'qed':
        ff = ff_qed
        fitness = fitness_qed
    elif args.opt == 'logP':
        ff = ff_plogp
        fitness = fitness_plogp
    elif args.opt == 'qedplogp':
        ff = ff_qedlogp
        fitness = fitness_qedlogp
        col = ['SMILES', 'mol_id', 'qed', 'sim', 'plogp_imp', 'CV']
    elif args.opt == 'qeddrd':
        ff = ff_qeddrd
        fitness = fitness_qeddrd
        tres = [0.8, 0.3, 0.4]
        col = ['SMILES', 'mol_id', 'qed', 'sim', 'drd', 'CV']
    elif args.opt == 'qedsagsk':
        ff = ff_qedgsksa
        fitness = fitness_qedgsksa
        tres = [0.7, 0.4, 0.7, 0.2]
        col = ['SMILES', 'mol_id', 'qed', 'gskb', 'sa_nom', 'sim', 'cv']
    elif args.opt == '4lde':
        ff = ff_4lde
        fitness = fitness_4lde
        tres = [0.8, 0.3, 10]
        col = ['SMILES', 'mol_id', 'qed', 'sim', '4lde', 'CV']
    args = parser.parse_args()
    # device = 'cuda'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    ######加载预训练的JTVAE
    model = CDDDModel()
    # canonicalize
    data = pd.read_csv('./data/gsk3_test.csv').values

    simbank = pd.read_csv('./data/archive/gsk3_simbank_100.csv').values

    smi_iter_last = []  # 保留所有分子最后一代，一个分子npop个子代
    HV_pop = []
    HV_arc = []
    smi_stage1 = []
    num_feasible = []  # 存储阶段一种群和阶段二档案中可行解个数
    num_feasible_arc = []  # 存储阶段一档案中可行解个数
    runtime = []
    # 参数设置
    nPop = 100  # 种群个体数
    pc = 1  # 交叉率
    pm = 0.5  # 变异率
    nChr = 512  # 染色体长度
    lb = -1  # 下界
    rb = 1  # 上界
    d = 0.26  # 线性交叉参数
    numberOfGroups = 16
    nIter = 100  # 迭代次数
    nIter_1  = 50  # 迭代次数
    restart = 1
    m = 5  # 变异个数
    SR = 0  # 记录优化成功的分子数0.9，0.4
    a = [0]
    b = [1]
    c = 2  # 几条约束，计算约束违反度

    for num in range(len(a)):
        mm1 = a[num]
        mm2 = b[num]
        pop_smis_iter = [['SMILES', 'mol_id', 'iter', 'QED','gsk3','sa', 'sim', 'CV']]  # 分批存储每一代进化属性值，CV！！！
        archive_smis_iter = [['SMILES', 'mol_id', 'iter', 'QED','gsk3','sa', 'sim', 'CV']]
        for i in range(mm1,mm2):
            #num = num +1
            nn = i
            #一个分子SMILES序列
            smiles = data[i][0]
            print('ori_smiles:', smiles)
            # 一个分子SMILES序列
            mol_0 = Chem.MolFromSmiles(smiles)
            seq = Chem.MolToSmiles(mol_0)
            #print(seq)
            run_str = 'optiza'
            results_dir = os.path.join('results', args.seq)
            os.makedirs(results_dir, exist_ok=True)
            writer = SummaryWriter(os.path.join('runs', args.seq, run_str))
        #调试

            z_0 = model.encode(smiles)  # 对分子序列进行编码
            fp_0 = morgan_fingerprint(Chem.MolFromSmiles(seq))  # 分子序列的摩根指纹，用于计算相似性
            fits_0 = ff(seq, mol_0, fp_0)
            print(fits_0)
            #simbank嵌入
            aa = simbank[:, 1] == i
            index = np.where(aa == 1)
            simbank_i = simbank[index]  # 提取指定分子的初始种群
            num_int = len(simbank_i)
            bank_emb = np.zeros((num_int, 512))
            for i in range(num_int):
                bank_emb[i] = model.encode(simbank_i[i][0])
            bank_emb = torch.tensor(bank_emb).to(torch.float32)
            ########### 隐向量的扰动数量、控制相似性的系数，越大生成的相似性越低，容易产生空集
            r = 1  # 重启
            while r <= restart:
                hv_pop = np.zeros(nIter)  # 初始第一阶段值为0
                hv_arc = []
                num_feasible_i = []  # 储存当前分子种群可行解个数
                num_feasible_arc_i = np.zeros(nIter + 1)
                t1 = time.time()  # 开始时间

                # bank库
                bankpop = crossover_z0_2(z_0, bank_emb, pc, d, lb, rb)
                bankpop = torch.tensor(bankpop).to(torch.float32)
                # 解码评估
                bankmol, banksmiles = model.decode(bankpop)
                banksmiles = np.array(banksmiles)
                bankfits = fitness(banksmiles, bankmol, fp_0)  # 适应度计算
                bankfits[np.isnan(bankfits)] = -20
                CV_bank = CV(bankmol, c)

                pops, fits, CV_pops, smis = select1_uni(nPop, bankpop, bankfits, CV_bank, banksmiles)
                num_feasible_i.append(CV_pops.count(0))  # 初始种群的可行解个数
                num_feasible_arc_i[0] = CV_pops.count(0) # 初始种群的可行解个数和档案相同
                print('初始种群')
                iter = 0
                Archive_pops = np.zeros((1, 1))
                Archive_fits = np.zeros((1, 1))
                Archive_CV = np.zeros((1, 1))
                Archive_smi = np.zeros((1, 1))
                while iter < nIter:
                    # 进度条
                    print("【进度】【{0:20s}】【正在进行{1}代...】【共{2}代】". \
                          format('▋' * int(iter / nIter * 20), iter, nIter), end='\r')
                    #交叉变异
                    #print(fits)
                    if iter < nIter_1:
                        # 进化多个子代2N
                        print('stage1进化子代')
                        chrpops = crossover_2(pops, pc, d, lb, rb)  # 混合线性交叉
                        # chrpops = mutate(chrpops, pm, nChr, m)  # 变异产生子种群
                        chrpops = group_mut(chrpops, lb, rb, pm, numberOfGroups)
                        # 评估子代属性值
                        chrpops = torch.from_numpy(chrpops)
                        chrmol, chrsmiles = model.decode(chrpops)  ###解码潜在表示，分子和SMILES
                        chrfits = fitness(chrsmiles, chrmol, fp_0)
                        chrfits[np.isnan(chrfits)] = -20
                        # 更新Archive，不重复
                        print('stage1更新archive')
                        CV_offspring = CV(chrmol, c)  # 评估子代CV值
                        MergePops = np.concatenate((pops, chrpops), axis=0)
                        MergeFits = np.concatenate((fits, chrfits), axis=0)
                        MergeCV = np.concatenate((CV_pops, CV_offspring), axis=0)
                        Mergesmiles = np.concatenate((smis, chrsmiles), axis=0)
                        Archive_pops, Archive_fits, Archive_CV, Archive_smi = archive_stage1(MergePops, MergeFits,
                                                                                             MergeCV, Mergesmiles, nPop,
                                                                                             Archive_pops, Archive_fits,
                                                                                             Archive_CV, Archive_smi,
                                                                                             iter)
                        num_feasible_arc_i[iter+1] = np.count_nonzero(Archive_CV == 0)
                        # 更新种群，不重复
                        print('stage1更新population')
                        pops, fits, CV_pops, smis = optSelect_uni(pops, fits, CV_pops, smis, chrpops, chrfits, CV_offspring, chrsmiles, nPop)
                        num_feasible_i.append(CV_pops.count(0))  # 初始种群的可行解个数
                        ###save smiles in population at each iteration
                        for i in range(len(smis)):
                            tuple = [smis[i], nn, iter, fits[i][0], fits[i][1], fits[i][2],fits[i][3], CV_pops[i]]
                            pop_smis_iter.append(tuple)
                        ### save smiles in archive at stage1
                        for i in range(len(Archive_smi)):
                            tuple = [Archive_smi[i], nn, iter, Archive_fits[i][0], Archive_fits[i][1], Archive_fits[i][2],Archive_fits[i][3], Archive_CV[i]]
                            archive_smis_iter.append(tuple)
                        ###save HV, PD in stage 1
                        fits_c = fits[np.where(np.array(CV_pops) == 0)]
                        try:
                            dominated_hypervolume = pg.hypervolume(np.array([-1.0 * np.array(fit) for fit in fits_c if
                                                                             (np.array(fit) >= [0, 0, 0,
                                                                                                0]).all()])).compute(
                                np.zeros(len(fits[0])))
                        except:
                            dominated_hypervolume = 0
                        #hv_pop.append((dominated_hypervolume))
                        #pure_d_pop.append(pure_div)
                        hv_pop[iter] = dominated_hypervolume

                        archive_fits_c = Archive_fits[np.where(Archive_CV == 0)]
                        try:
                            dominated_hypervolume = pg.hypervolume(np.array(
                                [-1.0 * np.array(fit) for
                                 fit in archive_fits_c if
                                 (np.array(fit) >= [0, 0, 0, 0]).all()])).compute(np.array([0, 0, 0, 0]))
                        except:
                            dominated_hypervolume = 0
                            pure_div = 0
                        hv_arc.append((dominated_hypervolume))
                        # Stage 2:
                    else:
                        # 档案库开始进化
                        print('stage2更新子代')
                        num_feasible_i.append(np.count_nonzero(Archive_CV == 0))#初始档案时的可行解个数存储
                        chrpops = crossover_2(Archive_pops, pc, d, lb, rb)  # 混合线性交叉
                        chrpops = group_mut(chrpops,lb, rb, pm, numberOfGroups)
                        # 评估子代属性值
                        chrpops = torch.from_numpy(chrpops)
                        chrmol, chrsmiles = model.decode(chrpops)  ###解码潜在表示，分子和SMILES
                        chrfits = fitness(chrsmiles, chrmol, fp_0)
                        chrfits[np.isnan(chrfits)] = -20
                        CV_offspring = CV(chrmol, c)
                        # 合并父代种群和子代种群构成一个新种群
                        MergePops = np.concatenate((Archive_pops, chrpops), axis=0)
                        MergeFits = np.concatenate((Archive_fits, chrfits), axis=0)
                        MergeCV = np.concatenate((Archive_CV, CV_offspring), axis=0)
                        Mergesmiles = np.concatenate((Archive_smi, chrsmiles), axis=0)
                        # 首先去除重复
                        MergeFits_uni, indices = np.unique(MergeFits, axis=0, return_index=True)
                        MergePops_uni = MergePops[indices]
                        MergeCV_uni = MergeCV[indices]
                        Mergesmiles_uni = Mergesmiles[indices]
                        # Rank1
                        print('stage2排序1')
                        S1 = score1_nsga2(MergePops_uni, MergeFits_uni)
                        # Rank2
                        print('stage2排序2')
                        S2 = score2_CDP(MergePops_uni, MergeFits_uni, MergeCV_uni)
                        # Score_all,排名前N个分子进入archive，排序索引
                        alpha = math.cos((iter - nIter_1) / (nIter - nIter_1) * math.pi)
                        S_all = 0.5 * (1 + alpha) * S1 + (1 - 0.5 * (1 + alpha)) *S2  # 分数越小越好#
                        sortedID = np.argsort(S_all)  # 按照分数由小到大的位置索引
                        print('stage2更新population')
                        Archive_pops = MergePops_uni[sortedID[0:nPop]]
                        Archive_fits = MergeFits_uni[sortedID[0:nPop]]
                        Archive_CV = MergeCV_uni[sortedID[0:nPop]]
                        Archive_smi = Mergesmiles_uni[sortedID[0:nPop]]

                        # molecules in stage 2 saved in Archive file
                        for i in range(len(Archive_smi)):
                            tuple = [Archive_smi[i], nn, iter, Archive_fits[i][0], Archive_fits[i][1], Archive_fits[i][2], Archive_fits[i][3],Archive_CV[i]]
                            pop_smis_iter.append(tuple)
                        # HV,PD in stage 2
                        archive_fits_c = Archive_fits[np.where(Archive_CV == 0)]
                        try:
                            dominated_hypervolume = pg.hypervolume(np.array(
                                [-1.0 * np.array(fit) for
                                 fit in archive_fits_c if
                                 (np.array(fit) >= [0, 0, 0, 0]).all()])).compute(np.array([0, 0, 0, 0]))
                        except:
                            dominated_hypervolume = 0
                            pure_div = 0
                        hv_pop[iter] = dominated_hypervolume

                    iter = iter + 1
                    print(iter)
                        
                #解码最终种群分子
                endsmiles = np.array(Archive_smi)
                CV_endmol = Archive_CV
                endfits = Archive_fits
                # 最后一代约束pareto前沿上的分子
                ranks = CDP_simple(endsmiles, endfits, CV_endmol)  # 约束非支配排序
                paretoendsmiles = endsmiles[ranks == 0]
                paretoFits = endfits[ranks == 0]
                paretoCV = CV_endmol[ranks == 0]
                # 判断是否优化成功
                rr = []  # 储存endfits是否满足>=0.9,>=0.4，True=1，False=0
                unique_smiles = []
                for i in range(len(paretoendsmiles)):
                    if paretoCV[i] == 0:
                        if paretoendsmiles[i] not in unique_smiles:
                            unique_smiles.append(paretoendsmiles[i])
                            tuple = (paretoendsmiles[i], nn, paretoFits[i][0], paretoFits[i][1], paretoFits[i][2], paretoFits[i][3], paretoCV[i])
                            smi_iter_last.append(tuple)
                            if (np.array(paretoFits[:][i]) >= np.array(tres)).all() == 1:
                                rr.append(1)

                if 1 in rr:
                    r = restart + 1
                    SR = SR + 1
                else:
                    r = r + 1
                print('restart:', r)
                t2 = time.time()  # 一个分子重启动进化后时间
                time_1 = (t2 - t1) / 60  # 扰动个数
                print('run time:', time_1)
                runtime.append(time_1)

            HV_pop.append(hv_pop)
            HV_arc.append(hv_arc)
            num_feasible.append(num_feasible_i)
            num_feasible_arc.append(num_feasible_arc_i)

            result = [nn - a[0] + 1, SR]
            print('result-all,SR:', result)
            print('save mol:', nn)

            task = 'task4_bank_g'
            np.savetxt('./results/'+task+'/TSCMO_'+task+'_mol' + str(a[0]) + str(b[-1]) + '_HV_pop.txt', HV_pop, fmt='%s')
            np.savetxt('./results/'+task+'/TSCMO_'+task+'_mol' + str(a[0]) + str(b[-1]) + '_HV_arc.txt', HV_arc, fmt='%s')
            np.savetxt('./results/'+task+'/TSCMO_'+task+'_mol' + str(a[0]) + str(b[-1]) + '_num_feasible.txt',
                       num_feasible,fmt='%d')
            np.savetxt('./results/'+task+'/TSCMO_'+task+'_mol' + str(a[0]) + str(b[-1]) + '_num_arcfea.txt',
                       num_feasible_arc,fmt='%d')
            df_to_save_A_to_B = pd.DataFrame(smi_iter_last, columns=col)
            df_to_save_A_to_B.to_csv('./results/'+task+'/TSCMO_'+task+'_mol' + str(a[0]) + str(b[-1]) + '_endsmiles.csv',
                                     index=False)
            np.savetxt('./results/'+task+'/TSCMO_'+task+'_mol' + str(mm1) + str(mm2) + '_smi_iter_pop.txt', pop_smis_iter,
                       fmt='%s')
            np.savetxt('./results/'+task+'/TSCMO_'+task+'_mol' + str(mm1) + str(mm2) + '_smi_iter_arc.txt', archive_smis_iter,
                       fmt='%s')
            np.savetxt('./results/'+task+'/TSCMO_'+task+'_mol' + str(a[0]) + str(b[-1]) + '_result.txt', result, fmt='%s')
            np.savetxt('./results/'+task+'/TSCMO_'+task+'_mol' + str(a[0]) + str(b[-1]) + '_runtime.txt', runtime,
                       fmt='%s')




# writer.close()
