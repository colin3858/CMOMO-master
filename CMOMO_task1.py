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
    parser.add_argument("--opt", default='qedplogp',
                        choices=['qed', 'logP', 'qedplogp', 'qeddrd', 'qedsajnk', 'qedsagsk', 'qedsadrd2'])
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
        tres = [0.8, 0.5, 0.7, 0.3]
        col = ['SMILES', 'mol_id', 'qed', 'gskb', 'sa_nom', 'sim']
    args = parser.parse_args()
    # device = 'cuda'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    ######loading the pre-trained encoder-decoder
    model = CDDDModel()
    # loading dataset
    data = pd.read_csv('./data/qedplogp_test.csv').values
    simbank = pd.read_csv('./data/archive/qedplogp_simbank_100.csv').values
    smi_iter_last = []  
    HV_pop = []
    HV_arc = []
    smi_stage1 = []
    num_feasible = [] 
    num_feasible_arc = [] 
    runtime = []
    # parameter settings
    nPop = 100  # the size of population
    pc = 1  # crossover rate
    pm = 0.5  # mutation rate
    nChr = 512  # length of vectors
    lb = -1  # lower bound
    rb = 1  # upper bound
    numberOfGroups = 16 #number of groups in mutation
    d = 0.26  # the parameter in crossover
    disM = 20
    nIter = 100  # the total number of evolution generation
    nIter_1  = 50  # evolution generation in stage 1
    restart = 1
    SR = 0
    # the ID of lead molecules in dataset
    a = list(range(0, 20, 20))  # 180
    b = list(range(20, 100, 20))  # 200
    c = 2  # 几条约束，计算约束违反度

    for num in range(len(a)):
        mm1 = a[num]
        mm2 = b[num]
        pop_smis_iter = [['SMILES', 'mol_id', 'iter', 'QED', 'sim', 'pogp_imp','CV']]
        archive_smis_iter = [['SMILES', 'mol_id', 'iter', 'QED', 'sim', 'pogp_imp','CV']]
        # the optimization of each lead molecule
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
            z_0 = model.encode(smiles)  # the encoding of the lead
            fp_0 = morgan_fingerprint(Chem.MolFromSmiles(seq))
            fits_0 = ff_qedlogp(seq, mol_0, fp_0)
            print(fits_0)
            # the embedding of molecules inthe bank library
            aa = simbank[:, 1] == i
            index = np.where(aa == 1)
            simbank_i = simbank[index]
            num_int = len(simbank_i)
            bank_emb = np.zeros((num_int, 512))
            for i in range(num_int):
                bank_emb[i] = model.encode(simbank_i[i][0])
            bank_emb = torch.tensor(bank_emb).to(torch.float32)

            ########### optimization process
            r = 1  # the time of re-optimization
            while r <= restart:
                hv_pop = np.zeros(nIter)
                hv_arc = []
                num_feasible_i = []
                num_feasible_arc_i = np.zeros(nIter + 1)
                t1 = time.time()

                # Population initialization
                bankpop = crossover_z0_2(z_0, bank_emb, pc, d, lb, rb)
                bankpop = torch.tensor(bankpop).to(torch.float32)
                # 解码评估
                bankmol, banksmiles = model.decode(bankpop)
                banksmiles = np.array(banksmiles)
                bankfits = fitness_qedlogp(banksmiles, bankmol, fp_0)  # 适应度计算
                bankfits[np.isnan(bankfits)] = -20
                CV_bank = CV(bankmol, c)

                pops, fits, CV_pops, smis = select1_uni(nPop, bankpop, bankfits, CV_bank, banksmiles)
                num_feasible_i.append(CV_pops.count(0))  # number of feasible molecules in the initial population
                num_feasible_arc_i[0] = CV_pops.count(0)
                print('the initial population')

                iter = 0
                Archive_pops = np.zeros((1, 1))
                Archive_fits = np.zeros((1, 1))
                Archive_CV = np.zeros((1, 1))
                Archive_smi = np.zeros((1, 1))
                # optimization generation
                while iter < nIter:
                    # 进度条
                    print("【进度】【{0:20s}】【正在进行{1}代...】【共{2}代】". \
                          format('▋' * int(iter / nIter * 20), iter, nIter), end='\r')
                    #optimization stage 1
                    if iter < nIter_1:
                        print('stage1')
                        #generate offspring molecules
                        chrpops = crossover_2(pops, pc, d, lb, rb)
                        chrpops = group_mut(chrpops,lb, rb, pm, numberOfGroups)
                        # evaluate fitness
                        chrpops = torch.from_numpy(chrpops)
                        chrmol, chrsmiles = model.decode(chrpops)  ###decoding
                        chrfits = fitness_qedlogp(chrsmiles, chrmol, fp_0)
                        chrfits[np.isnan(chrfits)] = -20
                        print('stage1 update archive')
                        CV_offspring = CV(chrmol, c)
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
                        print('stage1 update population')
                        pops, fits, CV_pops, smis = optSelect_uni(pops, fits, CV_pops, smis, chrpops, chrfits, CV_offspring, chrsmiles, nPop)
                        num_feasible_i.append(CV_pops.count(0))
                        ###save smiles in population at each iteration
                        for i in range(len(smis)):
                            tuple = [smis[i], nn, iter, fits[i][0], fits[i][2],fits[i][1]-fits_0[1], CV_pops[i]]
                            pop_smis_iter.append(tuple)
                        ### save smiles in archive at stage1
                        for i in range(len(Archive_smi)):
                            tuple = [Archive_smi[i], nn, iter, Archive_fits[i][0], Archive_fits[i][2],Archive_fits[i][1]-fits_0[1], Archive_CV[i]]
                            archive_smis_iter.append(tuple)
                        ###save cHV, cPD in stage 1
                        fits_c = fits[np.where(np.array(CV_pops)==0)]
                        try:
                            dominated_hypervolume = pg.hypervolume(np.array(
                                [[-1, -0.1, -1] * (np.array(fit) - np.array([0, fits_0[1], 0])) for
                                 fit in fits_c if
                                 fit[1] > fits_0[1]])).compute(np.array([0, 0, 0]))
                        except:
                            dominated_hypervolume = 0
                        hv_pop[iter] = dominated_hypervolume

                        archive_fits_c = Archive_fits[np.where(Archive_CV == 0)]
                        try:
                            dominated_hypervolume = pg.hypervolume(np.array(
                                [[-1, -0.1, -1] * (np.array(fit) - np.array([0, fits_0[1], 0])) for
                                 fit in archive_fits_c if
                                 fit[1] > fits_0[1]])).compute(np.array([0, 0, 0]))
                        except:
                            dominated_hypervolume = 0
                            pure_div = 0
                        hv_arc.append((dominated_hypervolume))

                        # #optimization stage 2
                    else:
                        print('stage2')
                        #generate offspring molecules
                        num_feasible_i.append(np.count_nonzero(Archive_CV == 0))
                        chrpops = crossover_2(Archive_pops, pc, d, lb, rb)
                        chrpops = group_mut(chrpops,lb, rb, pm, numberOfGroups)
                        # calculate fitness
                        chrpops = torch.from_numpy(chrpops)
                        chrmol, chrsmiles = model.decode(chrpops)
                        chrfits = fitness_qedlogp(chrsmiles, chrmol, fp_0)
                        chrfits[np.isnan(chrfits)] = -20
                        CV_offspring = CV(chrmol, c)
                        # merge parent and offspring molecules
                        MergePops = np.concatenate((Archive_pops, chrpops), axis=0)
                        MergeFits = np.concatenate((Archive_fits, chrfits), axis=0)
                        MergeCV = np.concatenate((Archive_CV, CV_offspring), axis=0)
                        Mergesmiles = np.concatenate((Archive_smi, chrsmiles), axis=0)
                        MergeFits_uni, indices = np.unique(MergeFits, axis=0, return_index=True)
                        MergePops_uni = MergePops[indices]
                        MergeCV_uni = MergeCV[indices]
                        Mergesmiles_uni = Mergesmiles[indices]
                        # Rank1
                        print('stage2 rank1')
                        S1 = score1_nsga2(MergePops_uni, MergeFits_uni)
                        # Rank2
                        print('stage2  rank2')
                        S2 = score2_CDP(MergePops_uni, MergeFits_uni, MergeCV_uni)
                        # Score_all, select molecules
                        alpha = math.cos((iter - nIter_1) / (nIter - nIter_1) * math.pi)
                        S_all = 0.5 * (1 + alpha) * S1 + (1 - 0.5 * (1 + alpha)) *S2 
                        sortedID = np.argsort(S_all) 
                        print('stage2 update population')
                        Archive_pops = MergePops_uni[sortedID[0:nPop]]
                        Archive_fits = MergeFits_uni[sortedID[0:nPop]]
                        Archive_CV = MergeCV_uni[sortedID[0:nPop]]
                        Archive_smi = Mergesmiles_uni[sortedID[0:nPop]]

                        # molecules in stage 2 saved in population file
                        for i in range(len(Archive_smi)):
                            tuple = [Archive_smi[i], nn, iter, Archive_fits[i][0], Archive_fits[i][2], Archive_fits[i][1]-fits_0[1],Archive_CV[i]]
                            pop_smis_iter.append(tuple)
                        # HV,PD in stage 2
                        archive_fits_c = Archive_fits[np.where(Archive_CV == 0)]
                        try:
                            dominated_hypervolume = pg.hypervolume(np.array(
                                [[-1, -0.1, -1] * (np.array(fit) - np.array([0, fits_0[1], 0])) for
                                 fit in archive_fits_c if
                                 fit[1] > fits_0[1]])).compute(np.array([0, 0, 0]))
                        except:
                            dominated_hypervolume = 0
                        hv_pop[iter] = dominated_hypervolume
                    iter = iter + 1
                    print(iter)
                        
                #the last population
                endsmiles = np.array(Archive_smi)
                CV_endmol = Archive_CV
                endfits = Archive_fits
                # Pareto molecules in the last population
                ranks = CDP_simple(endsmiles, endfits, CV_endmol)  # 约束非支配排序
                paretoendsmiles = endsmiles[ranks == 0]
                paretoFits = endfits[ranks == 0]
                paretoCV = CV_endmol[ranks == 0]
                # the optimized molecules
                rr = []
                unique_smiles = []
                for i in range(len(paretoendsmiles)):
                    if paretoCV[i] == 0:
                        if paretoendsmiles[i] not in unique_smiles:
                            unique_smiles.append(paretoendsmiles[i])
                            tuple = (paretoendsmiles[i], nn, paretoFits[i][0], paretoFits[i][2], paretoFits[i][1]-fits_0[1], paretoCV[i])
                            smi_iter_last.append(tuple)
                            if (np.array(paretoFits[:][i]) >= np.array([0.85, (fits_0[1] + 3), 0.3])).all() == 1:
                                rr.append(1)

                if 1 in rr:
                    r = restart + 1
                    SR = SR + 1
                else:
                    r = r + 1
                print('restart:', r)
                t2 = time.time()
                time_1 = (t2 - t1) / 60
                print('run time:', time_1)
                runtime.append(time_1)

            HV_pop.append(hv_pop)
            HV_arc.append(hv_arc)
            num_feasible.append(num_feasible_i)
            num_feasible_arc.append(num_feasible_arc_i)

            result = [nn - a[0] + 1, SR]
            print('result-all,SR:', result)
            print('save mol:', nn)

            task = 'task1'
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
