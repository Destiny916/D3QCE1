# coding:utf-8
import math
import random
from DataRead import DataReadDHHJSP1
import os
from Initial import *
import numpy as np
from CalFitness import FitDHHFSP,EnergySave_DHHFSP
from GA import *
import  copy
import time as TIME   #怎么用
from Tool import *
from  LocalSearch import *
from DDQN_model import DoubleDQN
import torch
import argparse
from D3QN1 import D3QN
# from judgement import *
from  NEWlocalsearch1 import *
FILENAME=['20J3S2F.txt','20J5S2F.txt','20J3S3F.txt','20J5S3F.txt',\
          '40J3S2F.txt','40J5S2F.txt','40J3S3F.txt','40J5S3F.txt',\
          '60J3S2F.txt','60J5S2F.txt','60J3S3F.txt','60J5S3F.txt',\
          '80J3S2F.txt','80J5S2F.txt','80J3S3F.txt','80J5S3F.txt',\
          '100J3S2F.txt','100J5S2F.txt','100J3S3F.txt','100J5S3F.txt'];
#parameter

filenum=20;
# FILENAME=['20J3S2F.txt','20J5S2F.txt','20J3S3F.txt','20J5S3F.txt',\
#           '40J3S2F.txt','40J5S2F.txt','40J3S3F.txt','40J5S3F.txt',\
#           '60J3S2F.txt','60J5S2F.txt','60J3S3F.txt','60J5S3F.txt',\
#           '80J3S2F.txt','80J5S2F.txt','80J3S3F.txt','80J5S3F.txt',\
#           '100J3S2F.txt','100J5S2F.txt','100J3S3F.txt','100J5S3F.txt'];
# #parameter
# filenum=20;
runtime=1;
#ps=80;Pc=1.0;Pm=0.2;
ps=100;#改
Pc=1.0;Pm=0.2;#代数，遗传交叉，遗传突变

lr=0.005;batch_size=32;
EPSILON = 0.9               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 7   # target update frequency

N_ACTIONS = 20  # 6种候选的算子，
EPOCH=1

actions = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3],

           [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [0, 4, 5],

           [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],

           [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]

print(torch.cuda.is_available())
DATAPATH='DATASET_DHHFSP1/';
INSPATH=[];RESPATH=[];
for file in range(filenum):
    temp=DATAPATH+FILENAME[file];
    INSPATH.append(temp)
    if file<9:
        id='0'+str(file+1);
    else:
        id=str(file+1);
    temp2='DHHFSP'+id+'\\'
    RESPATH.append(temp2)
for file in range(filenum):
    HVALL=0
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=500)
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/D3QN/')
    parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
    parser.add_argument('--epsilon_p   ath', type=str, default='./output_images/epsilon.png')

    args = parser.parse_args()
    Timeall=0
    N,TS,F,NS,time,JP,JDD,Length = DataReadDHHJSP1(INSPATH[file])#N行数，TS列数，F几组数据 NS 3 2 3 time具体数据 jp 倒数第二 jdd倒数第一
    MaxNFEs = 400 * N  # 8000
    if MaxNFEs < 20000:
        MaxNFEs = 20000
    # create filepath to store the pareto solutions set for each independent run
    respath = 'CPTEST2\\last\\';
    sprit = '\\'
    respath = respath + RESPATH[file]
    isExist = os.path.exists(respath)
    # if the result path has not been created
    if not isExist:
        currentpath = os.getcwd()
        os.makedirs(currentpath + sprit + respath)
    print(RESPATH[file], 'is being Optimizing\n')
    # start independent run for GMA
    for rround in range(runtime):#一文件里大循环20次
        Time1 = TIME.time()
        p_chrom, f_chrom, _ = HInitial(ps, N, F, TS, time, JP, JDD)  # pchrom是操作序列 fchrom是选择机器
        fitness = np.zeros(shape=(ps, 4))
        NFEs = 0  # number of function evaluation
        # calucate fitness of each solution
        for i in range(ps):
            fitness[i, 0], fitness[i, 1], fitness[i, 2], fitness[i, 3] = EnergySave_DHHFSPgai(p_chrom[i, :], f_chrom[i, :], N,time,F,TS,NS, JP, JDD, Length)#到这100个解
        AP = [];AF = [];AFit = []  # Elite archive
        # build model
        N_STATES = 2 * N
        CountOpers = np.zeros(N_ACTIONS)
        PopCountOpers = []
        dq_net_0 = D3QN(alpha=0.0003, state_dim=N_STATES, action_dim=N_ACTIONS,
                        fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.95, tau=0.001, epsilon=1.0,
                        eps_end=0.05, eps_dec=5e-4, max_size=500000, batch_size=128)
        u = 1
        while NFEs < MaxNFEs:#算法
            print(FILENAME[file] + ' round ', rround + 1, 'iter ', u)
            u = u + 1
            p_chrom, f_chrom, fitness = NSGA2POXES(p_chrom, f_chrom, fitness, Pc, Pm, ps, N, time, F, TS, NS, JP, JDD, Length)#算法？ 100个解
            NFEs = NFEs + 2 * ps
            # Elite strategy
            PF = pareto4(fitness)#此处？？？
            if len(AFit) == 0:
                AP = p_chrom[PF, :]
                AF = f_chrom[PF, :]
                AFit = fitness[PF, :]
            else:
                AP = np.vstack((AP, p_chrom[PF, :]))
                AF = np.vstack((AF, f_chrom[PF, :]))
                AFit = np.vstack((AFit, fitness[PF, :]))
            PF = pareto4(AFit)
            AP = AP[PF, :]
            AF = AF[PF, :]
            AFit = AFit[PF, :]
            AP, AF, AFit = DeleteReaptE4(AP,AF,AFit)
            #Elite local search
            L = len(AFit)
            current_state = np.zeros(N_STATES, dtype=int)
            next_state = np.zeros(N_STATES, dtype=int)
            Fit=np.zeros(4)
            for j in range(L):
                #localsearch(AP[j, :], AF[j, :], AFit[j, :],N,F,JDD,JP,j,L)
                current_state[0:N] = copy.copy(AP[j, :])
                current_state[N:N * 2] = copy.copy(AF[j, :])
                action = dq_net_0.choose_action(current_state, isTrain=True)
                # action = dq_net_0.choose_action(current_state)
                k = int(random.choice(actions[action]))

                if k == 0:
                    P1, F1 = DSwap(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP, Length)  # N3
                    NFEs = NFEs + 1
                if k == 1:
                    P1, F1 = PSwap(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP, Length)  # N1
                    NFEs = NFEs + 1

                elif k == 2:
                    P1, F1 = PInsert(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP, Length)  # N2
                    NFEs = NFEs + 1
                elif k == 3:
                    P1, F1 = DInsert5(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP, Length)  # N4
                    NFEs = NFEs + 1

                elif k == 4:
                    P1, F1 ,NFEs= PSWAPNEW(AP[j, :], AF[j, :], N, F, JDD, JP, Length,NFEs)  # N9
                    P1 = P1.astype(int)
                    F1 = F1.astype(int)
                elif k == 5:
                    P1, F1 ,NFEs= PINSERTNEW(AP[j, :], AF[j, :], N, F, JDD, JP, Length,NFEs)  # N9
                    P1 = P1.astype(int)
                    F1 = F1.astype(int)
                Fit[0], Fit[1], Fit[2], Fit[3] = EnergySave_DHHFSPgai(P1, F1, N, time, F, TS, NS, JP, JDD, Length)
                nr = NDS(Fit, AFit[j, :])
                if nr == 1:
                    AP[j,:] = copy.copy(P1)
                    AF[j,:] = copy.copy(F1)
                    AFit[j,:] = copy.copy(Fit)
                    reward = 20
                elif nr == 0:
                    AP = np.vstack((AP, P1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit))
                    if AFit[j, 0] < Fit[0]:
                        reward = 15
                    elif AFit[j, 1] < Fit[1]:
                        reward = 10
                    elif AFit[j, 2] < Fit[1]:
                        reward = 10
                    else:
                        reward = 0
                else:
                    reward = 0
                next_state[0:N] = copy.copy(P1)#尝试无深度学习时
                next_state[N:N * 2] = copy.copy(F1)
                for action_i in range(len(actions)):
                    if k in actions[action_i]:
                        # dq_net_0.store_transition(current_state, action_i, reward, next_state)
                        dq_net_0.remember(current_state, action_i, reward, next_state, True)
                        dq_net_0.learn()
        #write elite solutions in txt
        PF = pareto4(AFit)
        AP = AP[PF, :];
        AF = AF[PF, :];
        AFit = AFit[PF, :];

        PF = pareto4(AFit)
        l = len(PF)
        obj = AFit[:, 0:3]
        P = AP
        F1 = AF

        newobj = []
        newp = []
        newf = []

        for i in range(l):
            newobj.append(obj[PF[i], :])
            newp.append(P[PF[i], :])
            newf.append(F1[PF[i], :])
        newobj = np.array(newobj)
        newp = np.array(newp)
        newf = np.array(newf)
        newobj, unique_indices = np.unique(newobj, axis=0, return_index=True)
        newp = newp[unique_indices]
        newf = newf[unique_indices]
        print(len(newobj), len(newp), len(newf))
        tmp = 'res'
        resPATH = respath + sprit + tmp + str(rround + 1) + '.txt'
        f = open(resPATH, "w", encoding='utf-8')
        l = len(newobj)
        for i in range(l):
            item = '%5.2f %6.2f %5.2f \n' % (newobj[i][0], newobj[i][1], newobj[i][2])  # fomat writing into txt file
            f.write(item)
        f.close()

        tmp = 'P'
        resPATH = respath + sprit + tmp + str(rround + 1) + '.txt'
        with open(resPATH, "w", encoding='utf-8') as f:
            l = len(newp)
            for i in range(l):
                ll = len(newp[i])
                for j in range(ll):
                    item = f'{newp[i][j]:f} '  # 使用f-string格式化
                    f.write(item)
                f.write('\n')  # 正确的换行符

        tmp = 'F'
        resPATH = respath + sprit + tmp + str(rround + 1) + '.txt'
        with open(resPATH, "w", encoding='utf-8') as f:
            l = len(newf)  # 应该基于newf计算长度，以确保一致性
            for i in range(l):
                ll = len(newf[i])
                for j in range(ll):
                    item = f'{newf[i][j]:f} '
                    f.write(item)
                f.write('\n')  # 正确的换行符
    print('finish ' + FILENAME[file])
print('finish running')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/