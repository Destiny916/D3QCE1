# coding:utf-8
import numpy as np
import math
import random
from DataRead import DataReadDHHJSP
import os
from Initial import *
import numpy as np

import  copy
import time as TIME   #怎么用
from Tool import *
from  LocalSearch import *
from DDQN_model import DoubleDQN
import torch
from judgement import *

from NEWlocalsearch1 import *





from  FastNDSort import *
from CalFitness import *

from FastNDSort_3 import *
def TSelection(p_chrom,f_chrom,fitness,ps,N):
    #mating selection pool
    pool_size=ps
    P_pool = np.zeros(shape=(ps, N), dtype=int)
    F_pool = np.zeros(shape=(ps, N), dtype=int) #fitness of pool solutions
    # compeitor number
    tour=2
    # tournament selection
    for i in range(pool_size):#100代
        index1=int(math.floor(random.random()*ps))#(0-100)
        index2 = int(math.floor(random.random() * ps))
        while index1==index2:
            index2 = int(math.floor(random.random() * ps))
        f1=fitness[index1,0:4]
        f2=fitness[index2,0:4]
        if (NDS(f1, f2) == 1):
            P_pool[i,:]=p_chrom[index1,:]
            F_pool[i,:]=f_chrom[index1,:]
        elif(NDS(f1, f2) == 2):
            P_pool[i, :] = p_chrom[index2, :]
            F_pool[i, :] = f_chrom[index2, :]
        else:
            if random.random() <= 0.5:
                P_pool[i, :] = p_chrom[index1, :]
                F_pool[i, :] = f_chrom[index1, :]
            else:
                P_pool[i, :] = p_chrom[index2, :]
                F_pool[i, :] = f_chrom[index2, :]
    return P_pool,F_pool

def POX(P1,P2,N):#p的交叉互换
    #inital offerspring
    NP1=P1;
    NP2=P2;
    #index of each operation in P1 and P2
    ci1=np.zeros(N,dtype=int)
    ci2 = np.zeros(N, dtype=int)
    # store some jobs in J1
    temp=[random.random() for _ in range(N) ]
    temp=mylistRound(temp)
    J1=find_all_index(temp,1)#find the index where value equal to 1  J1返回temp中那几位是1
    for j in range(N):
        if Ismemeber(P1[j], J1)==1: #if is in job set J
            ci1[j] = P1[j]+1

        if Ismemeber(P2[j], J1)==0: #if is not in job set J
            ci2[j] = P2[j]+1#=0的ci2 =1的归ci1
    index_1_1 = find_all_index(ci1,0) # find the empty positions in ci1 这是找ci1中等于0的
    index_1_2 = find_all_index_not(ci2,0) # find the positions in ci2 which is not zero 这是找ci1中不等于0的

    index_2_1 = find_all_index(ci2,0)#与上相反
    index_2_2 = find_all_index_not(ci1,0)
    l1=len(index_1_1);l2=len(index_2_1)
    for j in range(l1):
        ci1[index_1_1[j]] = NP2[index_1_2[j]]
    for j in range(l2):
        ci2[index_2_1[j]] = NP1[index_2_2[j]]
    l1 = len(index_2_2);l2 = len(index_1_2)
    for j in range(l1):
        ci1[index_2_2[j]] = ci1[index_2_2[j]]-1
    for j in range(l2):
        ci2[index_1_2[j]] = ci2[index_1_2[j]] - 1
    NP1=ci1
    NP2 =ci2
    return NP1,NP2

def PMX(P1,P2,N):
    #partially matching crossover operator (PMX) 1985
    #inital offerspring
    pos1 = math.floor(N/2-N/4)
    pos2 = math.floor(N-pos1)
    pos2=int(pos2)
    L = min(pos1, pos2)
    U = max(pos1, pos2)
    np1 = np.zeros(N,dtype=int)
    np2 = np.zeros(N,dtype=int)
    part1 = P1[L:U]

    part2 = P2[L:U]
    np1 = copy.copy(P1)
    np2 = copy.copy(P2)
    np1[L:U]=copy.copy(part2)
    try:
        np2[L:U]=copy.copy(part1)
    except:
        print('something wrong')
    tot=sum(i for i in range(N))
    for j in range(N):
        if j<L or j>U-1:
            x=find_all_index(part2,np1[j])
            if len(x)!=0:
                np1[j]=part1[x]
                x = find_all_index(part2, np1[j])
                while len(x)!=0:
                    np1[j] = part1[x]
                    x = find_all_index(part2, np1[j])

            x2 = find_all_index(part1, np2[j])
            if len(x2) != 0:
                np2[j] = part2[x2]
                x2 = find_all_index(part1, np2[j])
                while len(x2) != 0:
                    np2[j] = part2[x2]
                    x2 = find_all_index(part1, np2[j])
    return np1,np2

def UX_F(F1,F2,N,F):#可能没用
    nf1 = copy.copy(F1);nf2 = copy.copy(F2);
    #s = [random.random() for _ in range(N)]
    #s = mylistRound(s)
    for i in range(N):
        s=round(random.random()*1)
        if (s == 1):
            temp = copy.copy(nf1[i]);
            nf1[i] = copy.copy(nf2[i]);
            nf2[i] = copy.copy(temp);
    f_num = np.zeros(F,dtype=int);
    for f in range(F):
        f_num[f]= len(find_all_index(nf1, f));
    for f in range(F):
        if f_num[f]==0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = copy.copy(FC)
            random.shuffle(tmp2)
            nf1 = copy.copy(tmp2)

    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(nf2, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = copy.copy(FC)
            random.shuffle(tmp2)
            nf2= copy.copy(tmp2)
    return nf1,nf2

def mutation_p(p_chrom,N):
    #swap for operation sequence as mutation operator
    SH=N
    p1=math.floor(random.random()*N) #(0,19)
    p2 = math.floor(random.random() * N)
    while p1==p2:
        p2 = math.floor(random.random() * N)
    t = copy.copy(p_chrom[p1])
    p_chrom[p1] = copy.copy(p_chrom[p2])
    p_chrom[p2] = copy.copy(t);


    return p_chrom

def mutation_f(f_chrom,N,F):
    #swap for operation sequence as mutation operator
    pos1=int(np.floor(random.random()*N))
    cf=f_chrom[pos1]
    f=math.floor(random.random()*F)
    while cf==f:
        f=np.floor(random.random()*F)
        f=int(f)
        f_chrom[pos1]=f
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom= copy.copy(tmp2)

    return f_chrom

def NSGA2(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD,Length):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            [np1, np2] = PMX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(4);f2=np.zeros(4);
        f1[0], f1[1], f1[2], f1[3] = EnergySave_DHHFSPgai(np1, nf1, N, time, F, TS, NS, JP, JDD, Length)  # 到这
        f2[0], f2[1], f2[2], f2[3] = EnergySave_DHHFSPgai(np2, nf2, N, time, F, TS, NS, JP, JDD, Length)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]

    return p_chrom,f_chrom,fitness

def NSGA2POX(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            [np1, np2] = POX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(3);f2=np.zeros(3);
        f1[0],f1[1],f1[2] = FitDHHFSP(np1, nf1,N,time,F,TS,NS, JP, JDD)
        f2[0],f2[1],f2[2] = FitDHHFSP(np2, nf2,N,time,F,TS,NS, JP, JDD)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)#cp是np1操作顺序 cfit是是满意程度，功耗，最大期望顺序排列，cf是np1的机器选择
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))#一次循环后完成顺序以及选择机器
    QFit = np.vstack((fitness, CFit))#一次循环后完成目标值
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]

    return p_chrom,f_chrom,fitness

def NSGA2POXES(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD,Length):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        index = math.floor(random.random() * ps)#(0,99)
        P1 = Pool_P[index, :]#随机排列[1，20]
        F1 = Pool_F[index, :]#随机排列【0，1二十次】
        np1=copy.copy(p_chrom[i, :]);np2=copy.copy(P1);
        nf1 = copy.copy(f_chrom[i, :]);nf2 = copy.copy(F1);
        if random.random() < Pc:
            [np1, np2] = POX(p_chrom[i, :], P1, N)#np1 np2中间随机交叉互换.
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)#待观察
            if random.random() < Pm:
                np1 = mutation_p(np1,N) #p1p2 间某点再换 可不可以改变即几率突变为其他数据呢？
                nf1 = mutation_f(nf1,N,F)# 与Fchorm有关，待观察
            if random.random() < Pm:
                np2 = mutation_p(np2, N)   #再突变
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(4);f2=np.zeros(4);
        f1[0],f1[1],f1[2],f1[3] = EnergySave_DHHFSPgai(np1, nf1,N,time,F,TS,NS, JP, JDD, Length)#到这
        f2[0],f2[1],f2[2],f2[3] = EnergySave_DHHFSPgai(np2, nf2,N,time,F,TS,NS, JP, JDD, Length)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS4(QFit, 100)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]
    return p_chrom,f_chrom,fitness

def NSGA2PMXES(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            [np1, np2] = PMX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(3);f2=np.zeros(3);
        f1[0],f1[1],f1[2] = EnergySave_DHHFSP(np1, nf1,N,time,F,TS,NS, JP, JDD)
        f2[0],f2[1],f2[2] = EnergySave_DHHFSP(np2, nf2,N,time,F,TS,NS, JP, JDD)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]

    return p_chrom,f_chrom,fitness


def NSGA2MOX(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            if random.random()<0.5:
                [np1, np2] = POX(p_chrom[i, :], P1, N)
            else:
                [np1, np2] = PMX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(3);f2=np.zeros(3);
        f1[0],f1[1],f1[2] = FitDHHFSP(np1, nf1,N,time,F,TS,NS, JP, JDD)
        f2[0],f2[1],f2[2] = FitDHHFSP(np2, nf2,N,time,F,TS,NS, JP, JDD)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]

    return p_chrom,f_chrom,fitness

def MOEADPOX(p_chrom,f_chrom,index,T,neighbour,Pc,Pm,N,time,F,TS,NS, JP,JDD,Length):
    nei = neighbour[index, :]
    R1 = math.floor(random.random() * T)
    R1 = nei[R1]
    R2 = math.floor(random.random() * T)
    R2 = nei[R2]

    np1 = copy.copy(p_chrom[R1, :])
    np2 = copy.copy(p_chrom[R2, :])
    nf1 = copy.copy(f_chrom[R1, :])
    nf2= copy.copy(f_chrom[R2, :])
    while R1 == R2:
        R2 = math.floor(random.random() * T)
        R2 = nei[R2]
    if random.random() < Pc:
        [np1, np2] = PMX(p_chrom[R1, :], p_chrom[R2, :], N)
        [nf1, nf2] = UX_F(f_chrom[R1, :], f_chrom[R2, :], N, F)
        if random.random() < Pm:
            np1 = mutation_p(np1, N)
            nf1 = mutation_f(nf1, N, F)
        if random.random() < Pm:
            np2 = mutation_p(np2, N)
            nf2 = mutation_f(nf2, N, F)

        f1=np.zeros(4);f2=np.zeros(4);
        f1[0], f1[1], f1[2], f1[3] = EnergySave_DHHFSPgai(np1, nf1, N, time, F, TS, NS, JP, JDD, Length)  # 到这
        f2[0], f2[1], f2[2], f2[3] = EnergySave_DHHFSPgai(np2, nf2, N, time, F, TS, NS, JP, JDD, Length)

    return np1,nf1,f1,np2,nf2,f2


def NSGA3POXES(p_chrom, f_chrom, fitness, Pc, Pm, ps, N, time, F, TS, NS, JP, JDD, Length):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness, ps, N);
    CP = [];
    CF = [];
    CFit = []
    for i in range(ps):
        index = math.floor(random.random() * ps)  # (0,99)
        P1 = Pool_P[index, :]  # 随机排列[1，20]
        F1 = Pool_F[index, :]  # 随机排列【0，1二十次】
        np1 = copy.copy(p_chrom[i, :]);
        np2 = copy.copy(P1);
        nf1 = copy.copy(f_chrom[i, :]);
        nf2 = copy.copy(F1);
        if random.random() < Pc:
            [np1, np2] = POX(p_chrom[i, :], P1, N)  # np1 np2中间随机交叉互换.
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N, F)  # 待观察
            if random.random() < Pm:
                np1 = mutation_p(np1, N)  # p1p2 间某点再换 可不可以改变即几率突变为其他数据呢？
                nf1 = mutation_f(nf1, N, F)  # 与Fchorm有关，待观察
            if random.random() < Pm:
                np2 = mutation_p(np2, N)  # 再突变
                nf2 = mutation_f(nf2, N, F)
        f1 = np.zeros(4);
        f2 = np.zeros(4);
        f1[0], f1[1], f1[2], f1[3] = EnergySave_DHHFSP(np1, nf1, N, time, F, TS, NS, JP, JDD, Length)  # 到这
        f2[0], f2[1], f2[2], f2[3]= EnergySave_DHHFSP(np2, nf2, N, time, F, TS, NS, JP, JDD, Length)
        if len(CP) == 0:
            CP.append(np1);
            CFit.append(f1);
            CF.append(nf1)
            CP = np.vstack((CP, np2));
            CFit = np.vstack((CFit, f2));
            CF = np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));
            CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));
            CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));
            CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP, QF, QFit = DeleteReapt(QP, QF, QFit, ps)
    p_chrom, f_chrom = FastNDS_3(QP, QF, QFit, ps)
    return p_chrom, f_chrom

def NSGA3(p_chrom, f_chrom, fitness, Pc, Pm, ps, N, time, F, TS, NS, JP, JDD, Length):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness, ps, N);
    CP = [];
    CF = [];
    CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            [np1, np2] = PMX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N, F)
            if random.random() < Pm:
                np1 = mutation_p(np1, N)
                nf1 = mutation_f(nf1, N, F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1 = np.zeros(4);
        f2 = np.zeros(4);
        f1[0], f1[1], f1[2], f1[3] = EnergySave_DHHFSP(np1, nf1, N, time, F, TS, NS, JP, JDD, Length)  # 到这
        f2[0], f2[1], f2[2], f2[3] = EnergySave_DHHFSP(np2, nf2, N, time, F, TS, NS, JP, JDD, Length)
        if len(CP) == 0:
            CP.append(np1);
            CFit.append(f1);
            CF.append(nf1)
            CP = np.vstack((CP, np2));
            CFit = np.vstack((CFit, f2));
            CF = np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));
            CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));
            CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));
            CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP, QF, QFit = DeleteReapt(QP, QF, QFit, ps)
    p_chrom, f_chrom = FastNDS_3(QP, QF, QFit, ps)
    return p_chrom, f_chrom

def NSGA2SA(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD,Length,current_T,alpha = 0.95):

    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            [np1, np2] = PMX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(4);f2=np.zeros(4);
        f1[0], f1[1], f1[2], f1[3] = EnergySave_DHHFSPgai(np1, nf1, N, time, F, TS, NS, JP, JDD, Length)  # 到这
        f2[0], f2[1], f2[2], f2[3] = EnergySave_DHHFSPgai(np2, nf2, N, time, F, TS, NS, JP, JDD, Length)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]
    p_chrom, f_chrom, fitness = sa_local_search(p_chrom, f_chrom, fitness, ps, current_T,
                                   N, F, time, TS, NS, JP, JDD, Length)
    # ================== 温度更新 ==================
    current_T *= alpha

    return p_chrom,f_chrom,fitness,current_T
def sa_local_search(QP, QF, QFit, ps, T, N, F, time, TS, NS, JP, JDD, Length):
    """
    对临界层解进行SA优化
    """
    # 非支配排序
    fronts = FastNDS(QFit, len(QFit))

    # 仅对最后接受的临界层进行优化
    critical_front = fronts[-1]

    for idx in range(100):
        seq = QP[idx]
        fac = QF[idx]

        # SA局部搜索
        improved_seq, improved_fac = sa_mutation(
            seq, fac, N, F, T, Pm=0.3,  # 提高变异概率
            time=time, TS=TS, NS=NS, JP=JP, JDD=JDD, Length=Length,
            iterations=3
        )
        improved_f = np.zeros(4)
        improved_f[0],improved_f[1],improved_f[2],improved_f[3] = EnergySave_DHHFSPgai(improved_seq, improved_fac,
                                          N, time, F, TS, NS, JP, JDD, Length)

        # 替换原解
        if (improved_f <= QFit[idx]).all() and (improved_f < QFit[idx]).any():
            QP[idx] = improved_seq
            QF[idx] = improved_fac
            QFit[idx] = improved_f

    return QP, QF, QFit

def sa_mutation(seq, fac, N, F, T, Pm, time, TS, NS, JP, JDD, Length, iterations):
    """
    SA增强型变异：多阶段扰动+退火接受
    """
    best_seq = seq.copy()
    best_fac = fac.copy()
    best_f = np.zeros(4)
    best_f[0],best_f[1],best_f[2],best_f[3] = EnergySave_DHHFSPgai(best_seq, best_fac, N, time, F, TS, NS, JP, JDD, Length)

    for _ in range(iterations):
        # 生成变异解
        new_seq = mutation_p(seq, N)
        new_fac = mutation_f(fac, N, F)

        # 评估新解
        new_f = np.zeros(4)
        new_f[0],new_f[1],new_f[2],new_f[3] = EnergySave_DHHFSPgai(new_seq, new_fac, N, time, F, TS, NS, JP, JDD, Length)

        # 退火接受准则
        delta = (new_f - best_f).sum()
        if delta < 0 or random.random() < math.exp(-delta / T):
            best_seq = new_seq
            best_fac = new_fac
            best_f = new_f

    return best_seq, best_fac
# def MOEADPOXDRL(p_chrom,f_chrom,index,T,neighbour,Pc,Pm,N,time,F,TS,NS, JP,JDD,Length,k,fitness,NFEs):
#     nei = neighbour[index, :]
#     R1 = math.floor(random.random() * T)
#     R1 = nei[R1]
#     R2 = math.floor(random.random() * T)
#     R2 = nei[R2]
#
#     np1 = copy.copy(p_chrom[R1, :])
#     np2 = copy.copy(p_chrom[R2, :])
#     nf1 = copy.copy(f_chrom[R1, :])
#     nf2= copy.copy(f_chrom[R2, :])
#     k = 5
#     while R1 == R2:
#         R2 = math.floor(random.random() * T)
#         R2 = nei[R2]
#     if k ==0:
#         if random.random() < Pc:
#             [np1, np2] = POX(p_chrom[R1, :], p_chrom[R2, :], N)
#             [nf1, nf2] = UX_F(f_chrom[R1, :], f_chrom[R2, :], N, F)
#             if random.random() < Pm:
#                 np1 = mutation_p(np1, N)
#                 nf1 = mutation_f(nf1, N, F)
#             if random.random() < Pm:
#                 np2 = mutation_p(np2, N)
#                 nf2 = mutation_f(nf2, N, F)
#         NFEs = NFEs + 2
#     if k == 1:
#         np1, nf1 = DSwap(np1, nf1, fitness[R1], N, F, JDD, JP, Length)  # N3
#         np2, np2 = DSwap(np2, nf2, fitness[R2], N, F, JDD, JP, Length)
#         NFEs = NFEs + 2
#
#     # elif k == 1:
#     #     P1, F1 = Fswap1(AP[j, :], AF[j, :], N, F, JDD, JP,time, TS, NS)#N7
#     if k == 2:
#         np1, nf1 = PSwap(np1, nf1, fitness[R1], N, F, JDD, JP, Length)  # N1
#         np2, np2 = PSwap(np2, nf2, fitness[R2], N, F, JDD, JP, Length)
#         NFEs = NFEs + 2
#     # elif k == 3:
#     #     P1, F1 = Fswap2(AP[j, :], AF[j, :], N, F, JDD, JP,time, TS, NS)#N5
#     elif k == 3:
#         np1, nf1 = PInsert(np1, nf1, fitness[R1], N, F, JDD, JP, Length)  # N2
#         np2, np2 = PInsert(np2, nf2, fitness[R2], N, F, JDD, JP, Length)
#         NFEs = NFEs + 2
#     elif k == 4:
#         np1, nf1 = DInsert5(np1, nf1, fitness[R1], N, F, JDD, JP, Length)  # N4
#         np2, np2 = DInsert5(np2, nf2, fitness[R2], N, F, JDD, JP, Length)
#         NFEs = NFEs + 2
#     # elif k == 2:
#     #     P1, F1 = FInsert1(AP[j, :], AF[j, :], N, F, JDD, JP,time, TS, NS,Length)#N8
#     # elif k == 3:
#     #     P1, F1 = FInsert2(AP[j, :], AF[j, :], N, F, JDD, JP,time, TS, NS,Length)#N6
#     elif k == 5:
#         np1, nf1, NFEs = PSWAPNEW(np1, nf1,  N, F, JDD, JP, Length, NFEs)  # N9
#         np2, nf2, NFEs = PSWAPNEW(np2, nf2, N, F, JDD, JP, Length, NFEs)
#         np1 = np1.astype(int)
#         nf1 = nf1.astype(int)
#         np2 = np2.astype(int)
#         nf2 = nf2.astype(int)
#     elif k == 6:
#         np1, nf1, NFEs = PINSERTNEW(np1, nf1,  N, F, JDD, JP, Length, NFEs)  # N9
#         np2, nf2, NFEs = PINSERTNEW(np2, nf2,N, F, JDD, JP, Length, NFEs)  # N9
#         np1 = np1.astype(int)
#         nf1 = nf1.astype(int)
#         np2 = np2.astype(int)
#         nf2 = nf2.astype(int)
#
#     f1=np.zeros(4);f2=np.zeros(4);
#     f1[0], f1[1], f1[2], f1[3] = EnergySave_DHHFSP(np1, nf1, N, time, F, TS, NS, JP, JDD, Length)  # 到这
#     f2[0], f2[1], f2[2], f2[3] = EnergySave_DHHFSP(np2, nf2, N, time, F, TS, NS, JP, JDD, Length)
#
#     return np1,nf1,f1,np2,nf2,f2,NFEs

def POX1(P1,P2,N):#p的交叉互换
    #inital offerspring
    NP1=P1;
    NP2=P2;
    #index of each operation in P1 and P2
    ci1=np.zeros(N,dtype=int)
    ci2 = np.zeros(N, dtype=int)
    # store some jobs in J1
    temp=[random.random() for _ in range(N) ]
    temp=mylistRound(temp)
    J1=find_all_index(temp,1)#find the index where value equal to 1  J1返回temp中那几位是1
    for j in range(N):
        if Ismemeber(P1[j], J1)==1: #if is in job set J
            ci1[j] = P1[j]+1

        if Ismemeber(P2[j], J1)==0: #if is not in job set J
            ci2[j] = P2[j]+1#=0的ci2 =1的归ci1
    index_1_1 = find_all_index(ci1,0) # find the empty positions in ci1 这是找ci1中等于0的
    index_1_2 = find_all_index_not(ci2,0) # find the positions in ci2 which is not zero 这是找ci1中不等于0的

    index_2_1 = find_all_index(ci2,0)#与上相反
    index_2_2 = find_all_index_not(ci1,0)
    l1=len(index_1_1);l2=len(index_2_1)
    for j in range(l1):
        ci1[index_1_1[j]] = NP2[index_1_2[j]]
    for j in range(l2):
        ci2[index_2_1[j]] = NP1[index_2_2[j]]
    l1 = len(index_2_2);l2 = len(index_1_2)
    for j in range(l1):
        ci1[index_2_2[j]] = ci1[index_2_2[j]]-1
    for j in range(l2):
        ci2[index_1_2[j]] = ci2[index_1_2[j]] - 1
    NP1=ci1
    NP2 =ci2
    return NP1