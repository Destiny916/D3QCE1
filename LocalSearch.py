# coding:utf-8
import copy
import math
import random
from Tool import *
import numpy as np
from CalFitness import *

def Swap(p_chrom,N):
    #swap for operation sequence as mutation operator
    SH=N
    p1=math.floor(random.random()*N)
    p2 = math.floor(random.random() * N)
    while p1==p2:
        p2 = math.floor(random.random() * N)
    t = copy.copy(p_chrom[p1])
    p_chrom[p1] = copy.copy(p_chrom[p2])
    p_chrom[p2] = copy.copy(t);

    return p_chrom

def Insert(p_chrom,N):
    #swap for operation sequence as mutation operator
    SH=N
    pos1=math.floor(random.random()*N)
    pos2 = math.floor(random.random() * N)
    while pos1==pos2:
        pos2 = math.floor(random.random() * N)
    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)

    return p_chrom

def DInsert(p_chrom,f_chrom,N,F,JDD):
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
            f_chrom = copy.copy(tmp2)

    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    sf = math.floor(random.random() * F);
    SP = copy.copy(P[sf]);
    SL = len(SP);
    J1 = math.floor(random.random() * SL);
    pos1 = FJ[sf][J1];
    J1 = SP[J1];

    J2 = math.floor(random.random() * SL);
    pos2 = FJ[sf][J2];
    J2 = SP[J2];
    count = 0;
    while count < 10:
        if J1 == J2:
            J2 = math.floor(random.random() * SL);
            pos2 = FJ[sf][J2];
            J2 = SP[J2];
        else:
            if pos2 > pos1 and JDD[J2] < JDD[J1]:
                break;

            if pos2 < pos1 and JDD[J2] > JDD[J1]:
                break;

        count = count + 1;
    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)

    return p_chrom,f_chrom

def DInsert2(p_chrom,f_chrom,fitness,N,F,JDD):
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
            f_chrom = copy.copy(tmp2)

    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];
    count = 0;
    while count<10:
        if J1 == J2:
            J2 = math.floor(random.random() * (SL - posJ));
            pos2 = FJ[cf][J2];
            J2 = SP[J2];
        else:
            if JDD[J2] > JDD[J1]:
                break;
        count = count+1;

    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)

    return p_chrom,f_chrom

def DInsert3(p_chrom,f_chrom,fitness,N,F,JDD,JP):
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
            f_chrom = copy.copy(tmp2)

    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];

    for i in range(posJ,-1,-1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JDD[J2] > JDD[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);
        elif JP[J2] > JP[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);


    return p_chrom,f_chrom

# find the suitable place before critical job and swap them according to duedate first and priorities second在关键工作之前找到合适的位置，并根据日期第一、优先顺序第二的原则进行交换
def  DSwap(p_chrom,f_chrom,fitness,N,F,JDD,JP,length):
    maxj = int(fitness[3]);
    cf = f_chrom[maxj];
    P = [];
    JDD_t = JDD.copy()
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])
    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)
        JDD_t[t1] = JDD_t[t1] - length[t3][t1]
    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];
    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];
    for i in range(posJ,-1,-1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JDD_t[J2] > JDD_t[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);
            break;
        elif JDD_t[J2] == JDD_t[J1] and JP[J2] > JP[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);
            break;
    return p_chrom,f_chrom

# find the suitable place before critical job and insert the latter into the former according to duedate first and priorities second在关键工作之前找到合适的位置，并根据日期优先和优先级次，将后者插入前者中
def DInsert5(p_chrom,f_chrom,fitness,N,F,JDD,JP,length):
    '''
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
            f_chrom = copy.copy(tmp2)
    '''
    JDD_t = JDD.copy()
    maxj = int(fitness[3]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)
        JDD_t[t1] = JDD_t[t1] - length[t3][t1]

    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];

    for i in range(posJ,-1,-1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JDD_t[J2] > JDD_t[J1]:
            low = min(pos1, pos2)
            up = max(pos1, pos2)
            tmp = p_chrom[up];
            for i in range(up, low, -1):
                p_chrom[i] = copy.copy(p_chrom[i - 1])
            p_chrom[low] = copy.copy(tmp)
            break;
        elif JDD_t[J2] == JDD_t[J1] and JP[J2] > JP[J1]:
            low = min(pos1, pos2)
            up = max(pos1, pos2)
            tmp = p_chrom[up];
            for i in range(up, low, -1):
                p_chrom[i] = copy.copy(p_chrom[i - 1])
            p_chrom[low] = copy.copy(tmp)
            break;
    return p_chrom,f_chrom

# find the suitable place before critical job and insert the latter into the former according to duedate first and priorities second
def PInsert4(p_chrom,f_chrom,fitness,N,F,JDD,JP):
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
            f_chrom = copy.copy(tmp2)

    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];

    for i in range(posJ,-1,-1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JP[J2] > JP[J1]:
            low = min(pos1, pos2)
            up = max(pos1, pos2)
            tmp = p_chrom[up];
            for i in range(up, low, -1):
                p_chrom[i] = copy.copy(p_chrom[i - 1])
            p_chrom[low] = copy.copy(tmp)
            break;
        elif JP[J2] == JP[J1] and JDD[J2] > JDD[J1]:
            low = min(pos1, pos2)
            up = max(pos1, pos2)
            tmp = p_chrom[up];
            for i in range(up, low, -1):
                p_chrom[i] = copy.copy(p_chrom[i - 1])
            p_chrom[low] = copy.copy(tmp)
            break;
    return p_chrom,f_chrom

# find the suitable place before critical job and swap them according to priorities first and  duedate second
def PSwap(p_chrom,f_chrom,fitness,N,F,JDD,JP,length):
    '''
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
            f_chrom = copy.copy(tmp2)
    '''
    JDD_t = JDD.copy()
    maxj = int(fitness[3]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)
        JDD_t[t1] = JDD_t[t1] - length[t3][t1]

    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];
    for i in range(posJ,-1,-1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JP[J2] > JP[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);
            break;
        elif JP[J2] == JP[J1] and JDD_t[J2] > JDD_t[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);
            break;
    return p_chrom,f_chrom

def PInsert(p_chrom,f_chrom,fitness,N,F,JDD,JP,length):
    '''
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
            f_chrom = copy.copy(tmp2)
    '''
    JDD_t = JDD.copy()
    maxj = int(fitness[3]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)
        JDD_t[t1] = JDD_t[t1] - length[t3][t1]
    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ = find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL - posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];

    for i in range(posJ, -1, -1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JP[J2] > JP[J1]:
            break;
        elif JP[J2] == JP[J1] and JDD_t[J2] > JDD_t[J1]:
            break;

    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)
    return p_chrom,f_chrom

def PSWAPNEW1(AP, AF, N, F, JDD, JP):
    t_k = random.randint(1,F)
    t_0 = []
    t_1 = []
    t_2 = []
    J_1 = 0
    J_2 = 0
    J_3 = 0
    P = np.array([])
    for i in range(N):
        if AF[AP[i]] == 0:
            t_0.append(AP[i])
        elif AF[AP[i]] == 1:
            t_1.append(AP[i])
        elif AF[AP[i]] == 2:
            t_2.append(AP[i])
    if t_k == 1:
        for i in range(len(t_0)):
            if JP[t_0[i]] == 1:
                J_1 = J_1 + 1
            elif JP[t_0[i]] == 2:
                J_2 = J_2 + 1
            elif JP[t_0[i]] == 3:
                J_3 = J_3 + 1
        J_1 = 4 * J_1
        J_2 = 2 * J_2
        J_3 = 1 * J_3
        t_r = random.randint(1,J_1 + J_2 + J_3)
        if t_r <=J_3:
            for i in range(len(t_0)-1,-1,-1):
                if JP[t_0[i]] == 3:
                    for n in range(i):
                        if JP[t_0[n]] > JP[t_0[i]] or (JP[t_0[n]] == JP[t_0[i]] and JDD[t_0[n]] > JDD[t_0[i]] ):
                            t_p = copy.copy(t_0[n])
                            t_0[n] = copy.copy(t_0[i])
                            t_0[i] = copy.copy(t_p)
                    break
        elif t_r <= J_3 + J_2:
            for i in range(len(t_0)-1,-1,-1):
                if JP[t_0[i]] == 2:
                    for n in range(i):
                        if JP[t_0[n]] > JP[t_0[i]] or (JP[t_0[n]] == JP[t_0[i]] and JDD[t_0[n]] > JDD[t_0[i]] ):
                            t_p = copy.copy(t_0[n])
                            t_0[n] = copy.copy(t_0[i])
                            t_0[i] = copy.copy(t_p)
                    break
        elif t_r <= J_3 + J_2 +J_1:
            for i in range(len(t_0)-1,-1,-1):
                if JP[t_0[i]] == 1:
                    for n in range(i):
                        if JP[t_0[n]] > JP[t_0[i]] or (JP[t_0[n]] == JP[t_0[i]] and JDD[t_0[n]] > JDD[t_0[i]] ):
                            t_p = copy.copy(t_0[n])
                            t_0[n] = copy.copy(t_0[i])
                            t_0[i] = copy.copy(t_p)
                    break
        P = np.hstack((t_0, t_1, t_2))
        return P,AF
    if t_k == 2:
        for i in range(len(t_1)):
            if JP[t_1[i]] == 1:
                J_1 = J_1 + 1
            elif JP[t_1[i]] == 2:
                J_2 = J_2 + 1
            elif JP[t_1[i]] == 3:
                J_3 = J_3 + 1
        J_1 = 4 * J_1
        J_2 = 2 * J_2
        J_3 = 1 * J_3
        t_r = random.randint(1, (J_1 + J_2 + J_3))
        if t_r <=J_3:
            for i in range(len(t_1)-1,-1,-1):
                if JP[t_1[i]] == 3:
                    for n in range(i):
                        if JP[t_1[n]] > JP[t_1[i]] or (JP[t_1[n]] == JP[t_1[i]] and JDD[t_1[n]] > JDD[t_1[i]] ):
                            t_p = copy.copy(t_1[n])
                            t_1[n] = copy.copy(t_1[i])
                            t_1[i] = copy.copy(t_p)
                    break
        elif t_r <= J_3 + J_2:
            for i in range(len(t_1)-1,-1,-1):
                if JP[t_1[i]] == 2:
                    for n in range(i):
                        if JP[t_1[n]] > JP[t_1[i]] or (JP[t_1[n]] == JP[t_1[i]] and JDD[t_1[n]] > JDD[t_1[i]]):
                            t_p = copy.copy(t_1[n])
                            t_1[n] = copy.copy(t_1[i])
                            t_1[i] = copy.copy(t_p)
                    break
        elif t_r <= J_3 + J_2 +J_1:
            for i in range(len(t_1)-1,-1,-1):
                if JP[t_1[i]] == 1:
                    for n in range(i):
                        if JP[t_1[n]] > JP[t_1[i]] or (JP[t_1[n]] == JP[t_1[i]] and JDD[t_1[n]] > JDD[t_1[i]]):
                            t_p = copy.copy(t_1[n])
                            t_1[n] = copy.copy(t_1[i])
                            t_1[i] = copy.copy(t_p)
                    break
        P = np.hstack((t_0, t_1, t_2))
        return P, AF
    if t_k == 3:
        for i in range(len(t_2)):
            if JP[t_2[i]] == 1:
                J_1 = J_1 + 1
            elif JP[t_2[i]] == 2:
                J_2 = J_2 + 1
            elif JP[t_2[i]] == 3:
                J_3 = J_3 + 1
        J_1 = 4 * J_1
        J_2 = 2 * J_2
        J_3 = 1 * J_3
        t_r = random.randint(1,J_1 + J_2 + J_3)
        if t_r <=J_3:
            for i in range(len(t_2)-1,-1,-1):
                if JP[t_2[i]] == 3:
                    for n in range(i):
                        if JP[t_2[n]] > JP[t_2[i]] or (JP[t_2[n]] == JP[t_2[i]] and JDD[t_2[n]] > JDD[t_2[i]] ):
                            t_p = copy.copy(t_2[n])
                            t_2[n] = copy.copy(t_2[i])
                            t_2[i] = copy.copy(t_p)
                    break
        elif t_r <= J_3 + J_2:
            for i in range(len(t_2)-1,-1,-1):
                if JP[t_2[i]] == 2:
                    for n in range(i):
                        if JP[t_2[n]] > JP[t_2[i]] or (JP[t_2[n]] == JP[t_2[i]] and JDD[t_2[n]] > JDD[t_2[i]]):
                            t_p = copy.copy(t_2[n])
                            t_2[n] = copy.copy(t_2[i])
                            t_2[i] = copy.copy(t_p)
                    break
        elif t_r <= J_3 + J_2 +J_1:
            for i in range(len(t_2)-1,-1,-1):
                if JP[t_2[i]] == 1:
                    for n in range(i):
                        if JP[t_2[n]] > JP[t_2[i]] or (JP[t_2[n]] == JP[t_2[i]] and JDD[t_2[n]] > JDD[t_2[i]]):
                            t_p = copy.copy(t_2[n])
                            t_2[n] = copy.copy(t_2[i])
                            t_2[i] = copy.copy(t_p)
                    break
        P = np.hstack((t_0, t_1, t_2))
        return P, AF


def Fswap1(AP, AF, N, F, JDD, JP, time, TS, NS):  # 置换工件时，只能在其他工厂中寻找权值相同的且预期时间被置换的大于置换的，并且在选择工厂尽量时寻找多次加工时间和最短的
    # 将在AP，AF中选择同样的m个数
    t_0 = []
    t_1 = []
    t_2 = []
    for i in range(N):
        if AF[AP[i]] == 0:
            t_0.append(AP[i])
        elif AF[AP[i]] == 1:
            t_1.append(AP[i])
        elif AF[AP[i]] == 2:
            t_2.append(AP[i])
    NewP = np.array([])
    NewP = np.append(NewP, t_0)
    NewP = np.append(NewP, t_1)
    NewP = np.append(NewP, t_2)
    NewP = NewP.astype(int)
    t_0 = len(t_0)
    t_1 = len(t_1)
    t_2 = len(t_2)

    m = 1  # 选择的个数
    n_index = []  # 随机获取一个下标
    consumption, Satisfaction, Ajob = EnergySave_DHHFSP1(NewP, AF, N, time, F, TS, NS, JP, JDD, m)
    for i in range(len(Ajob)):
        for j in range(len(NewP)):
            if Ajob[i] == NewP[j]:  # 找到Ajob[i]在AP中的下标
                n_index.append(j)  # 将下标加入到n_index中
                break

    NewF = AF.copy()
    # 选择合适的工厂
    n_F = AF[AP[n_index[0]]]
    # while True:
    #     n_NewF = random.randint(0, F)
    #     if n_NewF != n_F:
    #         break
    f = F
    F = [i for i in range(f)]
    F.remove(n_F)
    n_NewF = random.choice(F)
    F.remove(n_NewF)

    def swap(n):
        if n == 0:
            for i in range(t_0):
                if (JDD[NewP[n_index[0]]] < JDD[NewP[i]]) or (
                        JDD[NewP[n_index[0]]] == JDD[NewP[i]] and JP[NewP[n_index[0]]] < JP[NewP[i]]):
                    t_P = NewP[n_index[0]].copy()
                    NewP[n_index[0]] = NewP[i].copy()
                    NewP[i] = t_P.copy()
                    NewF[NewP[n_index[0]]] = n
                    NewF[NewP[i]] = n_F
                    break
            if i == t_0 - 1 and F != []:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(n_NewF)
            if i == t_0 - 1 and F == []:
                for i in range(t_0 - 1, -1, -1):
                    if JDD[NewP[n_index[0]]] == JDD[NewP[i]] and JP[NewP[n_index[0]]] == JP[NewP[i]]:
                        t_P = NewP[n_index[0]].copy()
                        NewP[n_index[0]] = NewP[i].copy()
                        NewP[i] = t_P.copy()
                        NewF[NewP[n_index[0]]] = n
                        NewF[NewP[i]] = n_F
                        break
        elif n == 1:
            for i in range(t_0, t_0 + t_1):
                if (JDD[NewP[n_index[0]]] < JDD[NewP[i]]) or (
                        JDD[NewP[n_index[0]]] == JDD[NewP[i]] and JP[NewP[n_index[0]]] < JP[NewP[i]]):
                    t_P = NewP[n_index[0]].copy()
                    NewP[n_index[0]] = NewP[i].copy()
                    NewP[i] = t_P.copy()
                    NewF[NewP[n_index[0]]] = n
                    NewF[NewP[i]] = n_F
                    break
            if i == t_0 + t_1 - 1 and F != []:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(n_NewF)
            if i == t_0 + t_1 - 1 and F == []:
                for i in range(t_0 + t_1 - 1, t_0 - 1, -1):
                    if JDD[NewP[n_index[0]]] == JDD[NewP[i]] and JP[NewP[n_index[0]]] == JP[NewP[i]]:
                        t_P = NewP[n_index[0]].copy()
                        NewP[n_index[0]] = NewP[i].copy()
                        NewP[i] = t_P.copy()
                        NewF[NewP[n_index[0]]] = n
                        NewF[NewP[i]] = n_F
                        break
        elif n == 2:
            for i in range(t_0 + t_1, t_0 + t_1 + t_2):
                if (JDD[NewP[n_index[0]]] < JDD[NewP[i]]) or (
                        JDD[NewP[n_index[0]]] == JDD[NewP[i]] and JP[NewP[n_index[0]]] < JP[NewP[i]]):
                    t_P = NewP[n_index[0]].copy()
                    NewP[n_index[0]] = NewP[i].copy()
                    NewP[i] = t_P.copy()
                    NewF[NewP[n_index[0]]] = n
                    NewF[NewP[i]] = n_F
                    break
            if i == t_0 + t_1 + t_2 - 1 and F != []:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(n_NewF)
            if i == t_0 + t_1 - 1 and F == []:
                for i in range(t_0 + t_1 + t_2 - 1, t_0 + t_1 - 1, -1):
                    if JDD[NewP[n_index[0]]] == JDD[NewP[i]] and JP[NewP[n_index[0]]] == JP[NewP[i]]:
                        t_P = NewP[n_index[0]].copy()
                        NewP[n_index[0]] = NewP[i].copy()
                        NewP[i] = t_P.copy()
                        NewF[NewP[n_index[0]]] = n
                        NewF[NewP[i]] = n_F
                        break

    swap(n_NewF)

    # print('F')
    # print(TIME.time() - TIME3)
    return NewP, NewF


def Fswap2(AP, AF, N, F, JDD, JP, time, TS, NS):
    t_0 = []
    t_1 = []
    t_2 = []
    for i in range(N):
        if AF[AP[i]] == 0:
            t_0.append(AP[i])
        elif AF[AP[i]] == 1:
            t_1.append(AP[i])
        elif AF[AP[i]] == 2:
            t_2.append(AP[i])
    NewP = np.array([])
    NewP = np.append(NewP, t_0)
    NewP = np.append(NewP, t_1)
    NewP = np.append(NewP, t_2)
    NewP = NewP.astype(int)
    t_0 = len(t_0)
    t_1 = len(t_1)
    t_2 = len(t_2)

    m = 1  # 选择的个数
    n_index = []  # 随机获取一个下标
    consumption, Satisfaction, Ajob = EnergySave_DHHFSP1(NewP, AF, N, time, F, TS, NS, JP, JDD, m)
    for i in range(len(Ajob)):
        for j in range(len(NewP)):
            if Ajob[i] == NewP[j]:  # 找到Ajob[i]在AP中的下标
                n_index.append(j)  # 将下标加入到n_index中
                break

    NewF = AF.copy()

    # 选择合适的工厂
    n_F = AF[AP[n_index[0]]]
    # while True:
    #     n_NewF = random.randint(0, F)
    #     if n_NewF != n_F:
    #         break
    f = F
    F = [i for i in range(f)]
    F.remove(n_F)
    n_NewF = random.choice(F)
    F.remove(n_NewF)

    def swap(n):
        if n == 0:
            for i in range(t_0):
                if (JP[NewP[n_index[0]]] < JP[NewP[i]]) or (
                        JP[NewP[n_index[0]]] == JP[NewP[i]] and JDD[NewP[n_index[0]]] < JDD[NewP[i]]):
                    t_P = NewP[n_index[0]].copy()
                    NewP[n_index[0]] = NewP[i].copy()
                    NewP[i] = t_P.copy()
                    NewF[NewP[n_index[0]]] = n
                    NewF[NewP[i]] = n_F
                    break
            if i == t_0 - 1 and F != []:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(n_NewF)
            if i == t_0 - 1 and F == []:
                for i in range(t_0 - 1, -1, -1):
                    if JDD[NewP[n_index[0]]] == JDD[NewP[i]] and JP[NewP[n_index[0]]] == JP[NewP[i]]:
                        t_P = NewP[n_index[0]].copy()
                        NewP[n_index[0]] = NewP[i].copy()
                        NewP[i] = t_P.copy()
                        NewF[NewP[n_index[0]]] = n
                        NewF[NewP[i]] = n_F
                        break

        elif n == 1:
            for i in range(t_0, t_0 + t_1):
                if (JP[NewP[n_index[0]]] < JP[NewP[i]]) or (
                        JP[NewP[n_index[0]]] == JP[NewP[i]] and JDD[NewP[n_index[0]]] < JDD[NewP[i]]):
                    t_P = NewP[n_index[0]].copy()
                    NewP[n_index[0]] = NewP[i].copy()
                    NewP[i] = t_P.copy()
                    NewF[NewP[n_index[0]]] = n
                    NewF[NewP[i]] = n_F
                    break
            if i == t_0 + t_1 - 1 and F != []:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(n_NewF)
            if i == t_0 + t_1 - 1 and F == []:
                for i in range(t_0 + t_1 - 1, t_0 - 1, -1):
                    if JDD[NewP[n_index[0]]] == JDD[NewP[i]] and JP[NewP[n_index[0]]] == JP[NewP[i]]:
                        t_P = NewP[n_index[0]].copy()
                        NewP[n_index[0]] = NewP[i].copy()
                        NewP[i] = t_P.copy()
                        NewF[NewP[n_index[0]]] = n
                        NewF[NewP[i]] = n_F
                        break

        elif n == 2:
            for i in range(t_0 + t_1, t_0 + t_1 + t_2):
                if (JP[NewP[n_index[0]]] < JP[NewP[i]]) or (
                        JP[NewP[n_index[0]]] == JP[NewP[i]] and JDD[NewP[n_index[0]]] < JDD[NewP[i]]):
                    t_P = NewP[n_index[0]].copy()
                    NewP[n_index[0]] = NewP[i].copy()
                    NewP[i] = t_P.copy()
                    NewF[NewP[n_index[0]]] = n
                    NewF[NewP[i]] = n_F
                    break
            if i == t_0 + t_1 + t_2 - 1 and F != []:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(n_NewF)
            if i == t_0 + t_1 - 1 and F == []:
                for i in range(t_0 + t_1 + t_2 - 1, t_0 + t_1 - 1, -1):
                    if JDD[NewP[n_index[0]]] == JDD[NewP[i]] and JP[NewP[n_index[0]]] == JP[NewP[i]]:
                        t_P = NewP[n_index[0]].copy()
                        NewP[n_index[0]] = NewP[i].copy()
                        NewP[i] = t_P.copy()
                        NewF[NewP[n_index[0]]] = n
                        NewF[NewP[i]] = n_F
                        break

    swap(n_NewF)
    return NewP, NewF

def FInsert1(AP, AF, N, F, JDD, JP, time, TS, NS):
    # 将在AP，AF中选择同样的m个数
    m = 1  # 选择的个数
    n_index = []  # 随机获取一个下标
    consumption, Satisfaction, Ajob = EnergySave_DHHFSP1(AP, AF, N, time, F, TS, NS, JP, JDD, m)
    for i in range(len(Ajob)):
        for j in range(len(AP)):
            if Ajob[i] == AP[j]:  # 找到Ajob[i]在AP中的下标
                n_index.append(j)  # 将下标加入到n_index中
                break
    choice_P = AP[n_index[0]]
    AP = AP[AP != choice_P]
    t_0 = []
    t_1 = []
    t_2 = []
    for i in range(N - 1):
        if AF[AP[i]] == 0:
            t_0.append(AP[i])
        elif AF[AP[i]] == 1:
            t_1.append(AP[i])
        elif AF[AP[i]] == 2:
            t_2.append(AP[i])
    NewP = np.array([])
    NewP = np.append(NewP, t_0)
    NewP = np.append(NewP, t_1)
    NewP = np.append(NewP, t_2)
    NewP = NewP.astype(int)
    t_0 = len(t_0)
    t_1 = len(t_1)
    t_2 = len(t_2)
    NewF = AF.copy()
    # 选择合适的工厂
    n_F = NewF[choice_P]
    # while True:
    #     n_NewF = random.randint(0, F)
    #     if n_NewF != n_F:
    #         break
    f = F
    F = [i for i in range(f)]
    F.remove(n_F)
    n_NewF = random.choice(F)
    F.remove(n_NewF)

    def swap(NewP, n):
        NewP = list(NewP)
        if n == 0:
            for i in range(t_0):
                if (JDD[choice_P] < JDD[NewP[i]]) or (JDD[choice_P] == JDD[NewP[i]] and JP[choice_P] < JP[NewP[i]]):
                    NewP.insert(i, choice_P)
                    NewF[choice_P] = n
                    break
            if i == t_0 - 1 and F != [] and len(NewP) == N - 1:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(NewP, n_NewF)
            if i == t_0 - 1 and F == [] and len(NewP) == N - 1:
                NewP.insert(i + 1, choice_P)
                NewF[choice_P] = n

        elif n == 1:
            for i in range(t_0, t_0 + t_1):
                if (JDD[choice_P] < JDD[NewP[i]]) or (JDD[choice_P] == JDD[NewP[i]] and JP[choice_P] < JP[NewP[i]]):
                    NewP.insert(i, choice_P)
                    NewF[choice_P] = n
                    break
            if i == t_0 + t_1 - 1 and F != [] and len(NewP) == N - 1:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(NewP, n_NewF)
            if i == t_0 + t_1 - 1 and F == [] and len(NewP) == N - 1:
                if i == len(NewP) - 1:
                    NewP.append(choice_P)
                else:
                    NewP.insert(i + 1, choice_P)
                NewF[choice_P] = n

        elif n == 2:
            for i in range(t_0 + t_1, t_0 + t_1 + t_2):
                if (JDD[choice_P] < JDD[NewP[i]]) or (JDD[choice_P] == JDD[NewP[i]] and JP[choice_P] < JP[NewP[i]]):
                    NewP.insert(i, choice_P)
                    NewF[choice_P] = n
                    break
            if i == t_0 + t_1 + t_2 - 1 and F != [] and len(NewP) == N - 1:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(NewP, n_NewF)
            if i == t_0 + t_1 + t_2 - 1 and F == [] and len(NewP) == N - 1:
                NewP.append(choice_P)
                NewF[choice_P] = n

        NewP = np.array(NewP)
        return NewP

    NewP = swap(NewP, n_NewF)
    return NewP, NewF


def FInsert2(AP, AF, N, F, JDD, JP, time, TS, NS):
    # 将在AP，AF中选择同样的m个数
    m = 1  # 选择的个数
    n_index = []  # 随机获取一个下标
    consumption, Satisfaction, Ajob = EnergySave_DHHFSP1(AP, AF, N, time, F, TS, NS, JP, JDD, m)
    for i in range(len(Ajob)):
        for j in range(len(AP)):
            if Ajob[i] == AP[j]:  # 找到Ajob[i]在AP中的下标
                n_index.append(j)  # 将下标加入到n_index中
                break
    choice_P = AP[n_index[0]]
    AP = AP[AP != choice_P]
    t_0 = []
    t_1 = []
    t_2 = []
    for i in range(N - 1):
        if AF[AP[i]] == 0:
            t_0.append(AP[i])
        elif AF[AP[i]] == 1:
            t_1.append(AP[i])
        elif AF[AP[i]] == 2:
            t_2.append(AP[i])
    NewP = np.array([])
    NewP = np.append(NewP, t_0)
    NewP = np.append(NewP, t_1)
    NewP = np.append(NewP, t_2)
    NewP = NewP.astype(int)
    t_0 = len(t_0)
    t_1 = len(t_1)
    t_2 = len(t_2)
    NewF = AF.copy()
    # 选择合适的工厂
    n_F = NewF[choice_P]
    # while True:
    #     n_NewF = random.randint(0, F)
    #     if n_NewF != n_F:
    #         break
    f = F
    F = [i for i in range(f)]
    F.remove(n_F)
    n_NewF = random.choice(F)
    F.remove(n_NewF)

    def swap(NewP, n):
        NewP = list(NewP)
        if n == 0:
            for i in range(t_0):
                if (JP[choice_P] < JP[NewP[i]]) or (JP[choice_P] == JP[NewP[i]] and JDD[choice_P] < JDD[NewP[i]]):
                    NewP.insert(i, choice_P)
                    NewF[choice_P] = n
                    break
            if i == t_0 - 1 and F != [] and len(NewP) == N - 1:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(NewP, n_NewF)
            if i == t_0 - 1 and F == [] and len(NewP) == N - 1:
                NewP.insert(i + 1, choice_P)
                NewF[choice_P] = n

        elif n == 1:
            for i in range(t_0, t_0 + t_1):
                if (JP[choice_P] < JP[NewP[i]]) or (JP[choice_P] == JP[NewP[i]] and JDD[choice_P] < JDD[NewP[i]]):
                    NewP.insert(i, choice_P)
                    NewF[choice_P] = n

                    break
            if i == t_0 + t_1 - 1 and F != [] and len(NewP) == N - 1:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(NewP, n_NewF)
            if i == t_0 + t_1 - 1 and F == [] and len(NewP) == N - 1:
                if i == len(NewP) - 1:
                    NewP.append(choice_P)
                else:
                    NewP.insert(i + 1, choice_P)
                NewF[choice_P] = n

        elif n == 2:
            for i in range(t_0 + t_1, t_0 + t_1 + t_2):
                if (JP[choice_P] < JP[NewP[i]]) or (JP[choice_P] == JP[NewP[i]] and JDD[choice_P] < JDD[NewP[i]]):
                    NewP.insert(i, choice_P)
                    NewF[choice_P] = n

                    break
            if i == t_0 + t_1 + t_2 - 1 and F != [] and len(NewP) == N - 1:
                n_NewF = random.choice(F)
                F.remove(n_NewF)
                swap(NewP, n_NewF)
            if i == t_0 + t_1 + t_2 - 1 and F == [] and len(NewP) == N - 1:
                if i == len(NewP) - 1:
                    NewP.append(choice_P)
                else:
                    NewP.insert(i + 1, choice_P)
                NewF[choice_P] = n

        NewP = np.array(NewP)
        return NewP

    NewP = swap(NewP, n_NewF)
    return NewP, NewF


















# def FInsert2(p_chrom,f_chrom,fitness,N,F,JDD,JP):
#     f_num = np.zeros(F, dtype=int);
#     for f in range(F):
#         f_num[f] = len(find_all_index(f_chrom, f));
#     for f in range(F):
#         if f_num[f] == 0:
#             FC = np.zeros(N, dtype=int)
#             # generate operation sequence randomly
#             for i in range(N):
#                 FC[i] = i % F
#             tmp2 = FC
#             random.shuffle(tmp2)
#             f_chrom = copy.copy(tmp2)
#
#     maxj = int(fitness[2]);
#     cf = f_chrom[maxj];
#     P = [];
#     FJ = []
#     for f in range(F):
#         P.append([])
#         FJ.append([])
#
#     for i in range(N):
#         t1 = p_chrom[i]
#         t3 = f_chrom[t1]
#         P[t3].append(p_chrom[i])
#         FJ[t3].append(i)
#     sf1=cf
#     sf2 = math.floor(random.random() * F);
#     while sf1==sf2:
#         sf2 = math.floor(random.random() * F);
#     J1 = maxj
#     pos1 = find_all_index(p_chrom,J1)[0];
#     SP = copy.copy(P[sf2]);
#     SL = len(SP);
#
#     J2 = math.floor(random.random() * SL);
#     pos2 = FJ[sf2][J2];
#     J2 = SP[J2];
#     for i in range(SL-1,-1,-1):
#         J2 = i;
#         pos2 = FJ[sf2][J2];
#         J2 = SP[J2];
#         if JDD[J2] > JDD[J1]:
#             break;
#         elif JDD[J2] == JDD[J1] and JP[J2] > JP[J1]:
#             break;
#
#     low = min(pos1, pos2)
#     up = max(pos1, pos2)
#     tmp = p_chrom[up];
#     for i in range(up, low, -1):
#         p_chrom[i] = copy.copy(p_chrom[i - 1])
#     p_chrom[low] = copy.copy(tmp)
#     f_chrom[J1] = sf2;
#
#     f_num = np.zeros(F, dtype=int);
#     for f in range(F):
#         f_num[f] = len(find_all_index(f_chrom, f));
#     for f in range(F):
#         if f_num[f] == 0:
#             FC = np.zeros(N, dtype=int)
#             # generate operation sequence randomly
#             for i in range(N):
#                 FC[i] = i % F
#             tmp2 = FC
#             random.shuffle(tmp2)
#             f_chrom = copy.copy(tmp2)
#     return p_chrom,f_chrom
#
# def FInsert(p_chrom,f_chrom,fitness,N,F,JDD):
#     '''
#     f_num = np.zeros(F, dtype=int);
#     for f in range(F):
#         f_num[f] = len(find_all_index(f_chrom, f));
#     for f in range(F):
#         if f_num[f] == 0:
#             FC = np.zeros(N, dtype=int)
#             # generate operation sequence randomly
#             for i in range(N):
#                 FC[i] = i % F
#             tmp2 = FC
#             random.shuffle(tmp2)
#             f_chrom = copy.copy(tmp2)
#     '''
#     maxj = int(fitness[2]);
#     cf = f_chrom[maxj];
#     P = [];
#     FJ = []
#     for f in range(F):
#         P.append([])
#         FJ.append([])
#
#     for i in range(N):
#         t1 = p_chrom[i]
#         t3 = f_chrom[t1]
#         P[t3].append(p_chrom[i])
#         FJ[t3].append(i)
#     sf1=cf
#     sf2 = math.floor(random.random() * F);
#     while sf1==sf2:
#         sf2 = math.floor(random.random() * F);
#     J1 = maxj
#     pos1 = find_all_index(p_chrom,J1)[0];
#     SP = copy.copy(P[sf2]);
#     SL = len(SP);
#     J2 = math.floor(random.random() * SL);
#     pos2 = FJ[sf2][J2];
#     J2 = SP[J2];
#     count = 0;
#     while count < 10:
#         if J1 == J2:
#             J2 = math.floor(random.random() * SL);
#             pos2 = FJ[sf2][J2];
#             J2 = SP[J2];
#         else:
#             if JDD[J2] > JDD[J1]:
#                 break;
#         count = count + 1;
#     low = min(pos1, pos2)
#     up = max(pos1, pos2)
#     tmp = p_chrom[up];
#     for i in range(up, low, -1):
#         p_chrom[i] = copy.copy(p_chrom[i - 1])
#     p_chrom[low] = copy.copy(tmp)
#     f_chrom[J1] = sf2;
#
#     f_num = np.zeros(F, dtype=int);
#     for f in range(F):
#         f_num[f] = len(find_all_index(f_chrom, f));
#     for f in range(F):
#         if f_num[f] == 0:
#             FC = np.zeros(N, dtype=int)
#             # generate operation sequence randomly
#             for i in range(N):
#                 FC[i] = i % F
#             tmp2 = FC
#             random.shuffle(tmp2)
#             f_chrom = copy.copy(tmp2)
#     return p_chrom,f_chrom
#
# def FSwap2(p_chrom,f_chrom,fitness,N,F,JDD):
#     f_num = np.zeros(F, dtype=int);
#     for f in range(F):
#         f_num[f] = len(find_all_index(f_chrom, f));
#     for f in range(F):
#         if f_num[f] == 0:
#             FC = np.zeros(N, dtype=int)
#             # generate operation sequence randomly
#             for i in range(N):
#                 FC[i] = i % F
#             tmp2 = FC
#             random.shuffle(tmp2)
#             f_chrom = copy.copy(tmp2)
#
#
#     P = [];
#     FJ = []
#     for f in range(F):
#         P.append([])
#         FJ.append([])
#
#     for i in range(N):
#         t1 = p_chrom[i]
#         t3 = f_chrom[t1]
#         P[t3].append(p_chrom[i])
#         FJ[t3].append(i)
#
#     sf1= math.floor(random.random() * F);
#     sf2 = math.floor(random.random() * F);
#     while sf1==sf2:
#         sf2 = math.floor(random.random() * F);
#     SP = copy.copy(P[sf1]);
#     SL = len(SP);
#     J1 = math.floor(random.random() * SL);
#     pos1 = FJ[sf1][J1];
#     J1 = SP[J1];
#     SP = copy.copy(P[sf2]);
#     SL = len(SP);
#     J2 = math.floor(random.random() * SL);
#     pos2 = FJ[sf2][J2];
#     J2 = SP[J2];
#     count = 0;
#     while count < 10:
#         if J1 == J2:
#             J2 = math.floor(random.random() * SL);
#             pos2 = FJ[sf2][J2];
#             J2 = SP[J2];
#         else:
#             if pos2 < pos1 and JDD[J2] > JDD[J1]:
#                 break;
#
#             if pos2 > pos1 and JDD[J2] < JDD[J1]:
#                 break;
#         count = count + 1;
#     t = copy.copy(p_chrom[pos1])
#     p_chrom[pos1] = copy.copy(p_chrom[pos2])
#     p_chrom[pos2] = copy.copy(t);
#
#     f_chrom[J2] = sf1;
#     f_chrom[J1] = sf2;
#
#     f_num = np.zeros(F, dtype=int);
#     for f in range(F):
#         f_num[f] = len(find_all_index(f_chrom, f));
#     for f in range(F):
#         if f_num[f] == 0:
#             FC = np.zeros(N, dtype=int)
#             # generate operation sequence randomly
#             for i in range(N):
#                 FC[i] = i % F
#             tmp2 = FC
#             random.shuffle(tmp2)
#             f_chrom = copy.copy(tmp2)
#     return p_chrom,f_chrom
