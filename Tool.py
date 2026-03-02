# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial.distance import cdist
import math

def mymin(arr):
    index=[]
    l=len(arr)
    if l>1:
        minvalue=min(arr)
        for i in range(l):
            if arr[i]==minvalue:
                index.append(i)
    else:
        index.append(0)
    return index

def mylistRound(arr):
    l=len(arr)
    for i in range(l):
        arr[i]=round(arr[i])
    return arr

def find_all_index(arr, item):
    return [i for i, a in enumerate(arr) if a == item]

def find_all_index_not(arr, item):
    l=len(arr)
    flag=np.zeros(l)
    index=find_all_index(arr,item)
    flag[index]=1
    not_index=find_all_index(flag,0)
    return not_index

def NDS(fit1,fit2):
    v=0
    dom_less = 0;
    dom_equal = 0;
    dom_more = 0;
    for k in range(3):
        if fit1[k] > fit2[k]:
            dom_more = dom_more + 1;
        elif fit1[k] == fit2[k]:
            dom_equal = dom_equal + 1;
        else:
            dom_less = dom_less + 1;

    if dom_less == 0 and dom_equal != 2:
        v = 2
    if dom_more == 0 and dom_equal != 2:
        v = 1
    return v

def Ismemeber(item,list):
    l=len(list)
    flag=0
    for i in range(l):
        if list[i]==item:
            flag=1
            break
    return flag

def DeleteReapt(QP,QF,QFit,ps):
    row=np.size(QFit,0)
    i=0
    while i<row:
        if i>=row:
            #print('break 1')
            break

        F=QFit[i,:]
        j=i+1
        while j<row:
            if QFit[j][0] == F[0] and QFit[j][1] == F[1] and QFit[j][2] == F[2] and QFit[j][3] == F[3]:
                QP = np.delete(QP, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                j=j-1
                row=row-1
                if row<2*ps+1:
                    break
            j=j+1
        i=i+1
        if row < 2 * ps + 1:
            #print('break 2')
            break
    return QP,QF,QFit

def DeleteReaptE(QP,QF,QFit): #for elite strategy
     row = np.size(QFit,0)
     i=0
     while i<row:
         if i>=row:
             print('break 1')
             break
         F=QFit[i,:]
         j=i+1
         while j<row:
             if QFit[j][0]==F[0] and QFit[j][1]==F[1]:
                 QP = np.delete(QP, j, axis=0)
                 QF = np.delete(QF, j, axis=0)
                 QFit = np.delete(QFit, j, axis=0)
                 j=j-1
                 row=row-1
             j=j+1
         i=i+1
     return QP,QF,QFit



def DeleteReaptACOMOEAD(QP,QF,QFit,QT): #for elite strategy
    row=np.size(QFit,0)
    i=0
    while i<row:
        if i>=row:
            #print('break 1')
            break

        F=QFit[i,:]
        j=i+1
        while j<row:
            if QFit[j][0]==F[0] and QFit[j][1]==F[1]:
                QP = np.delete(QP, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                QT = np.delete(QT, j, axis=0)
                j=j-1
                row=row-1
            j=j+1
        i=i+1

    return QP,QF,QFit,QT

def DeleteReaptE2(QP,QF,QFit,Fnum): #for elite strategy
    row=np.size(QFit,0)
    i=0
    while i<row:
        if i>=row:
            #print('break 1')
            break

        F=QFit[i,:]
        j=i+1
        while j<row:
            if QFit[j][0]==F[0] and QFit[j][1]==F[1]:
                QP = np.delete(QP, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                j=j-1
                row=row-1
            f_num = np.zeros(Fnum, dtype=int);
            for f in range(Fnum):
                f_num[f] = len(find_all_index(QF[j,:], f));
            for f in range(Fnum):
                if f_num[f] == 0:
                    QP = np.delete(QP, j, axis=0)
                    QF = np.delete(QF, j, axis=0)
                    QFit = np.delete(QFit, j, axis=0)
                    j = j - 1
                    row = row - 1
            j=j+1
        i=i+1

    return QP,QF,QFit

def pareto(fitness):
    PF=[]
    L=np.size(fitness,axis=0)
    pn=np.zeros(L,dtype=int)
    for i in range(L):
        for j in range(L):
            dom_less = 0;
            dom_equal = 0;
            dom_more = 0;
            for k in range(2):#number of objectives
                if (fitness[i][k] > fitness[j][k]):
                    dom_more = dom_more + 1
                elif(fitness[i][k] == fitness[j][k]):
                    dom_equal = dom_equal + 1
                else:
                    dom_less = dom_less + 1

            if dom_less == 0 and dom_equal != 2: # i is dominated by j
                pn[i] = pn[i] + 1;
        if pn[i] == 0: # add i into pareto front
            PF.append(i)
    return PF


def pareto4(fitness):
    # A = [0,1,3,4]
    PF=[]
    L=np.size(fitness,axis=0)
    pn=np.zeros(L,dtype=int)
    for i in range(L):
        for j in range(L):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for k in range(3):#number of objectives
                if (fitness[i][k] > fitness[j][k]):
                    dom_more = dom_more + 1
                elif(fitness[i][k] == fitness[j][k]):
                    dom_equal = dom_equal + 1
                else:
                    dom_less = dom_less + 1
            if dom_less == 0 and dom_equal != 3: # i is dominated by j
                pn[i] = pn[i] + 1
        if pn[i] == 0: # add i into pareto front
            PF.append(i)
    return PF
def DeleteReaptE4(QP, QF, QFit):  # for elite strategy
    row = np.size(QFit, 0)
    i = 0
    while i < row:
        if i >= row:
            print('break 1')
            break
        F = QFit[i, :]
        j = i + 1
        while j < row:
            if QFit[j][0] == F[0] and QFit[j][1] == F[1] and QFit[j][2] == F[2]:#if QFit[j][0] == F[0] and QFit[j][1] == F[1] and QFit[j][2] == F[2] and QFit[j][3] == F[3]:
                QP = np.delete(QP, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                j = j - 1
                row = row - 1
            j = j + 1
        i = i + 1
    return QP, QF, QFit
def NDS4(fit1,fit2):
    v=0
    # A = [0,1,3,4]
    dom_less = 0;
    dom_equal = 0;
    dom_more = 0;
    for k in range(3):
        if fit1[k] > fit2[k]:
            dom_more = dom_more + 1;
        elif fit1[k] == fit2[k]:
            dom_equal = dom_equal + 1;
        else:
            dom_less = dom_less + 1;
    if dom_less == 0 and dom_equal != 4:
        v = 2
    if dom_more == 0 and dom_equal != 4:
        v = 1
    return v
def initialize_moead_params(n_subproblems, n_objectives=3, T_ratio=0.15):
    """
    初始化MOEA/D参数：权重向量、邻居关系、邻域大小
    :param n_subproblems: 子问题数量（种群大小）
    :param n_objectives: 目标函数数量
    :param T_ratio: 邻域大小占比（默认取种群大小的15%）
    :return: weights, neighbour, T
    """
    # ===== 生成三目标均匀权重向量 =====
    from pymoo.util.reference_direction import UniformReferenceDirectionFactory
    ref_dirs = UniformReferenceDirectionFactory(n_dim=n_objectives, n_partitions=15).do()
    weights = ref_dirs[:n_subproblems]  # 截取所需数量的权重

    # ===== 计算邻居关系矩阵 =====
    # 方法1：余弦相似度（适合目标空间方向分布）
    # similarity = 1 - cdist(weights, weights, metric='cosine')

    # 方法2：欧氏距离（默认选择）
    distances = cdist(weights, weights, 'euclidean')
    neighbour = np.argsort(distances, axis=1)  # 每行按距离从小到大排序的索引

    # ===== 确定邻域大小T =====
    T = max(2, int(n_subproblems * T_ratio))  # 至少保留2个邻居

    return weights, neighbour, T