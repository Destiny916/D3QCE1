# coding:utf-8
#f= open("../DATASET/J10M5O5.txt", "r",encoding='utf-8')
import numpy as np
import sys

def DataReadDHHJSP(Filepath):
    data = []
    enter='\n'
    # for example "../DATASET/J10M5O5.txt"

    with open(Filepath, "r",encoding='utf-8') as f1:
        for line in f1:
            temp = line.split(' ');
            l = len(temp)
            for i in range(l):
                if temp[i] != enter:
                    data.append(int(temp[i]))
    N = data[0]#20 列数
    TS = data[1]# 3 行
    F = data[2]# 2 两组数据
    NS = np.zeros(TS, dtype=int)
    p=3
    for s in range(TS):
        NS[s] = data[p];# 3 2 3
        p = p + 1;
    p = p + 1;#p=7
    time = np.zeros(shape=(F, TS, N))
    for f in range(F):
        for s in range(TS):
            for i in range(N):
                time[f][s][i]=data[p]
                p = p + 1;
        p = p + 1;

    p = p - 1;
    JP = np.zeros(N, dtype=int)
    JDD = np.zeros(N)
    for i in range(N):
        JP[i] = data[p];
        p = p + 1;

    for i in range(N):
        JDD[i] = data[p];
        p = p + 1;
    # f1.close()

    print(data)
    print(p)
    print(len(data))
    Length = []
    for _ in range(F):
        Length.append(data[p])
        p = p + 1;
    f1.close()


    return N,TS,F,NS,time,JP,JDD,Length

'''
data = f.read()
N=int(data[1])
TM=int(data[3])
H=np.zeros(1,N)
print(N)
print(TM)'''
def DataReadDHHJSP1(Filepath):
    data = []
    enter='\n'
    # for example "../DATASET/J10M5O5.txt"

    with open(Filepath, "r",encoding='utf-8') as f1:
        for line in f1:
            temp = line.split(' ');
            l = len(temp)
            for i in range(l):
                if temp[i] != enter:
                    data.append(int(temp[i]))
    N = data[0]#20 列数
    TS = data[1]# 3 行
    F = data[2]# 2 两组数据
    NS = np.zeros(TS, dtype=int)
    p=3
    for s in range(TS):
        NS[s] = data[p];# 3 2 3
        p = p + 1;
    p = p + 1;#p=7
    time = np.zeros(shape=(F, TS, N))
    for f in range(F):
        for s in range(TS):
            for i in range(N):
                time[f][s][i]=data[p]
                p = p + 1;
        p = p + 1;

    p = p - 1;
    JP = np.zeros(N, dtype=int)
    JDD = np.zeros(N)
    for i in range(N):
        JP[i] = data[p];
        p = p + 1;

    for i in range(N):
        JDD[i] = data[p];
        p = p + 1;
    # f1.close()

    print(data)
    print(p)
    print(len(data))
    Length = []
    for _ in range(N):
        Length.append(data[p])
        p = p + 1;
    Length1 = []
    for _ in range(N):
        Length1.append(data[p])
        p = p + 1;
    Length2 = []
    for _ in range(N):
        Length2.append(data[p])
        p = p + 1;
    Length = np.array(Length)
    Length1 = np.array(Length1)
    Length2 = np.array(Length2)
    Length = np.vstack((Length,Length1))
    Length = np.vstack((Length, Length2))
    f1.close()


    return N,TS,F,NS,time,JP,JDD,Length

def ReadP1F1(directory_path):
    """
    读取 P1.txt 和 F1.txt 文件
    将浮点数转换为整数数组
    
    参数:
        directory_path: 目录路径，例如 "d:\\daima\\pythonProject\\venv\\python\\25219\\D2QCE-TASE-2023py\\CPTEST2\\last1\\last\\DHHFSP01"
    
    返回:
        P: P1.txt 的整数数组
        F: F1.txt 的整数数组
    """
    import os
    
    P1_path = os.path.join(directory_path, "P1.txt")
    F1_path = os.path.join(directory_path, "F1.txt")
    
    P = []
    F = []
    
    if os.path.exists(P1_path):
        with open(P1_path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    row = [int(float(x)) for x in parts]
                    P.append(row)
    
    if os.path.exists(F1_path):
        with open(F1_path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    row = [int(float(x)) for x in parts]
                    F.append(row)
    
    P = np.array(P, dtype=int)
    F = np.array(F, dtype=int)
    
    return P, F

