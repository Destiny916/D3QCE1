# coding:utf-8
import copy
import random

import numpy as np

def FitHFSP(p_chrom,FJ,f_index,TS, time ,NS, JP, JDD):
    #processing power and idle power
    N=len(FJ)
    finish = np.zeros((N, TS));
    start = np.zeros((N, TS));
    workpower = 0;totalidletime = 0;
    W_power = 4;Idle_power = 1;
    for k in range(TS):
        mftime = np.zeros(NS[k]);s = k;
        for i in range(N):
            t = p_chrom[i];
            if k == 0:
                if i==0:
                    start[i][k] = 0;
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[0] = finish[i][k]
                else:
                    m_index = np.argmin(mftime); #找到最小完工时间的机器
                    start[i][k] = mftime[m_index]
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[m_index] = finish[i][k]
                workpower = workpower + time[f_index][s][t] * W_power;
            else:
                if i == 0:
                    start[i][k] = finish[i][k-1]
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[0] = finish[i][k]
                else:
                    m_index = np.argmin(mftime)
                    start[i][k] = max(finish[i][k-1], mftime[m_index]);
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    totalidletime = totalidletime + start[i][k] - mftime[m_index]
                    mftime[m_index] = finish[i][k]
                workpower = workpower + time[f_index][s][t] * W_power;
                    #start decoding

    try:
        Cmax = finish[N - 1][TS - 1];
    except:
        Cmax = 0;
    TEC = workpower + totalidletime * Idle_power;
    DueData_Validate = 0;
    Custom_satisified = 0;
    MaxDue = 0;MaxJob = 1;
    for i in range(N):
        x = finish[i][TS - 1] - JDD[p_chrom[i]];
        DueData_Validate = max(0, x);
        if DueData_Validate > 0:
            temp = 0;
            p = JP[p_chrom[i]];
            if p == 1:
                temp = DueData_Validate * 2;
                # 必保的项目，死命令的项目，如果超过交货期则不可接受，那么客户满意度为无穷大，这里为10000
                Custom_satisified = Custom_satisified + temp;
            elif p == 2:
                temp = DueData_Validate * 1;
                Custom_satisified = Custom_satisified + temp;
            elif p == 3:
                temp = DueData_Validate * 0.5;
                Custom_satisified = Custom_satisified + temp;
            if MaxDue < temp:
                MaxDue = temp;
                MaxJob = p_chrom[i];

    return Cmax,TEC,Custom_satisified,MaxJob,MaxDue

def FitDHHFSP(p_chrom,f_chrom,N,time,F,TS,NS, JP, JDD):
    P0=[];P=[];FJ=[]
    for f in range(F):
        P.append([])
        FJ.append([])
    for i in range(N):
        t1=p_chrom[i]
        t3=f_chrom[t1]
        P[t3].append(p_chrom[i])#是（0，19）随机排列
        FJ[t3].append(i)#是（0，19）
    sub_f_fit=np.zeros(shape=(F,5))
    for f in range(F):
        sub_f_fit[f][0],sub_f_fit[f][1],sub_f_fit[f][2],sub_f_fit[f][3],sub_f_fit[f][4]=FitHFSP(P[f],FJ[f],f,TS, time ,NS, JP, JDD)
    fit1=sub_f_fit[0][0]
    fit3 = 1;fit2 = 0;fit4 = 0;
    for f in range(F):
        fit2 = sub_f_fit[f][1] + fit2;
        fit4 = sub_f_fit[f][2] + fit4;
        if fit1 < sub_f_fit[f][0]:
            fit1 = sub_f_fit[f][0]
            fit3 = f;
    index= np.argmax(sub_f_fit[:, 4]);
    fit5 = sub_f_fit[index][3];
    return fit4,fit2,fit5

def RightShift(p_chrom,FJ,f_index,TS, time ,NS, JP, JDD):
    #processing power and idle power
    N=len(FJ)
    finish = np.zeros((N, TS));
    start = np.zeros((N, TS));
    workpower = 0;totalidletime = 0;
    W_power = 4;Idle_power = 1;
    ms = np.zeros((N, TS));
    for k in range(TS):#顺序
        mftime = np.zeros(NS[k]);s = k;
        for i in range(N):
            t = p_chrom[i];#   t是工件是谁
            if k == 0:
                if i==0:
                    start[i][k] = 0;
                    finish[i][k] = start[i][k] + time[f_index][s][t];#工件完成到他那的时间
                    mftime[0] = finish[i][k]
                    ms[i][k] = 0;
                else:
                    m_index = np.argmin(mftime); #找到最小完工时间的机器
                    start[i][k] = mftime[m_index]
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[m_index] = finish[i][k]
                    ms[i][k] = m_index;  #机器的选择
                workpower = workpower + time[f_index][s][t] * W_power; #选择一个工厂中的工件所需要加工的时间的总和*Wpower
            else:
                if i == 0:
                    start[i][k] = finish[i][k-1]
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[0] = finish[i][k]
                    ms[i][k] = 0;
                else:
                    m_index = np.argmin(mftime)
                    start[i][k] = max(finish[i][k-1], mftime[m_index]);
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    totalidletime = totalidletime + start[i][k] - mftime[m_index]
                    mftime[m_index] = finish[i][k]
                    ms[i][k] = m_index;
                workpower = workpower + time[f_index][s][t] * W_power;
                    #start decoding
    try:
        Cmax = finish[N-1][TS-1];
    except:
        Cmax=0;
    finish2 = copy.copy(finish);
    start2 = copy.copy(start);
    Idletime2 = np.zeros((N, TS));
    totalidletime2 = 0;
    for k in range (TS-1,-1,-1):
        mstime = np.zeros(NS[k]);
        s = k;
        for i in range (N-1,-1,-1):
            t = p_chrom[i];
            if k == TS-1:
                if i == N-1:
                    cms = ms[i][k];#i号工件第k次加工选择的机器
                    cms=int(cms)
                    mstime[cms] = start2[i][k];#i号工件第k次加工开始时间
                else:
                    cms = ms[i][k];
                    cms = int(cms)
                    if mstime[cms] == 0:#还未赋值最后几个工件机器的开始时间
                        mstime[cms] = start2[i][k];
                    else:
                        if finish[i][k] < mstime[cms] and finish[i][k] < JDD[p_chrom[i]]:
                            finish2[i][k] = min(mstime[cms], JDD[p_chrom[i]]);
                            start2[i][k] = finish2[i][k] - time[f_index][s][t];
                            totalidletime2 = totalidletime2 + mstime[cms] - finish2[i][k]
                            Idletime2[i][k] = mstime[cms] - finish2[i][k];
                            mstime[cms] = start2[i][k]
            else:
                if i == N-1:
                    cms = ms[i][k];cms=int(cms)
                    mstime[cms] = start2[i][k];
                else:
                    cms = ms[i][k];cms=int(cms)
                    if mstime[cms] == 0:
                        mstime[cms] = start2[i][k];
                    else:
                        finish2[i][k] = min(mstime[cms], start2[i][k + 1]);
                        start2[i][k] = finish2[i][k] - time[f_index][s][t];
                        totalidletime2 = totalidletime2 + mstime[cms] - finish2[i][k];
                        Idletime2[i][k] = mstime[cms] - finish2[i][k];
                        mstime[cms] = start2[i][k]

    TEC = workpower + totalidletime * Idle_power;
    TEC2= workpower + totalidletime2 * Idle_power;
    DueData_Validate = 0;
    Custom_satisified = 0;
    MaxDue = 0;MaxJob = 1;
    for i in range(N):
        x = finish[i][TS-1]-JDD[p_chrom[i]];
        DueData_Validate = max(0, x);
        if DueData_Validate > 0:
            temp=0;
            p = JP[p_chrom[i]];
            if p == 1:
                temp = DueData_Validate * 2;
                # 必保的项目，死命令的项目，如果超过交货期则不可接受，那么客户满意度为无穷大，这里为10000
                Custom_satisified = Custom_satisified + temp;
            elif p==2:
                temp = DueData_Validate * 1;
                Custom_satisified = Custom_satisified + temp;
            elif p == 3:
                temp = DueData_Validate * 0.5;
                Custom_satisified = Custom_satisified + temp;
            if MaxDue < temp:
                MaxDue = temp;
                MaxJob = p_chrom[i]
    if N == 0:
        finish_time = 0
    else:
        finish_time = finish2[N - 1][TS - 1];
    return Cmax,TEC2,Custom_satisified,MaxJob,MaxDue,finish_time


def EnergySave_DHHFSP(p_chrom,f_chrom,N,time,F,TS,NS, JP, JDD, Lenth):
    P0=[];P=[];FJ=[]
    for f in range(F):
        P.append([])
        FJ.append([])
    for i in range(N):
        t1=p_chrom[i]
        t3=f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)
    #分为P【0】为一部分（0，19）P【1】为一部分（0，19）
    sub_f_fit=np.zeros(shape=(F,6))
    for f in range(F):
        JDD = np.array(JDD)
        Lenth = np.array(Lenth)
        sub_f_fit[f][0], sub_f_fit[f][1], sub_f_fit[f][2], sub_f_fit[f][3], sub_f_fit[f][4], sub_f_fit[f][5] = RightShift(P[f], FJ[f], f, TS, time, NS, JP, (JDD - Lenth[f]))
    fit1 = sub_f_fit[0][0]
    fit3 = 1;fit2 = 0;fit4 = 0;
    for f in range(F):
        fit2 = sub_f_fit[f][1] + fit2;#功耗
        fit4 = sub_f_fit[f][2] + fit4;#用户满意度
        if fit1 < sub_f_fit[f][0]:
            fit1 = sub_f_fit[f][0]
            fit3 = f;
    index= np.argmax(sub_f_fit[:, 4]);
    fit5 = sub_f_fit[index][3];
    time = max(sub_f_fit[:, 5])
    Money = 0
    for i in range(len(P)):
        Money += float(len(P[i]) * Lenth[i])

    return fit4, fit2, Money, fit5


def NewRightShift(p_chrom, FJ, f_index, TS, time, NS, JP, JDD,m):
    # processing power and idle power
    N = len(FJ)
    finish = np.zeros((N, TS));
    start = np.zeros((N, TS));
    workpower = 0;
    totalidletime = 0;
    W_power = 4;
    Idle_power = 1;
    ms = np.zeros((N, TS));
    for k in range(TS):  # 顺序
        mftime = np.zeros(NS[k]);
        s = k;
        for i in range(N):
            t = p_chrom[i];  # t是工件是谁
            if k == 0:
                if i == 0:
                    start[i][k] = 0;
                    finish[i][k] = start[i][k] + time[f_index][s][t];  # 工件完成到他那的时间
                    mftime[0] = finish[i][k]
                    ms[i][k] = 0;
                else:
                    m_index = np.argmin(mftime);  # 找到最小完工时间的机器
                    start[i][k] = mftime[m_index]
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[m_index] = finish[i][k]
                    ms[i][k] = m_index;  # 机器的选择
                workpower = workpower + time[f_index][s][t] * W_power;  # 选择一个工厂中的工件所需要加工的时间的总和*Wpower
            else:
                if i == 0:
                    start[i][k] = finish[i][k - 1]
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[0] = finish[i][k]
                    ms[i][k] = 0;
                else:
                    m_index = np.argmin(mftime)
                    start[i][k] = max(finish[i][k - 1], mftime[m_index]);
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    totalidletime = totalidletime + start[i][k] - mftime[m_index]
                    mftime[m_index] = finish[i][k]
                    ms[i][k] = m_index;
                workpower = workpower + time[f_index][s][t] * W_power;
                # start decoding
    try:
        Cmax = finish[N - 1][TS - 1];
    except:
        Cmax = 0;
        print(N, FJ)
    finish2 = copy.copy(finish);
    start2 = copy.copy(start);
    Idletime2 = np.zeros((N, TS));
    totalidletime2 = 0;
    for k in range(TS - 1, -1, -1):
        mstime = np.zeros(NS[k]);
        s = k;
        for i in range(N - 1, -1, -1):
            t = p_chrom[i];
            if k == TS - 1:
                if i == N - 1:
                    cms = ms[i][k];  # i号工件第k次加工选择的机器
                    cms = int(cms)
                    mstime[cms] = start2[i][k];  # i号工件第k次加工开始时间
                else:
                    cms = ms[i][k];
                    cms = int(cms)
                    if mstime[cms] == 0:  # 还未赋值最后几个工件机器的开始时间
                        mstime[cms] = start2[i][k];
                    else:
                        if finish[i][k] < mstime[cms] and finish[i][k] < JDD[p_chrom[i]]:
                            finish2[i][k] = min(mstime[cms], JDD[p_chrom[i]]);
                            start2[i][k] = finish2[i][k] - time[f_index][s][t];
                            totalidletime2 = totalidletime2 + mstime[cms] - finish2[i][k]
                            Idletime2[i][k] = mstime[cms] - finish2[i][k];
                            mstime[cms] = start2[i][k]
            else:
                if i == N - 1:
                    cms = ms[i][k];
                    cms = int(cms)
                    mstime[cms] = start2[i][k];
                else:
                    cms = ms[i][k];
                    cms = int(cms)
                    if mstime[cms] == 0:
                        mstime[cms] = start2[i][k];
                    else:
                        finish2[i][k] = min(mstime[cms], start2[i][k + 1]);
                        start2[i][k] = finish2[i][k] - time[f_index][s][t];
                        totalidletime2 = totalidletime2 + mstime[cms] - finish2[i][k];
                        Idletime2[i][k] = mstime[cms] - finish2[i][k];
                        mstime[cms] = start2[i][k]
    TEC = workpower + totalidletime * Idle_power;
    TEC2 = workpower + totalidletime2 * Idle_power;
    DueData_Validate = 0;
    Custom_satisified = 0;
    AngryJob = []
    for i in range(N):
        x = finish[i][TS - 1] - JDD[p_chrom[i]];
        DueData_Validate = max(0, x);
        if DueData_Validate > 0:
            temp = 0;
            p = JP[p_chrom[i]];
            if p == 1:
                temp = DueData_Validate * 2;
                # 必保的项目，死命令的项目，如果超过交货期则不可接受，那么客户满意度为无穷大，这里为10000
                Custom_satisified = Custom_satisified + temp;
            elif p == 2:
                temp = DueData_Validate * 1;
                Custom_satisified = Custom_satisified + temp;
            elif p == 3:
                temp = DueData_Validate * 0.5;
                Custom_satisified = Custom_satisified + temp;
            # if MaxDue < temp:
            #     MaxDue = temp;
            #     MaxJob = p_chrom[i];
            t_job = [p_chrom[i], temp]
            AngryJob.append(t_job)
            AngryJob.sort(key=lambda x: x[1], reverse=True)
            AngryJob = AngryJob[:m]
    AngryJob=np.array(AngryJob)
    return Cmax, TEC2, Custom_satisified, AngryJob

def EnergySave_DHHFSP1(p_chrom,f_chrom,N,time,F,TS,NS, JP, JDD,m):
    P0=[];P=[];FJ=[]
    for f in range(F):
        P.append([])
        FJ.append([])
    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)
    #分为P【0】为一部分（0，19）P【1】为一部分（0，19）
    sub_f_fit=np.zeros(shape=(F,3))
    Ajob=np.array([])
    for f in range(F):
        tjob=np.array([])
        sub_f_fit[f][0],sub_f_fit[f][1],sub_f_fit[f][2],tjob=NewRightShift(P[f],FJ[f],f,TS, time ,NS, JP, JDD,m)
        if Ajob.size==0:
            Ajob=tjob.copy()
        else:
            if tjob.size!=0:
                Ajob=np.vstack((Ajob,tjob))
    fit1=sub_f_fit[0][0]
    fit3 = 1;fit2 = 0;fit4 = 0;
    fit5=[]
    for i in range(len(Ajob)):
        fit5.append(list(Ajob[i]))
    fit5.sort(key=lambda x:x[1],reverse=True)
    fit5 = fit5[:m]
    fit5 = [row[0] for row in fit5]
    if len(fit5)<m:
        for j in range(m-len(fit5)):
            while True:
                tfit5=random.randint(0,N-1)
                if tfit5 not in fit5:
                    fit5.append(tfit5)
                    break
    return fit4,fit2,fit5


def EnergySave_DHHFSPgai(p_chrom, f_chrom, N, time, F, TS, NS, JP, JDD, Lenth):
    P0 = [];
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
    # 分为P【0】为一部分（0，19）P【1】为一部分（0，19）
    sub_f_fit = np.zeros(shape=(F, 6))
    for f in range(F):
        JDD = np.array(JDD)
        Lenth = np.array(Lenth)
        sub_f_fit[f][0], sub_f_fit[f][1], sub_f_fit[f][2], sub_f_fit[f][3], sub_f_fit[f][4], sub_f_fit[f][
            5] = RightShift(P[f], FJ[f], f, TS, time, NS, JP, (JDD - Lenth[f]))
    fit1 = sub_f_fit[0][0]
    fit2 = 0;
    fit4 = 0;
    for f in range(F):
        fit2 = sub_f_fit[f][1] + fit2;  # 功耗
        fit4 = sub_f_fit[f][2] + fit4;  # 用户满意度
        if fit1 < sub_f_fit[f][0]:
            fit1 = sub_f_fit[f][0]
    index = np.argmax(sub_f_fit[:, 4]);
    fit5 = sub_f_fit[index][3];

    Money = 0
    # for i in range(len(P)):
    #     Money += float(len(P[i]) * Lenth[i])
    for i in range(N):
        Money += float(Lenth[f_chrom[i]][i])
    return fit4, fit2, Money, fit5