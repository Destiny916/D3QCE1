import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from DataRead import DataReadDHHJSP1
def has_time_conflict(factory_idx, stage_idx, machine_idx, new_start, new_finish, current_schedule, current_i, current_k):
    """
    检测新的时间区间是否与现有调度冲突
    
    参数:
        factory_idx: 工厂索引
        stage_idx: 阶段索引
        machine_idx: 机器索引
        new_start: 新的开始时间
        new_finish: 新的结束时间
        current_schedule: 当前调度列表
        current_i: 当前工件索引
        current_k: 当前阶段索引
    
    返回:
        bool: 如果存在冲突返回True，否则返回False
    """
    for item in current_schedule:
        f, s, m, job, start, end = item
        # 检查是否是同一工厂、同一阶段、同一机器
        if f == factory_idx and s == stage_idx and m == machine_idx:
            # 检查时间区间是否重叠（不包括当前工件的其他阶段）
            # 时间区间重叠的条件：not (new_finish <= start or new_start >= end)
            if not (new_finish <= start or new_start >= end):
                return True
    return False

def dual_factory_scheduling(pchrom, fchrom, time, JDD=None, NS=None):
    """
    多工厂调度算法实现，包含右移优化
    
    参数:
    - pchrom: 工件序列，例如 [1, 2, 3, 4, 5, 6]
    - fchrom: 每个工件分配的工厂，例如 [1, 1, 1, 2, 2, 2]
    - time: 处理时间数组，格式为 [工厂数][阶段数][机器数][工件数] 或 [[stage1_machines], [stage2_machines]]
           如 [3][5][3][12] 表示3个工厂，5个阶段，每阶段3台机器，12个工件
    - JDD: 交货期数组，可选参数
    - NS: 每阶段机器数数组，例如 [3, 3, 3, 3, 3] 表示每个阶段3台机器
    
    返回:
    - schedule: 调度计划列表，每个元素为 [factory, stage, machine, job, start_time, end_time]
    """
    # 参数初始化
    N = len(pchrom)  # 工件数量
    F = len(set(fchrom))  # 工厂数量
    TS = len(time[0])  # 阶段数
    
    # 检测时间矩阵格式
    # 如果time[0][0]是列表，则为新格式 [f][s][m][j]，否则为旧格式 [f][s][j]
    is_new_format = isinstance(time[0][0], list) and isinstance(time[0][0][0], list)
    
    if is_new_format:
        # 新格式：[工厂数][阶段数][机器数][工件数]
        if NS is None:
            NS = [len(time[0][s]) for s in range(TS)]  # 根据time矩阵自动确定每阶段机器数
        else:
            # 确保NS长度与阶段数匹配
            if len(NS) != TS:
                raise ValueError(f"NS数组长度({len(NS)})必须等于阶段数({TS})")
    else:
        # 旧格式：[工厂数][阶段数][工件数]，默认每阶段2台机器
        NS = [2] * TS if NS is None else NS
        # 将旧格式转换为新格式
        new_time = []
        for f in range(F):
            factory_data = []
            for s in range(TS):
                stage_data = []
                for m in range(NS[s]):  # 每个阶段NS[s]台机器
                    # 为每台机器分配相同的时间（从旧格式复制）
                    machine_times = time[f][s][:]
                    stage_data.append(machine_times)
                factory_data.append(stage_data)
            new_time.append(factory_data)
        time = new_time  # 替换为新格式
    
    # 如果没有提供交货期，默认为空数组
    if JDD is None:
        JDD = [float('inf')] * N
    
    # 将工件按工厂分组
    P = [[] for _ in range(F)]
    FJ = [[] for _ in range(F)]
    
    for i in range(N):
        t1 = pchrom[i]
        t3 = fchrom[t1]  # 修正：使用工件编号作为索引，与CalFitness.py保持一致
        P[t3].append(pchrom[i])
        FJ[t3].append(i)
    
    # 为每个工厂分别进行调度
    all_schedule = []
    
    for f_idx in range(F):
        factory_jobs = P[f_idx]
        job_indices = FJ[f_idx]
        
        if not factory_jobs:
            continue
            
        n_jobs = len(factory_jobs)
        n_stages = TS
        
        # 初始化调度参数
        finish = np.zeros((n_jobs, n_stages))
        start = np.zeros((n_jobs, n_stages))
        machines = np.zeros((n_jobs, n_stages))  # 记录每个工序使用的机器
        
        # 前向调度
        for k in range(n_stages):  # 对每个阶段
            mftime = np.zeros(NS[k])  # 每台机器的完工时间
            s = k  # 当前阶段
            
            for i in range(n_jobs):  # 对每个工件
                job_idx = factory_jobs[i] - 1  # 转换为0索引
                
                if k == 0:  # 第一阶段
                    if i == 0:  # 第一个工件
                        # 选择最早完工的机器
                        proc_times = [time[f_idx][s][m][job_idx] for m in range(NS[k])]
                        m_index = np.argmin(proc_times)
                        start[i][k] = 0
                        finish[i][k] = start[i][k] + time[f_idx][s][m_index][job_idx]
                        mftime[m_index] = finish[i][k]
                        machines[i][k] = m_index  # 使用选中的机器
                    else:
                        # 找到最早完工的机器
                        m_index = np.argmin(mftime)
                        proc_time = time[f_idx][s][m_index][job_idx]
                        start[i][k] = mftime[m_index]
                        finish[i][k] = start[i][k] + proc_time
                        mftime[m_index] = finish[i][k]
                        machines[i][k] = m_index  # 记录使用的机器
                else:  # 后续阶段
                    if i == 0:  # 第一个工件
                        # 选择最早完工的机器
                        proc_times = [time[f_idx][s][m][job_idx] for m in range(NS[k])]
                        m_index = np.argmin(proc_times)
                        start[i][k] = finish[i][k-1]
                        finish[i][k] = start[i][k] + time[f_idx][s][m_index][job_idx]
                        mftime[m_index] = finish[i][k]
                        machines[i][k] = m_index  # 使用选中的机器
                    else:
                        m_index = np.argmin(mftime)
                        proc_time = time[f_idx][s][m_index][job_idx]
                        start[i][k] = max(finish[i][k-1], mftime[m_index])
                        finish[i][k] = start[i][k] + proc_time
                        mftime[m_index] = finish[i][k]
                        machines[i][k] = m_index  # 记录使用的机器
        
        # 右移优化 (反向调度) - 修正版：添加冲突检测
        finish2 = np.copy(finish)
        start2 = np.copy(start)
        Idletime2 = np.zeros((n_jobs, n_stages))
        totalidletime2 = 0
        
        # 构建临时调度列表用于冲突检测
        temp_schedule = []
        for i in range(n_jobs):
            for k in range(n_stages):
                job_num = factory_jobs[i]
                factory_num = f_idx
                stage_num = k + 1
                machine_num = int(machines[i][k]) + 1
                start_time = start[i][k]
                end_time = finish[i][k]
                temp_schedule.append([factory_num + 1, stage_num, machine_num, job_num, start_time, end_time])
        
        for k in range(n_stages-1, -1, -1):  # 从最后一个阶段向前处理
            mstime = np.zeros(NS[k])  # 每台机器的开始时间
            s = k  # 当前阶段
            
            for i in range(n_jobs-1, -1, -1):  # 从最后一个工件向前处理
                job_idx = factory_jobs[i] - 1  # 转换为0索引
                cms = int(machines[i][k])  # 当前工序使用的机器
                factory_num = f_idx + 1
                stage_num = k + 1
                machine_num = cms + 1
                
                if k == n_stages-1:  # 最后一个阶段
                    if i == n_jobs-1:  # 最后一个工件
                        mstime[cms] = start2[i][k]  # 记录机器开始时间
                    else:
                        if mstime[cms] == 0:  # 机器尚未记录开始时间
                            mstime[cms] = start2[i][k]
                        else:
                            # 修正：添加冲突检测
                            # 如果当前完工时间小于机器记录的开始时间且小于交货期
                            if finish[i][k] < mstime[cms] and finish[i][k] < JDD[factory_jobs[i]-1]:
                                new_finish = min(mstime[cms], JDD[factory_jobs[i]-1])
                                proc_time = time[f_idx][s][cms][job_idx]
                                new_start = new_finish - proc_time
                                
                                # 检测是否会产生时间冲突
                                if not has_time_conflict(factory_num, stage_num, machine_num, new_start, new_finish, temp_schedule, i, k):
                                    finish2[i][k] = new_finish
                                    start2[i][k] = new_start
                                    totalidletime2 = totalidletime2 + mstime[cms] - finish2[i][k]
                                    Idletime2[i][k] = mstime[cms] - finish2[i][k]
                                    mstime[cms] = new_start
                                    
                                    # 更新临时调度列表
                                    for idx, item in enumerate(temp_schedule):
                                        if item[0] == factory_num and item[1] == stage_num and item[2] == machine_num and item[3] == factory_jobs[i]:
                                            temp_schedule[idx][4] = new_start
                                            temp_schedule[idx][5] = new_finish
                                # 如果有冲突，则不进行右移
                else:  # 非最后一个阶段
                    if i == n_jobs-1:  # 最后一个工件
                        mstime[cms] = start2[i][k]
                    else:
                        if mstime[cms] == 0:  # 机器尚未记录开始时间
                            mstime[cms] = start2[i][k]
                        else:
                            # 修正：添加冲突检测
                            # 取机器空闲时间和后续工序开始时间的最小值
                            new_finish = min(mstime[cms], start2[i][k + 1])
                            proc_time = time[f_idx][s][cms][job_idx]
                            new_start = new_finish - proc_time
                            
                            # 检测是否会产生时间冲突
                            if not has_time_conflict(factory_num, stage_num, machine_num, new_start, new_finish, temp_schedule, i, k):
                                finish2[i][k] = new_finish
                                start2[i][k] = new_start
                                totalidletime2 = totalidletime2 + mstime[cms] - finish2[i][k]
                                Idletime2[i][k] = mstime[cms] - finish2[i][k]
                                mstime[cms] = new_start
                                
                                # 更新临时调度列表
                                for idx, item in enumerate(temp_schedule):
                                    if item[0] == factory_num and item[1] == stage_num and item[2] == machine_num and item[3] == factory_jobs[i]:
                                        temp_schedule[idx][4] = new_start
                                        temp_schedule[idx][5] = new_finish
                            # 如果有冲突，则不进行右移
        
        # 使用优化后的调度时间
        start = start2
        finish = finish2
        
        # 构建调度计划
        for i in range(n_jobs):
            for k in range(n_stages):
                job_num = factory_jobs[i]
                factory_num = f_idx
                stage_num = k + 1  # 阶段从1开始计数
                machine_num = int(machines[i][k]) + 1  # 机器从1开始计数
                start_time = start[i][k]
                end_time = finish[i][k]
                
                all_schedule.append([factory_num + 1, stage_num, machine_num, job_num, start_time, end_time])
    
    return all_schedule

def create_gantt_chart(schedule):
    """
    创建甘特图
    """
    if not schedule:
        print("调度结果为空，无法生成甘特图")
        return
    
    fig, ax = plt.subplots(figsize=(18, max(10, len(schedule)*0.4)), facecolor='white')
    
    # 获取工件数量，定义颜色
    max_jobs = max(item[3] for item in schedule) if schedule else 10
    colors = {}
    base_colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#FFD700', '#FFA07A', '#DDA0DD', 
                   '#9370DB', '#4682B4', '#32CD32', '#FF6347', '#40E0D0', '#FF1493']
    for i in range(1, max_jobs + 1):
        colors[i] = base_colors[(i-1) % len(base_colors)]
    
    # 获取唯一的工厂、阶段、机器组合
    unique_combinations = set()
    for item in schedule:
        factory, stage, machine, job, start, end = item
        unique_combinations.add((factory, stage, machine))
    
    # 按照工厂、阶段、机器排序
    sorted_combinations = sorted(list(unique_combinations))
    
    # 为每个组合分配Y坐标
    y_positions = {}
    for idx, (factory, stage, machine) in enumerate(sorted_combinations):
        y_positions[(factory, stage, machine)] = len(sorted_combinations) - idx - 0.5
    
    # 绘制调度块
    for item in schedule:
        factory, stage, machine, job, start, end = item
        y_pos = y_positions.get((factory, stage, machine), 0)
        
        # 绘制条形图
        ax.barh(y=y_pos, width=end-start, left=start, height=0.6, 
                color=colors.get(job, '#CCCCCC'), edgecolor='black', 
                linewidth=0.8, alpha=0.9, zorder=2)
        
        # 添加工件编号
        ax.text((start+end)/2, y_pos, f'J{job}-{factory}-{stage}-{machine}', va='center', ha='center',
                color='black', fontsize=10, fontweight='bold', zorder=3)
    
    # 设置Y轴标签
    y_labels = [f'F{f}-S{s}-M{m}' for f, s, m in sorted_combinations]
    y_ticks = [y_positions[combo] for combo in sorted_combinations]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0)
    
    # 设置X轴
    max_time = max(item[5] for item in schedule) if schedule else 10
    ax.set_xlim(0, max_time * 1.1)
    ax.set_xlabel('Time (hours)', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    
    # 设置标题
    ax.set_title('Multi-Factory Production Schedule', fontsize=18, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    plt.show()

def calc_ypos(factory, stage, machine):
    """计算垂直位置（Y轴从上到下排列）"""
    factory_offset = (2 - factory) * 2.5
    stage_offset = (2 - stage) * 1.2
    machine_offset = (2 - machine) * 0.4
    return factory_offset + stage_offset + machine_offset

def main():
    # 示例输入数据
    pchrom = []  # 工件序列
    fchrom = []  # 工厂分配 (0-indexed)
    # 时间矩阵: [工厂0[[阶段1机器], [阶段2机器]], 工厂1[[阶段1机器], [阶段2机器]]]

    FILENAME = ['20J3S2F.txt', '20J5S2F.txt', '20J3S3F.txt', '20J5S3F.txt', \
                '40J3S2F.txt', '40J5S2F.txt', '40J3S3F.txt', '40J5S3F.txt', \
                '60J3S2F.txt', '60J5S2F.txt', '60J3S3F.txt', '60J5S3F.txt', \
                '80J3S2F.txt', '80J5S2F.txt', '80J3S3F.txt', '80J5S3F.txt', \
                '100J3S2F.txt', '100J5S2F.txt', '100J3S3F.txt', '100J5S3F.txt'];
    # FILENAME = ['20J3S2F.txt', '20J5S2F.txt', '20J3S3F.txt', '20J5S3F.txt', \
    #             '40J3S2F.txt', '40J5S2F.txt', '40J3S3F.txt', '40J5S3F.txt', \
    #             '60J3S2F.txt', '60J5S2F.txt', '60J3S3F.txt', '60J5S3F.txt', \
    #             '80J3S2F.txt', '80J5S2F.txt', '80J3S3F.txt', '80J5S3F.txt', \
    #             '100J3S2F.txt', '100J5S2F.txt', '100J3S3F.txt', '100J5S3F.txt'];
    # parameter

    filenum = 1;
    DATAPATH = 'data/DATASET_DHHFSP1/';
    INSPATH = [];
    RESPATH = [];
    for file in range(filenum):
        temp = DATAPATH + FILENAME[file];
        INSPATH.append(temp)
        if file < 9:
            id = '0' + str(file + 1);
        else:
            id = str(file + 1);
        temp2 = 'DHHFSP' + id + '\\'
        RESPATH.append(temp2)
    for file in range(filenum):
        N, TS, F, NS, time, JP, JDD, Length = DataReadDHHJSP1(INSPATH[file])  # N行数，TS列数，F几组数据 NS 3 2 3 time具体数据 jp 倒数第二 jdd倒数第一
        print("输入数据:")
        print(f"工件序列 pchrom: {pchrom}")
        print(f"工厂分配 fchrom: {fchrom}")
        print(f"处理时间 time: {time}")
    
    # 执行调度
        schedule = dual_factory_scheduling(pchrom, fchrom, time)
    
        print("\n调度结果(含右移优化):")
        print("[工厂号, 阶段号, 机器号, 工件号, 开始时间, 结束时间]")
        for item in schedule:
            print(item)
    
        print(f"\n最终输出数组（共{len(schedule)}项）：")
        print("格式：[工厂号, 阶段号, 机器号, 工件号, 开始时间, 结束时间]")
        print("示例：[1,1,1,1,0,2] 表示工件1在工厂1的阶段1机器1上从时间0开始到时间2结束")
        print("实际输出:")
        print(schedule)

        # 创建甘特图
        print("\n正在生成甘特图...")
        create_gantt_chart(schedule)