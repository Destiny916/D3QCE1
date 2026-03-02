import numpy as np

def HV(FIT):
    """
    计算超体积
    FIT: 目标值矩阵，每行是一个解的目标向量
    """
    if len(FIT) == 0:
        return 0.0
    
    # 参考点设置为所有解的最大值
    ref_point = np.max(FIT, axis=0)
    
    # 计算每个解的贡献体积
    total_volume = 0.0
    sorted_indices = np.argsort(FIT[:, 0])
    sorted_fit = FIT[sorted_indices]
    
    for i in range(len(sorted_fit)):
        point = sorted_fit[i]
        
        # 计算这个点的贡献体积
        contrib_vol = 1.0
        for j in range(point.shape[0]):
            contrib_vol *= (ref_point[j] - point[j])
        
        # 减去被其他解支配的部分
        for j in range(i):
            dominated = True
            for k in range(point.shape[0]):
                if sorted_fit[j][k] > point[k]:
                    dominated = False
                    break
            if dominated:
                overlap = 1.0
                for k in range(point.shape[0]):
                    overlap *= (ref_point[k] - max(point[k], sorted_fit[j][k]))
                contrib_vol -= overlap
        
        total_volume += contrib_vol
    
    return total_volume


import numpy as np


def hv(population):
    hv = hypervolume(population)
    point = [1,1,1,1]
    score = hv.compute(point)
    return score