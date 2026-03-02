import numpy as np
from judgement import *
from IGD import *
def CPTEST2DataRead(filename):
    filedata = []
    for i in range(1, 12):
        DHHFSP_data = []
        if i < 10:
            id='0'+str(i);
        else:
            id=str(i);
        for j in range(1, 11):
            # print('CPTEST2\\' + filename + '\\DHHFSP' + id + '\\res' + str(j) + '.txt')
            file = open('CPTEST2\\' + filename + '\\DHHFSP' + id + '\\res' + str(j) + '.txt', 'r')
            res_data = []
            with file as f1:
                for line in f1:
                    temp = line.split(' ')
                    l=len(temp)
                    if l == 4:
                        t = [float(t.replace('\n', '')) for t in temp]
                        res_data.append(t)
            # print(data)
            DHHFSP_data.append(res_data)
        filedata.append(DHHFSP_data)
    return filedata

def NEW_CPTEST2DataRead(filename):
    filedata = []
    for i in range(1, 12):
        DHHFSP_data = []
        if i < 10:
            id='0'+str(i);
        else:
            id=str(i);
        for j in range(1, 11):
            # print('CPTEST2\\' + filename + '\\DHHFSP' + id + '\\res' + str(j) + '.txt')
            file = open('CPTEST2_t\\' + filename + '\\DHHFSP' + id + '\\res' + str(j) + '.txt', 'r')
            res_data = []
            with file as f1:
                for line in f1:
                    temp = line.split(' ')
                    l = len(temp)
                    if l == 4:
                        t = [float(t.replace('\n', '')) for t in temp]
                        res_data.append(t)
            # print(data)
            DHHFSP_data.append(res_data)
        filedata.append(DHHFSP_data)
    return filedata

def Normalize(filename):
    alldata = []
    for i in filename:
        alldata.append(NEW_CPTEST2DataRead(i))
    print("文件读取完成")
    for i in range(1,11): #12写完05
        DHHFSP_data = []   
        for m in range(len(alldata)):
            for m_t in range(len(alldata)):
                for j in range(1, 11):#11
                    DHHFSP_data += alldata[m_t][i - 1][j - 1]
            # print(DHHFSP_data)
            # print('\n')
            allHV = []
            for j in range(1, 11):#11
                new_res = []
                for k in range(len(alldata[m][i - 1][j - 1])):
                    new_data = []
                    for l in range(len(alldata[m][i - 1][j - 1][k])):
                        min_data = min([row[l] for row in DHHFSP_data])
                        max_data = max([row[l] for row in DHHFSP_data])
                        new_data.append((alldata[m][i - 1][j - 1][k][l] - min_data) / (max_data - min_data))
                    new_res.append(new_data)

                if i < 10:
                    id='0'+str(i);
                else:
                    id=str(i);
                # print(new_res)
                with open('CPTEST2_t\\' + filename[m] + '\\DHHFSP' + id + '\\new_res' + str(j) + '.txt', 'w') as f:
                    for k in range(len(new_res)):
                        f.write(' '.join(str(x) for x in new_res[k]) + '\n')
                    print('CPTEST2_t\\' + filename[m] + '\\DHHFSP' + id + '\\new_res' + str(j) + '.txt' + ' 写入完成')

                    # allHV.append(hv(new_res))
                    # f.write('HV = ' + str(hv(new_res)) + '\n')
                    # if j == 10:
                    #     f.write('AverHV = ' + str(np.mean(allHV)) + '\n')
                    #     f.write('HVStandardDeviation = ' + str(CalculateStandardDeviation(allHV)) + '\n')
                    
                    #     if i == 1:
                    #         with open('CPTEST2\\'  + filename[m] + '\\HV' + '.txt', 'w') as f2:
                    #             f2.write('CPTEST2\\' + filename[m] + '\\DHHFSP' + id + '  AverHV = ' + str(np.mean(allHV)) + '\n')
                    #             f2.write('CPTEST2\\' + filename[m] + '\\DHHFSP' + id + '  HVStandardDeviation = ' + str(CalculateStandardDeviation(allHV)) + '\n')
                    #     else:
                    #         with open('CPTEST2\\'  + filename[m] + '\\HV' + '.txt', 'a') as f2:
                    #             f2.write('CPTEST2\\' + filename[m] + '\\DHHFSP' + id + '  AverHV = ' + str(np.mean(allHV)) + '\n')
                    #             f2.write('CPTEST2\\' + filename[m] + '\\DHHFSP' + id + '  HVStandardDeviation = ' + str(CalculateStandardDeviation(allHV)) + '\n')
                    # f.close()
if __name__ == '__main__':
    filename = ['NSGA2_4', 'NSGA3_4']
    Normalize(filename)