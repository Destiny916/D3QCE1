import sys
import os
import numpy as np
from pathlib import Path
from DataRead import DataReadDHHJSP1, ReadP1F1
from dual_factory_scheduler import dual_factory_scheduling

def process_all_datasets():
    """
    处理所有数据集，将P1.txt和F1.txt中的每一行转换为调度结果
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "DATASET_DHHFSP1"
    last_dir = base_dir / "data" / "last" / "last"
    
    dataset_mapping = {
        "20J3S2F.txt": "01",
        "20J3S3F.txt": "02",
        "20J5S2F.txt": "03",
        "20J5S3F.txt": "04",
        "40J3S2F.txt": "05",
        "40J3S3F.txt": "06",
        "40J5S2F.txt": "07",
        "40J5S3F.txt": "08",
        "60J3S2F.txt": "09",
        "60J3S3F.txt": "10",
        "60J5S2F.txt": "11",
        "60J5S3F.txt": "12",
        "80J3S2F.txt": "13",
        "80J3S3F.txt": "14",
        "80J5S2F.txt": "15",
        "80J5S3F.txt": "16",
        "100J3S2F.txt": "17",
        "100J3S3F.txt": "18",
        "100J5S2F.txt": "19",
        "100J5S3F.txt": "20",
    }
    
    for dataset_file, folder_num in dataset_mapping.items():
        print(f"\n处理数据集: {dataset_file} -> DHHFSP{folder_num}")
        
        dataset_path = data_dir / dataset_file
        if not dataset_path.exists():
            print(f"  跳过：数据集文件不存在 {dataset_path}")
            continue
        
        output_dir = last_dir / f"DHHFSP{folder_num}"
        if not output_dir.exists():
            print(f"  跳过：输出目录不存在 {output_dir}")
            continue
        
        p1_f1_dir = output_dir
        P, F = ReadP1F1(str(p1_f1_dir))
        
        if P is None or F is None or len(P) == 0 or len(F) == 0:
            print(f"  跳过：P或F数据为空")
            continue
        
        N, TS, F_count, NS, time, JP, JDD, Length = DataReadDHHJSP1(str(dataset_path))
        
        print(f"  数据集参数: N={N}, TS={TS}, F={F_count}, NS={NS}")
        print(f"  P数据形状: {P.shape}, F数据形状: {F.shape}")
        
        num_schedules = len(P)
        all_results = []
        
        output_file = output_dir / "schedule.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(num_schedules):
                pchrom = P[i].tolist()
                fchrom = F[i].tolist()
                
                print(f"  处理第 {i+1}/{num_schedules} 组: pchrom长度={len(pchrom)}, fchrom长度={len(fchrom)}")
                
                try:
                    schedule = dual_factory_scheduling(pchrom, fchrom, time, JDD=JDD, NS=NS)
                    for item in schedule:
                        line = f"{item[0]} {item[1]} {item[2]} {item[3]} {item[4]} {item[5]}\n"
                        f.write(line)
                    f.write("\n")
                except Exception as e:
                    print(f"  错误: {e}")
                    continue
        
        print(f"  结果已保存到: {output_file}")
    
    print("\n所有数据集处理完成！")

if __name__ == "__main__":
    process_all_datasets()
