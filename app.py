from flask import Flask, render_template, jsonify
from pathlib import Path
import re


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("preface.html")

    @app.route("/api/dataset/<dataset_name>")
    def get_dataset_data(dataset_name):
        base_dir = Path(__file__).parent
        
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
        
        dataset_num = dataset_mapping.get(dataset_name)
        if dataset_num is None:
            return jsonify({"error": "Invalid dataset name"}), 400
        
        plot_path = base_dir / "data" / "last" / "last" / f"DHHFSP{dataset_num}" / "res1.txt"
        plot_points = []
        if plot_path.exists():
            with plot_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                    try:
                        x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
                    except ValueError:
                        continue
                    plot_points.append([x, y, z])
        
        return jsonify({"plot_points": plot_points})

    @app.route("/api/schedule/<dataset_name>/<int:point_index>")
    def get_schedule_data(dataset_name, point_index):
        base_dir = Path(__file__).parent
        
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
        
        dataset_num = dataset_mapping.get(dataset_name)
        if dataset_num is None:
            return jsonify({"error": "Invalid dataset name"}), 400
        
        schedule_path = base_dir / "data" / "last" / "last" / f"DHHFSP{dataset_num}" / "schedule.txt"
        processes = []
        
        if schedule_path.exists():
            with schedule_path.open("r", encoding="utf-8") as handle:
                content = handle.read()
            
            # 使用空行（回车符）作为分隔符分割数据点
            # 每个数据点之间有一个或多个空行
            data_points = []
            current_point = []
            
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped == '':
                    # 空行表示数据点分隔
                    if current_point:
                        data_points.append(current_point)
                        current_point = []
                else:
                    current_point.append(stripped)
            
            # 添加最后一个数据点（如果存在）
            if current_point:
                data_points.append(current_point)
            
            # 确保数据点索引在有效范围内
            if 0 <= point_index < len(data_points):
                target_point = data_points[point_index]
                for line in target_point:
                    parts = line.split()
                    if len(parts) != 6:
                        continue
                    try:
                        factory = int(parts[0])
                        stage = int(parts[1])
                        machine = int(parts[2])
                        # 工件编号从0-99调整为1-100
                        job = int(parts[3]) + 1
                        start = float(parts[4])
                        end = float(parts[5])
                        processes.append([factory, stage, machine, job, start, end])
                    except ValueError:
                        continue
        
        return jsonify({"processes": processes})

    @app.route("/results")
    @app.route("/results/<dataset_name>")
    def results(dataset_name=None):
        import os
        base_dir = Path(__file__).parent
        dataset_dir = base_dir / "data" / "DATASET_DHHFSP1"
        
        datasets = []
        if dataset_dir.exists():
            dataset_files = [p.name for p in dataset_dir.iterdir() if p.is_file() and p.suffix == ".txt"]
            datasets = sorted(dataset_files, key=lambda x: int(re.match(r'(\d+)', x).group(1)) if re.match(r'(\d+)', x) else 0)
        else:
            print(f"数据集目录不存在: {dataset_dir}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"基础目录: {base_dir}")
        
        if dataset_name is None and datasets:
            dataset_name = datasets[0]
        elif dataset_name is None:
            dataset_name = "20J3S2F.txt"
        
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
        
        dataset_num = dataset_mapping.get(dataset_name)
        if dataset_num is None:
            dataset_num = "01"
            
        plot_path = base_dir / "data" / "last" / "last" / f"DHHFSP{dataset_num}" / "res1.txt"
        plot_points = []
        if plot_path.exists():
            with plot_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                    try:
                        x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
                    except ValueError:
                        continue
                    plot_points.append([x, y, z])

        processes = [
           
        ]

        return render_template(
            "results.html",
            datasets=datasets,
            plot_points=plot_points,
            processes=processes,
            current_dataset=dataset_name,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
