import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run import IMDER_run  # 假设 run.py 文件中的函数已经可以正确运行

# 读取配置文件
config_file_path = Path(__file__).parent / "config" / "config.json"
# 读取配置文件并保存原始配置的副本
with open(config_file_path, 'r') as f:
    original_config = json.load(f)  # 保存原始配置的副本

# 再次读取配置文件用于修改
with open(config_file_path, 'r') as f:
    config = json.load(f)  # 用于修改的配置

# 设置网格搜索的参数范围
param_grid = {
    'learning_rate': [ 0.003, 0.0025, 0.0005, 0.0001, 0.0002,0.001, 0.002,]
    # 'learning_rate': [0.003]
}

# 用于存储结果的列表
results = []

# 进行网格搜索
for lr in param_grid['learning_rate']:
    # for bs in param_grid['batch_size']:
        # 更新配置文件中的参数
        config['imder']['datasetParams']['mosei']['learning_rate'] = lr
        # config['imder']['datasetParams']['mosi']['batch_size'] = bs
        
        # 将更新后的配置写回文件
        with open(config_file_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        
        # 运行模型
        result = IMDER_run(
            model_name='imder',
            dataset_name='mosei',
            seeds=[1114],  # 只用一个种子进行示例，实际应用中可能需要多个种子
            mode='train',
            mr=0.7
        )
        print('lr:')
        print(lr)
        # 存储结果
        results.append({
            'learning_rate': lr,
            # 'batch_size': bs,
            'result': result
        })

# 将结果保存到CSV文件中
results_df = pd.DataFrame(results)
results_df.to_csv('grid_search_results.csv', index=False)

# 恢复配置文件到原始状态
with open(config_file_path, 'w') as f:
    json.dump(original_config, f, indent=4)