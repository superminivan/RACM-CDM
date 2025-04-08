

# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle

# def load_data(file_path):
#     with open(file_path, 'rb') as file:
#         data = pickle.load(file)
#     return data

# def compute_similarity(features):
#     # 计算余弦相似度矩阵
#     similarity_matrix = cosine_similarity(features)
#     return similarity_matrix

# def find_top_k(similarity_matrix, k=1):
#     # 找到每个样本的top-k相似样本的索引
#     top_k_indices = np.argsort(-similarity_matrix)[:, 1:k+1]
#     return top_k_indices

# def weighted_merge(features, top_k_indices, weights=[0.95, 0.05]):
#     # 按照权重合并特征
#     merged_features = []
#     for i in range(len(features)):
#         merged_feature = weights[0] * features[i]
#         for j, idx in enumerate(top_k_indices[i]):
#             merged_feature += weights[j+1] * features[idx]
#         merged_features.append(merged_feature)
#     return np.array(merged_features)

# def replace_samples(data, mode, merged_text_features, merged_audio_features, merged_vision_features):
#     # 用合并后的特征替换原始特征
#     data[mode]['text'] = merged_text_features
#     data[mode]['audio'] = merged_audio_features
#     data[mode]['vision'] = merged_vision_features
#     return data

# def save_data(data, output_file_path):
#     with open(output_file_path, 'wb') as file:
#         pickle.dump(data, file)

# # 加载数据
# input_file_path = '/root/autodl-tmp/MOSEI/aligned_50.pkl'
# output_file_path = '/root/autodl-tmp/MOSEI/mergedaligned_50.pkl'
# data = load_data(input_file_path)

# # 定义处理的模式
# modes = ['train', 'valid']

# for mode in modes:
#     # 提取特征
#     text_features = data[mode]['text']
#     audio_features = data[mode]['audio']
#     vision_features = data[mode]['vision']

#     # 重塑特征以适应 cosine_similarity 函数
#     text_features_reshaped = text_features.reshape(text_features.shape[0], -1)
#     audio_features_reshaped = audio_features.reshape(audio_features.shape[0], -1)
#     vision_features_reshaped = vision_features.reshape(vision_features.shape[0], -1)

#     # 计算相似度矩阵
#     text_similarity = compute_similarity(text_features_reshaped)
#     audio_similarity = compute_similarity(audio_features_reshaped)
#     vision_similarity = compute_similarity(vision_features_reshaped)

#     # 找到每个样本的top-2相似样本的索引
#     text_top_k_indices = find_top_k(text_similarity, k=1)
#     audio_top_k_indices = find_top_k(audio_similarity, k=1)
#     vision_top_k_indices = find_top_k(vision_similarity, k=1)

#     # 按照权重合并特征
#     merged_text_features = weighted_merge(text_features, text_top_k_indices)
#     merged_audio_features = weighted_merge(audio_features, audio_top_k_indices)
#     merged_vision_features = weighted_merge(vision_features, vision_top_k_indices)

#     # 用合并后的特征替换原始特征
#     data = replace_samples(data, mode, merged_text_features, merged_audio_features, merged_vision_features)

# # 保存数据
# save_data(data, output_file_path)

# print(f"Data saved to {output_file_path}")

# def inspect_pkl_dict(file_path):
#     # 打开.pkl文件
#     with open(file_path, 'rb') as file:
#         # 使用pickle加载文件内容
#         data = pickle.load(file)
    
#     # 打印字典的键及其对应的值类型和结构
#     print("Keys in the dictionary and their types:")
#     for key, value in data.items():
#         print(f"  {key}: {type(value)}")
        
#         if isinstance(value, dict):
#             print("    Inner dictionary keys and their types:")
#             for inner_key, inner_value in value.items():
#                 print(f"      {inner_key}: {type(inner_value)}")
#                 if isinstance(inner_value, (list, np.ndarray)):
#                     print(f"        Length/Shape: {len(inner_value) if isinstance(inner_value, list) else inner_value.shape}")
#                 elif isinstance(inner_value, dict):
#                     print("        Inner dictionary keys and their types:")
#                     for inner_inner_key, inner_inner_value in inner_value.items():
#                         print(f"          {inner_inner_key}: {type(inner_inner_value)}")
#         elif isinstance(value, (list, np.ndarray)):
#             print(f"    Length/Shape: {len(value) if isinstance(value, list) else value.shape}")
#             if value and isinstance(value[0], dict):
#                 print("    First element is a dictionary. Keys and their types:")
#                 for inner_key, inner_value in value[0].items():
#                     print(f"      {inner_key}: {type(inner_value)}")
#         else:
#             print("    Value is not a list, numpy array, or dictionary.")

# inspect_pkl_dict(output_file_path)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def compute_similarity(features, method='cosine'):
    """计算多种相似度矩阵"""
    if method == 'cosine':
        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(features)
    elif method == 'euclidean':
        # 计算欧几里得距离，并转换为相似度
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(features)
        similarity_matrix = 1 / (1 + distances)  # 转换为相似度
    elif method == 'pearson':
        # 皮尔逊相关系数
        similarity_matrix = np.corrcoef(features)
    return similarity_matrix

def find_top_k(similarity_matrix, k=1):
    # 找到每个样本的top-k相似样本的索引
    top_k_indices = np.argsort(-similarity_matrix)[:, 1:k+1]
    return top_k_indices

def weighted_merge(features, top_k_indices, weights=[0.95, 0.05]):
    # 按照权重合并特征
    merged_features = []
    for i in range(len(features)):
        merged_feature = weights[0] * features[i]
        for j, idx in enumerate(top_k_indices[i]):
            merged_feature += weights[j+1] * features[idx]
        merged_features.append(merged_feature)
    return np.array(merged_features)

def enhance_features(features, enhancement_method, top_k_indices, similarity_matrix=None):
    """多种增强策略选项"""
    if enhancement_method == 'merge':
        # 现有的合并方法
        return weighted_merge(features, top_k_indices)
    
    elif enhancement_method == 'interpolate':
        # 线性插值
        enhanced_features = []
        for i in range(len(features)):
            alpha = np.random.beta(0.2, 0.2)  # 控制混合程度的参数
            idx = top_k_indices[i][0]  # 取最相似的一个样本
            enhanced_feature = alpha * features[i] + (1 - alpha) * features[idx]
            enhanced_features.append(enhanced_feature)
        return np.array(enhanced_features)
    
    elif enhancement_method == 'noise':
        # 添加噪声
        noise_level = 0.01
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    elif enhancement_method == 'mixup':
        # mixup增强
        enhanced_features = []
        for i in range(len(features)):
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            idx = top_k_indices[i][0]
            enhanced_feature = lam * features[i] + (1 - lam) * features[idx]
            enhanced_features.append(enhanced_feature)
        return np.array(enhanced_features)
    
    else:
        # 默认返回原始特征
        return features

def selective_enhancement(features, similarity_matrix, top_k_indices, threshold=0.7, enhancement_method='merge'):
    """选择性增强策略"""
    enhanced_features = features.copy()
    enhanced_count = 0  # 记录增强的样本数量
    
    for i in range(len(features)):
        # 获取样本i与其最相似样本的相似度
        idx = top_k_indices[i][0]
        sim = similarity_matrix[i, idx]
        
        # 只有当相似度低于阈值时才进行增强
        # 这里的逻辑是：相似度低的样本可能是难例或噪声样本，需要增强
        if sim > threshold:
            enhanced_count += 1
            
            # 针对不同的增强方法采取不同策略
            if enhancement_method == 'merge':
                # 使用加权合并方法
                weights = [0.95, 0.05]
                enhanced_feature = weights[0] * features[i]
                enhanced_feature += weights[1] * features[idx]
                enhanced_features[i] = enhanced_feature
                
            elif enhancement_method == 'interpolate':
                # 线性插值
                alpha = np.random.beta(0.2, 0.2)  # 控制混合程度的参数
                enhanced_feature = alpha * features[i] + (1 - alpha) * features[idx]
                enhanced_features[i] = enhanced_feature
                
            elif enhancement_method == 'noise':
                # 添加噪声
                noise_level = 0.01
                noise = np.random.normal(0, noise_level, features[i].shape)
                enhanced_features[i] = features[i] + noise
                
            elif enhancement_method == 'mixup':
                # # mixup增强
                # alpha = 0.2
                # lam = np.random.beta(alpha, alpha)
                # enhanced_feature = lam * features[i] + (1 - lam) * features[idx]
                # enhanced_features[i] = enhanced_feature
                # 高级mixup增强，使用多个相似样本
                num_samples = min(2, len(top_k_indices[i]))  # 使用前3个最相似样本
                alpha = 0.2
                enhanced_feature = features[i].copy() * 0.7  # 基础权重为70%
                
                # 为剩余30%权重分配给相似样本
                remaining_weight = 0.3
                for j in range(num_samples):
                    idx = top_k_indices[i][j]
                    lam = np.random.beta(alpha, alpha) * remaining_weight / num_samples
                    enhanced_feature += lam * features[idx]
                
                enhanced_features[i] = enhanced_feature
    
    print(f"Enhanced {enhanced_count}/{len(features)} samples")
    return enhanced_features

def replace_samples(data, mode, merged_text_features, merged_audio_features, merged_vision_features):
    # 用合并后的特征替换原始特征
    data[mode]['text'] = merged_text_features
    data[mode]['audio'] = merged_audio_features
    data[mode]['vision'] = merged_vision_features
    return data

def save_data(data, output_file_path):
    with open(output_file_path, 'wb') as file:
        pickle.dump(data, file)

# 加载数据
input_file_path = '/root/autodl-tmp/MOSEI/aligned_50.pkl'
output_file_path = '/root/autodl-tmp/MOSEI/mergedaligned_50.pkl'
data = load_data(input_file_path)

# 定义处理的模式
modes = ['train', 'valid']

# 设置增强参数
enhancement_method = 'mixup'  # 可选: 'merge', 'interpolate', 'noise', 'mixup'
similarity_threshold = 0.7    # 选择性增强的阈值
enable_selective = True       # 是否启用选择性增强

for mode in modes:
    # 提取特征
    text_features = data[mode]['text']
    audio_features = data[mode]['audio']
    vision_features = data[mode]['vision']

    # 重塑特征以适应 cosine_similarity 函数
    text_features_reshaped = text_features.reshape(text_features.shape[0], -1)
    audio_features_reshaped = audio_features.reshape(audio_features.shape[0], -1)
    vision_features_reshaped = vision_features.reshape(vision_features.shape[0], -1)

    # 计算相似度矩阵
    text_similarity = compute_similarity(text_features_reshaped)
    audio_similarity = compute_similarity(audio_features_reshaped)
    vision_similarity = compute_similarity(vision_features_reshaped)

    # 找到每个样本的top-k相似样本的索引
    text_top_k_indices = find_top_k(text_similarity, k=1)
    audio_top_k_indices = find_top_k(audio_similarity, k=1)
    vision_top_k_indices = find_top_k(vision_similarity, k=1)

    # 应用增强策略
    if enable_selective:
        # 选择性增强，只增强相似度低于阈值的样本
        enhanced_text_features = selective_enhancement(
            text_features, text_similarity, text_top_k_indices, 
            threshold=similarity_threshold, enhancement_method=enhancement_method
        )
        enhanced_audio_features = selective_enhancement(
            audio_features, audio_similarity, audio_top_k_indices, 
            threshold=similarity_threshold, enhancement_method=enhancement_method
        )
        enhanced_vision_features = selective_enhancement(
            vision_features, vision_similarity, vision_top_k_indices, 
            threshold=similarity_threshold, enhancement_method=enhancement_method
        )
    else:
        # 对所有样本应用相同的增强策略
        enhanced_text_features = enhance_features(
            text_features, enhancement_method, text_top_k_indices, text_similarity
        )
        enhanced_audio_features = enhance_features(
            audio_features, enhancement_method, audio_top_k_indices, audio_similarity
        )
        enhanced_vision_features = enhance_features(
            vision_features, enhancement_method, vision_top_k_indices, vision_similarity
        )

    # 用增强后的特征替换原始特征
    data = replace_samples(data, mode, enhanced_text_features, enhanced_audio_features, enhanced_vision_features)

# 保存数据
save_data(data, output_file_path)

print(f"Data saved to {output_file_path}")
print(f"Enhancement method: {enhancement_method}")
print(f"Selective enhancement: {'Enabled' if enable_selective else 'Disabled'}")
if enable_selective:
    print(f"Similarity threshold: {similarity_threshold}")

def inspect_pkl_dict(file_path):
    # 打开.pkl文件
    with open(file_path, 'rb') as file:
        # 使用pickle加载文件内容
        data = pickle.load(file)
    
    # 打印字典的键及其对应的值类型和结构
    print("Keys in the dictionary and their types:")
    for key, value in data.items():
        print(f"  {key}: {type(value)}")
        
        if isinstance(value, dict):
            print("    Inner dictionary keys and their types:")
            for inner_key, inner_value in value.items():
                print(f"      {inner_key}: {type(inner_value)}")
                if isinstance(inner_value, (list, np.ndarray)):
                    print(f"        Length/Shape: {len(inner_value) if isinstance(inner_value, list) else inner_value.shape}")
                elif isinstance(inner_value, dict):
                    print("        Inner dictionary keys and their types:")
                    for inner_inner_key, inner_inner_value in inner_value.items():
                        print(f"          {inner_inner_key}: {type(inner_inner_value)}")
        elif isinstance(value, (list, np.ndarray)):
            print(f"    Length/Shape: {len(value) if isinstance(value, list) else value.shape}")
            if value and isinstance(value[0], dict):
                print("    First element is a dictionary. Keys and their types:")
                for inner_key, inner_value in value[0].items():
                    print(f"      {inner_key}: {type(inner_value)}")
        else:
            print("    Value is not a list, numpy array, or dictionary.")

inspect_pkl_dict(output_file_path)