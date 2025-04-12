import os
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import scipy.io as scio
from transforms3d.axangles import axangle2mat

# 常量定义
WINDOW_SIZE = 512
DATASET_DIR = './large_scale_HARBox/'
NUM_USERS = 120
NUM_LABELS = 5
CLASS_SET = ['Call', 'Hop', 'typing', 'Walk', 'Wave']
DIMENSION_OF_FEATURE = 900

# 数据增强函数
def DA_Jitter(X, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

def DA_Scaling(X, sigma=0.1):
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    return X * scaling_factor

def time_warp(x, sigma=0.2, knots=4):
    seq_len, features = x.shape
    new_x = np.zeros_like(x)
    for i in range(features):
        original_knots = np.linspace(0, seq_len - 1, num=knots)
        new_knots = original_knots + np.random.normal(loc=0., scale=sigma, size=original_knots.shape) * (seq_len / knots)
        new_knots[0] = 0
        new_knots[-1] = seq_len - 1
        new_knots = np.clip(new_knots, 0, seq_len - 1)
        new_indices = np.arange(seq_len)
        new_time_steps = np.interp(new_indices, original_knots, new_knots)
        new_x[:, i] = np.interp(new_time_steps, new_indices, x[:, i])
    return new_x

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    num_samples, num_features = X.shape
    xx = np.linspace(0, num_samples - 1, knot + 2)
    curves = []
    for _ in range(num_features):
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
        cs = CubicSpline(xx, yy)
        curves.append(cs(np.arange(num_samples)))
    return np.column_stack(curves)

def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma)
    tt_cum = np.cumsum(tt, axis=0)
    t_scale = [(X.shape[0]-1)/tt_cum[-1, c] for c in range(X.shape[1])]
    for c in range(X.shape[1]):
        tt_cum[:, c] = tt_cum[:, c] * t_scale[c]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros_like(X)
    x_range = np.arange(X.shape[0])
    for c in range(X.shape[1]):
        X_new[:, c] = np.interp(x_range, tt_new[:, c], X[:, c])
    return X_new

def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros_like(X)
    idx = np.random.permutation(nPerm)
    while True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[:-1]) > minSegLength:
            break
    pp = 0
    for ii in range(nPerm):
        segment = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(segment), :] = segment
        pp += len(segment)
    return X_new

def RandSampleTimesteps(X, nSample):
    X_new = np.zeros_like(X)
    tt = np.zeros((nSample, X.shape[1]), dtype=int)
    for c in range(X.shape[1]):
        tt[1:-1, c] = np.sort(np.random.randint(1, X.shape[0]-1, nSample - 2))
    tt[-1, :] = X.shape[0] - 1
    return tt

def DA_RandSampling(X, nSample_rate):
    nSample = int(len(X) * nSample_rate)
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros_like(X)
    for c in range(X.shape[1]):
        X_new[:, c] = np.interp(np.arange(X.shape[0]), tt[:, c], X[tt[:, c], c])
    return X_new

def process_data(n_users):
    """
    针对每个用户，读取各个类别数据并进行数据增强，
    返回 user_ids、包含 8 种增强数据的列表，以及对应的标签列表。
    """
    user_ids = []
    # 为 8 种数据增强准备容器
    aug_segments = [[] for _ in range(8)]
    aug_labels = [[] for _ in range(8)]
    
    for user_id in range(1, n_users + 1):
        print(f"==> Processing data for user {user_id}...")
        for class_id in range(NUM_LABELS):
            read_path = os.path.join(DATASET_DIR, str(user_id), f"{CLASS_SET[class_id]}_train.txt")
            if os.path.exists(read_path):
                raw_data = np.loadtxt(read_path)
                sigma = 0.2
                # 定义增强函数：原始、Jitter、Scaling、time_warp、Permutation、MagWarp、TimeWarp、RandSampling
                augmentation_functions = [
                    lambda x: x,
                    DA_Jitter,
                    DA_Scaling,
                    time_warp,
                    DA_Permutation,
                    lambda x: DA_MagWarp(x, sigma),
                    lambda x: DA_TimeWarp(x, sigma),
                    lambda x: DA_RandSampling(x, 0.4)
                ]
                aug_outputs = [func(raw_data) for func in augmentation_functions]
                reshaped_outputs = [output.reshape(-1, 100, 10) for output in aug_outputs]
                for i, augmented_data in enumerate(reshaped_outputs):
                    # 提取第 2 至第 10 列 (索引 1:10) 得到 (-1, 100, 9) 并扩展维度 --> (-1, 1, 100, 9)
                    processed = np.expand_dims(augmented_data[:, :, 1:10], axis=1)
                    aug_segments[i].append(processed)
                    # 构建当前数据的标签数组
                    labels = np.full(processed.shape[0], class_id, dtype=int)
                    aug_labels[i].append(labels)
                    # 记录每个样本对应的 user_id
                    user_ids.extend([user_id] * processed.shape[0])
    
    # 合并各增强方式的所有数据
    reshaped_segments_all = [np.concatenate(aug_segments[i], axis=0) if aug_segments[i] else np.array([]) for i in range(8)]
    labels_all = [np.concatenate(aug_labels[i], axis=0) if aug_labels[i] else np.array([]) for i in range(8)]
    user_ids = np.array(user_ids)
    return user_ids, reshaped_segments_all, labels_all

def custom_train_test_split(X, X_aug, y, test_size=0.2, val_size=0.1):
    """
    将原始特征与标签划分为训练、测试和验证集，同时对增强数据（索引 1-7）进行同步划分。
    """
    total_size = X.shape[0]
    indices = np.random.permutation(total_size)
    test_count = int(total_size * test_size)
    val_count = int(total_size * val_size)
    train_count = total_size - test_count - val_count

    train_indices = indices[:train_count]
    test_indices = indices[train_count:train_indices.shape[0] + test_count]
    val_indices = indices[train_count + test_count:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # 对增强数据（索引 1 到 7）进行同步划分
    X_aug_train = {i: X_aug[i][train_indices] for i in range(1, 8)}
    X_aug_val = {i: X_aug[i][val_indices] for i in range(1, 8)}

    return X_train, X_aug_train, X_test, X_val, y_train, y_test, y_val, X_aug_val

def main():
    parser = argparse.ArgumentParser(description="Process and split HARBox data with augmentations")
    parser.add_argument("--n_class", type=int, default=NUM_LABELS, help="Number of classes")
    parser.add_argument("--min_sample", type=int, default=20, help="Minimum samples per user")
    parser.add_argument("--n_ways", type=int, default=NUM_LABELS, help="n ways")
    parser.add_argument("--k_shots", type=int, default=20, help="k shot: number of training samples")
    parser.add_argument("--stdv", type=int, default=0, help="Noise level for choosing k shot samples")
    parser.add_argument("--n_user", type=int, default=NUM_USERS, help="Number of users")
    parser.add_argument('--imbalance', action='store_true', default=False, help="Use imbalanced sampling")
    parser.add_argument("--sample_size", type=int, default=256, help="Sample size per user")
    args = parser.parse_args()

    print(f"Number of classes: {args.n_class}")
    print(f"Noise stdv: {args.stdv}")
    print(f"n_ways: {args.n_ways}")
    print(f"k_shots: {args.k_shots}")

    # 这里假设目录结构与原来的相同
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    if args.imbalance:
        train_path = os.path.join(dir_path, f'harbox/{args.n_user}u_{args.sample_size}l_data/train/')
        test_path = os.path.join(dir_path, f'harbox/{args.n_user}u_{args.sample_size}l_data/test/')
        logging_path = os.path.join(dir_path, 'harbox/VisualDataDistribution', f'{args.n_user}u_{args.sample_size}l_logging.json')
    else:
        train_path = os.path.join(dir_path, f'harbox/{args.k_shots}k_{args.n_user}b_{args.stdv}p_data/train/')
        test_path = os.path.join(dir_path, f'harbox/{args.k_shots}k_{args.n_user}b_{args.stdv}p_data/test/')
        logging_path = os.path.join(dir_path, 'harbox/VisualDataDistribution', f'{args.k_shots}k_{args.n_user}b_{args.stdv}p_logging.json')

    # 确保输出目录存在
    for path in (train_path, test_path, logging_path):
        out_dir = os.path.dirname(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # 数据处理与增强
    user_ids, segments, labels = process_data(args.n_user)
    print("Number of user ids:", user_ids.shape)
    print("Shape of original augmentation (index 0):", segments[0].shape)
    print("Shape of labels:", labels[0].shape)
    
    # 对原始数据及标签进行训练/测试/验证划分
    X_train, X_aug_train, X_test, X_val, y_train, y_test, y_val, X_aug_val = custom_train_test_split(
        segments[0], segments, labels[0], test_size=0.2, val_size=0.1)
    
    base_path = './'
    # 保存基础数据
    for fname, data_dict in zip(['basedata.npz', 'train.npz', 'test.npz'],
                                [{'x': X_val, 'y': y_val},
                                 {'x': X_train, 'y': y_train},
                                 {'x': X_test, 'y': y_test}]):
        with open(os.path.join(base_path, fname), 'wb') as f:
            np.savez_compressed(f, **data_dict)

    # 保存增强验证数据（统一格式： DA_XXX_basedata.npz）
    val_file_names = [
        'DA_Jitter_basedata.npz',
        'DA_Scaling_basedata.npz',
        'DA_time_warp_basedata.npz',
        'DA_Permutation_basedata.npz',
        'DA_MagWarp_basedata.npz',
        'DA_TimeWarp_basedata.npz',
        'DA_RandSampling_basedata.npz'
    ]
    for idx, file_name in enumerate(val_file_names, start=1):
        val_data = {'x': X_aug_val[idx], 'y': y_val}
        with open(os.path.join(base_path, file_name), 'wb') as f:
            np.savez_compressed(f, **val_data)
        print(f"Saved {file_name}")

    # 保存增强训练数据（统一格式： DA_XXX_train.npz）
    train_file_names = [
        'DA_Jitter_train.npz',
        'DA_Scaling_train.npz',
        'DA_time_warp_train.npz',
        'DA_Permutation_train.npz',
        'DA_MagWarp_train.npz',
        'DA_TimeWarp_train.npz',
        'DA_RandSampling_train.npz'
    ]
    for idx, file_name in enumerate(train_file_names, start=1):
        train_data = {'x': X_aug_train[idx], 'y': y_train}
        with open(os.path.join(base_path, file_name), 'wb') as f:
            np.savez_compressed(f, **train_data)
        print(f"Saved {file_name}")
    
    print("Data processing and saving completed!")

if __name__ == '__main__':
    main()
