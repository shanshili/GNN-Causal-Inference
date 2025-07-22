"""
Intel 实验室传感器数据半小时粒度处理工具
=======================================

本脚本从预处理后的数据中提取半小时粒度的传感器数据，主要功能：

1. 数据处理流程：
   - 读取预处理的3D传感器数据和时间戳
   - 将数据重新采样为半小时粒度
   - 处理时间对齐和缺失值

2. 输出数据：
   - 半小时粒度的3D传感器数据数组
   - 对应的时间戳数据

3. 关键特性：
   - 精确的时间段划分
   - 自动处理时间边界情况
   - 保留原始数据特征顺序
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import SimpleImputer

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
except:
    pass  # 如果绘图库未使用，忽略字体设置


# 设置数据路径
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, '..', 'dataset')

# 检查输入文件是否存在
sensor_data_path = os.path.join(dataset_dir, 'sensor_data_3d.npy')
timestamp_path = os.path.join(dataset_dir, 'timestamps.npy')

# 加载原始数据
try:
    data_3d = np.load(sensor_data_path)
    timestamps = np.load(timestamp_path, allow_pickle=True)
except Exception as e:
    raise RuntimeError(f"加载数据文件时发生错误: {str(e)}")

print(f"原始数据形状: {data_3d.shape}")
print(f"时间戳数量: {len(timestamps)}")

# 转换为DataFrame处理时间重采样
try:
    timestamps = pd.to_datetime(timestamps)
except Exception as e:
    raise RuntimeError(f"时间戳转换失败: {str(e)}")


def evaluate_data_coherence(data_3d, timestamps, node_threshold=0.3, time_threshold=0.7):
    """
    评估每个传感器节点的半小时数据连贯性，返回最佳时间段

    参数:
        data_3d: 3D传感器数据 [节点, 时间点, 特征]
        timestamps: 对应的时间戳
        node_threshold: 节点筛选阈值(缺失率>此值则剔除)
        time_threshold: 时间连贯性阈值(节点覆盖率>此值则保留)
    """

    # ========== 新增数据诊断代码 ==========
    print("\n=== 数据诊断信息 ===")
    print(f"数据形状 (节点数×时间点数×特征数): {data_3d.shape}")

    # 检查各特征列的缺失情况
    features = ['moteid', 'temperature', 'humidity', 'light', 'voltage', 'x', 'y']
    print("\n各特征缺失率:")
    for i, feat in enumerate(features):
        missing_rate = np.mean(np.isnan(data_3d[:, :, i]))
        print(f"- {feat}: {missing_rate:.2%}")

    # 随机检查几个节点的数据
    print("\n示例节点数据检查（前3个节点，前20个时间点）:")
    for node in range(min(10, data_3d.shape[0])):
        print(f"\n节点 {node} 前20个时间点:")
        print(data_3d[node, :20, :])  # 第1列是temperature
    # 转换为半小时粒度时间标签
    half_hour_labels = pd.to_datetime(timestamps).floor('30min')
    unique_intervals = pd.Series(half_hour_labels).unique()

    # 用温度列评估节点质量
    node_missing = np.mean(np.isnan(data_3d[:, :, 1]), axis=1)
    valid_nodes = node_missing < node_threshold
    filtered_data = data_3d[valid_nodes, :, :]
    print(f"\n节点筛选结果: 保留{np.sum(valid_nodes)}/{len(valid_nodes)}个节点")

    # 2. 评估每个半小时区间的节点覆盖率
    interval_quality = []
    for interval in unique_intervals:
        # 获取当前半小时的所有数据点
        mask = (half_hour_labels == interval)
        interval_data = filtered_data[:, mask, 0]  # 使用moteid列评估

        # 计算有多少节点在此半小时有数据
        node_has_data = ~np.isnan(interval_data).all(axis=1)
        coverage = np.mean(node_has_data)
        interval_quality.append((interval, coverage))

    # 3. 寻找最佳连续时间段 (节点覆盖率>threshold的最长连续段)
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0

    for i in range(len(interval_quality)):
        _, coverage = interval_quality[i]
        if coverage > time_threshold:
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
        else:
            current_start = i + 1
            current_length = 0

    # 获取最佳时间段
    best_intervals = [interval_quality[i][0]
                      for i in range(best_start, best_start + best_length)]
    best_mask = np.isin(half_hour_labels, best_intervals)

    # 输出统计信息
    print("\n=== 半小时数据分析结果 ===")
    print(f"总时间段数: {len(unique_intervals)}")
    print(f"最佳连续时间段数: {best_length}")
    if best_length > 0:
        print(
            f"时间段覆盖节点比例: {np.mean([iq[1] for iq in interval_quality[best_start:best_start + best_length]]):.2%}")
        print(f"开始时间: {best_intervals[0]}")
        print(f"结束时间: {best_intervals[-1]}")
    else:
        print("警告: 未找到满足条件的时间段，请降低阈值或检查数据质量")

    return best_mask, best_intervals


def filter_and_impute(data_3d, threshold=0.3):
    """筛选传感器节点并进行缺失值填补"""
    num_nodes = data_3d.shape[0]

    # 计算每个节点的缺失率
    missing_rates = np.mean(np.isnan(data_3d[:, :, 0]), axis=1)  # 使用moteid列

    # 筛选节点
    keep_nodes = missing_rates < threshold
    filtered_data = data_3d[keep_nodes, :, :]

    print(f"\n节点筛选结果: 保留{np.sum(keep_nodes)}/{num_nodes}个节点")
    print(f"剔除节点缺失率: {np.mean(missing_rates[~keep_nodes]):.2%}")

    # 对保留的节点进行缺失值填补
    for i in range(filtered_data.shape[0]):
        for j in range(filtered_data.shape[2]):
            if np.any(np.isnan(filtered_data[i, :, j])):
                # 使用前后值的线性插值
                valid_mask = ~np.isnan(filtered_data[i, :, j])
                if np.sum(valid_mask) > 1:  # 只有足够数据时才插值
                    indices = np.arange(len(valid_mask))
                    filtered_data[i, :, j] = np.interp(
                        indices,
                        indices[valid_mask],
                        filtered_data[i, valid_mask, j]
                    )
                else:
                    filtered_data[i, :, j] = 0  # 数据太少则填0

    return filtered_data, keep_nodes


def save_quality_report(data_3d, timestamps, output_dir):
    """生成并保存数据质量报告"""
    report_path = os.path.join(output_dir, 'quality_report.txt')

    with open(report_path, 'w') as f:
        f.write("传感器数据质量评估报告\n")
        f.write("=======================\n\n")

        # 时间范围信息
        f.write(f"时间范围: {timestamps[0]} 至 {timestamps[-1]}\n")
        f.write(f"总时间点数: {len(timestamps)}\n\n")

        # 节点完整性统计
        node_completeness = 1 - np.mean(np.isnan(data_3d[:, :, 0]), axis=1)
        f.write("节点数据完整性统计:\n")
        f.write(f"- 平均完整性: {np.mean(node_completeness):.2%}\n")
        f.write(f"- 最低完整性: {np.min(node_completeness):.2%}\n")
        f.write(f"- 最高完整性: {np.max(node_completeness):.2%}\n\n")

        # 特征缺失统计
        f.write("各特征缺失情况:\n")
        features = ['moteid', 'temperature', 'humidity', 'light', 'voltage', 'x', 'y']
        for i, feat in enumerate(features):
            missing = np.mean(np.isnan(data_3d[:, :, i]))
            f.write(f"- {feat}: {missing:.2%}缺失\n")

        # 时间连续性分析
        time_gaps = np.diff(timestamps).astype('timedelta64[h]')
        f.write("\n时间连续性分析:\n")
        f.write(f"- 最大时间间隔: {np.max(time_gaps)}小时\n")
        f.write(f"- 平均时间间隔: {np.mean(time_gaps):.1f}小时\n")


def process_half_hour_data(data_3d,timestamps):
    """处理半小时粒度传感器数据"""
    min_time, max_time = timestamps.min(), timestamps.max()

    # 创建半小时时间 bins
    print("\n创建半小时时间区间...")
    try:
        time_bins = pd.date_range(
            start=min_time.floor('h'),  # 使用'h'代替'H'
            end=max_time.ceil('h'),  # 使用'h'代替'H'
            freq='30min'
        )
    except Exception as e:
        raise RuntimeError(f"创建时间区间失败: {str(e)}")

    # 初始化结果数组
    num_nodes = data_3d.shape[0]
    num_features = data_3d.shape[2]
    half_hour_data = np.full((num_nodes, len(time_bins) - 1, num_features), np.nan)

    print("\n处理每个半小时区间...")
    print("注意: 开始处理时间区间，总进度如下:")
    for i in tqdm(range(len(time_bins) - 1), total=len(time_bins) - 1, desc="处理时间区间", unit="interval"):
        # 获取当前半小时区间
        start_time = time_bins[i]
        end_time = time_bins[i + 1]

        # 找出在此区间内的原始数据点
        time_mask = (timestamps >= start_time) & (timestamps < end_time)
        interval_data = data_3d[:, time_mask, :]

        # 计算半小时平均值 (忽略NaN)
        with np.errstate(invalid='ignore'):
            # 检查是否有有效数据
            if interval_data.shape[1] > 0 and not np.all(np.isnan(interval_data)):
                result = np.nanmean(interval_data, axis=1)
                # 确保结果维度匹配
                if result.shape[0] == num_nodes and result.shape[1] == num_features:
                    half_hour_data[:, i, :] = result

    # 保存结果
    output_path = os.path.join(dataset_dir, 'sensor_data_3d_30min.npy')
    try:
        np.save(output_path, half_hour_data)
    except Exception as e:
        raise RuntimeError(f"保存传感器数据失败: {str(e)}")

    # 保存时间戳
    time_labels = [f"{time_bins[i].strftime('%Y-%m-%d %H:%M')}"
                   for i in range(len(time_bins) - 1)]
    time_output_path = os.path.join(dataset_dir, 'timestamps_30min.npy')
    try:
        np.save(time_output_path, time_labels)
    except Exception as e:
        raise RuntimeError(f"保存时间戳失败: {str(e)}")

    # 添加数据质量报告
    print("\n=== 数据质量报告 ===")
    print(f"总时间区间数: {len(time_bins) - 1}")
    print(f"无数据的时间区间数: {np.isnan(half_hour_data).all(axis=(0, 2)).sum()}")
    print(f"每个传感器节点的非空值数量: {np.sum(~np.isnan(half_hour_data).all(axis=2), axis=1)}")

    print("\n处理完成!")
    print(f"半小时粒度数据形状: {half_hour_data.shape}")
    print(f"特征顺序保持原始顺序: [moteid, temperature, humidity, light, voltage, x, y]")
    print(f"数据已保存到: {output_path}")

    return half_hour_data,time_labels


if __name__ == '__main__':

    # 步骤1: 评估数据连贯性
    best_mask, best_dates = evaluate_data_coherence(data_3d, timestamps)
    coherent_data = data_3d[:, best_mask, :]
    coherent_timestamps = timestamps[best_mask]

    # 步骤2: 按半小时采样
    hourly_data, hourly_timestamps = process_half_hour_data(coherent_data, coherent_timestamps)

    # 步骤3: 筛选节点并填补
    filtered_data, keep_nodes = filter_and_impute(hourly_data)

    # 保存结果
    output_path = os.path.join(dataset_dir, 'filtered_sensor_data_3d.npy')
    np.save(output_path, filtered_data)
    np.save(os.path.join(dataset_dir, 'filtered_timestamps.npy'), hourly_timestamps)
