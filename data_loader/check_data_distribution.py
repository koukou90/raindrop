"""
数据分布诊断脚本
检查测试集中强降水样本的分布情况
"""
import numpy as np
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.dataset import (
    prepare_datasets, load_preprocessed_data,
    extract_dsd_features, create_sliding_window_samples
)
from utils.feature_engineering import apply_log_transform


def check_test_distribution(site_name, data_dir, seq_len, pred_len, stride,
                           train_ratio, val_ratio, threshold=20.0):
    """
    检查测试集数据分布

    Args:
        site_name: 站点名称
        data_dir: 数据目录
        seq_len: 输入序列长度
        pred_len: 预测长度
        stride: 滑动窗口步长
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        threshold: 强降水阈值 (mm/h)
    """
    print("="*70)
    print("数据分布诊断")
    print("="*70)

    # 加载数据和事件信息
    _, _, events_info = prepare_datasets(
        site_name=site_name,
        data_dir=data_dir,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    test_events = events_info['test_events']
    print(f"\n测试集包含 {len(test_events)} 个事件")

    # 重新加载原始数据用于逐事件分析
    df_dsd, df_params = load_preprocessed_data(site_name, data_dir)
    conc, vel = extract_dsd_features(df_dsd)
    phys = df_params[['RainRate', 'Dm', 'LogNw', 'LWC', 'Z']].values
    rain_rate = df_params['RainRate'].values
    conc = apply_log_transform(conc)

    # 收集所有测试集的标签（真实值）
    all_labels = []
    all_labels_by_step = [[] for _ in range(pred_len)]

    print("\n逐事件分析:")
    print("-"*70)

    for evt_idx, evt in enumerate(test_events):
        # 提取该事件的数据
        event_indices = list(range(evt['start_idx'], evt['end_idx'] + 1))
        event_conc = conc[event_indices]
        event_vel = vel[event_indices]
        event_phys = phys[event_indices]
        event_rain = rain_rate[event_indices]

        # 对该事件生成滑动窗口样本
        X_conc, X_vel, X_phys, Y = create_sliding_window_samples(
            event_conc, event_vel, event_phys, event_rain,
            seq_len, pred_len, stride
        )

        if X_conc is None or len(X_conc) == 0:
            continue

        # 收集标签
        all_labels.append(Y)  # (N_event, pred_len)

        # 按步长收集
        for step in range(pred_len):
            all_labels_by_step[step].extend(Y[:, step].tolist())

    # 合并所有标签
    all_labels_array = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    print(f"\n总样本数: {len(all_labels_array)}")
    print(f"标签形状: {all_labels_array.shape}")

    # 整体统计
    print("\n" + "="*70)
    print("整体统计信息:")
    print("-"*70)
    all_labels_flat = all_labels_array.flatten()
    print(f"最小值: {np.min(all_labels_flat):.4f} mm/h")
    print(f"最大值: {np.max(all_labels_flat):.4f} mm/h")
    print(f"平均值: {np.mean(all_labels_flat):.4f} mm/h")
    print(f"中位数: {np.median(all_labels_flat):.4f} mm/h")
    print(f"标准差: {np.std(all_labels_flat):.4f} mm/h")

    # 强降水统计（整体）
    print("\n" + "="*70)
    print(f"强降水统计 (>= {threshold} mm/h):")
    print("-"*70)
    heavy_rain_mask = all_labels_flat >= threshold
    heavy_rain_count = np.sum(heavy_rain_mask)
    heavy_rain_ratio = heavy_rain_count / len(all_labels_flat) * 100

    print(f"强降水样本数: {heavy_rain_count} / {len(all_labels_flat)}")
    print(f"强降水比例: {heavy_rain_ratio:.2f}%")

    if heavy_rain_count > 0:
        print(f"强降水最小值: {np.min(all_labels_flat[heavy_rain_mask]):.4f} mm/h")
        print(f"强降水最大值: {np.max(all_labels_flat[heavy_rain_mask]):.4f} mm/h")
        print(f"强降水平均值: {np.mean(all_labels_flat[heavy_rain_mask]):.4f} mm/h")
        print(f"强降水中位数: {np.median(all_labels_flat[heavy_rain_mask]):.4f} mm/h")
    else:
        print("警告: 测试集中没有强降水样本！")

    # 按预测步长统计
    print("\n" + "="*70)
    print("按预测步长统计:")
    print("-"*70)
    print(f"{'Step':<8} {'样本数':<10} {'>=threshold':<12} {'比例':<10} {'最大值':<10} {'平均值':<10}")
    print("-"*70)

    for step in range(pred_len):
        step_labels = np.array(all_labels_by_step[step])
        step_heavy = np.sum(step_labels >= threshold)
        step_heavy_ratio = step_heavy / len(step_labels) * 100 if len(step_labels) > 0 else 0

        print(f"T+{step+1:<6} {len(step_labels):<10} {step_heavy:<12} "
              f"{step_heavy_ratio:<9.2f}% {np.max(step_labels):<9.4f} {np.mean(step_labels):<9.4f}")

    # 分档统计
    print("\n" + "="*70)
    print("雨强分布（分档统计）:")
    print("-"*70)
    bins = [0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, np.inf]
    bin_labels = ['无雨(<0.1)', '小雨(0.1-1)', '小到中雨(1-5)', '中雨(5-10)',
                  '大雨(10-20)', '暴雨(20-50)', '大暴雨(50-100)', '特大暴雨(>=100)']

    hist, _ = np.histogram(all_labels_flat, bins=bins)
    print(f"{'雨强范围':<20} {'样本数':<12} {'比例':<10}")
    print("-"*50)
    for label, count in zip(bin_labels, hist):
        ratio = count / len(all_labels_flat) * 100
        print(f"{label:<20} {count:<12} {ratio:<9.2f}%")

    print("\n" + "="*70)
    print("诊断完成！")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='检查测试集数据分布')

    # 数据配置
    parser.add_argument('--site_name', type=str, default='W2129_buligou',
                        choices=['W2127_Haichaoba', 'W2128_Haichaoyinsi', 'W2129_buligou'],
                        help='站点名称')
    parser.add_argument('--data_dir', type=str, default=str(PROJECT_ROOT / 'data' / '图像preview3_Gap15_Len30'),
                        help='数据目录')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='输入序列长度（分钟）')
    parser.add_argument('--pred_len', type=int, default=5,
                        help='预测序列长度（分钟）')
    parser.add_argument('--stride', type=int, default=1,
                        help='滑动窗口步长')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--threshold', type=float, default=20.0,
                        help='强降水阈值（mm/h）')

    args = parser.parse_args()

    check_test_distribution(
        site_name=args.site_name,
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
