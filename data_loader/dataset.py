import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

# 本地模块导入
from utils.feature_engineering import FeatureScaler, apply_log_transform


class RainDropDataset(Dataset):
    """雨滴谱数据集"""
    
    def __init__(self, conc_data, vel_data, phys_data, labels, aux_labels=None):
        """
        Args:
            conc_data: 数浓度特征 (N, seq_len, 32)
            vel_data: 速度特征 (N, seq_len, 32)
            phys_data: 物理量特征 (N, seq_len, 5)
            labels: 标签 (N, pred_len)
            aux_labels: 辅助标签 (N, pred_len, 4)，对应未来 Dm/LogNw/LWC/Z
        """
        self.conc_data = torch.FloatTensor(conc_data)
        self.vel_data = torch.FloatTensor(vel_data)
        self.phys_data = torch.FloatTensor(phys_data)
        self.labels = torch.FloatTensor(labels)
        self.aux_labels = torch.FloatTensor(aux_labels) if aux_labels is not None else None
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            'conc': self.conc_data[idx],
            'vel': self.vel_data[idx],
            'phys': self.phys_data[idx],
            'label': self.labels[idx]
        }
        if self.aux_labels is not None:
            sample['aux_label'] = self.aux_labels[idx]
        return sample


def load_preprocessed_data(site_name, data_dir='data/图像preview3_Gap15_Len30'):
    """
    加载预处理后的数据文件
    
    Args:
        site_name: 站点名称，如 'W2127_Haichaoba'
        data_dir: 数据目录
        
    Returns:
        df_dsd: 雨滴谱数据 (columns: Timestamp, data1, data2, ..., data64)
        df_params: 物理参数数据 (columns: Timestamp, RainRate, Dm, LogNw, LWC, Z)
    """
    dsd_path = os.path.join(data_dir, f'{site_name}', f'{site_name}_dsd.csv')
    params_path = os.path.join(data_dir, f'{site_name}', f'{site_name}_params.csv')
    
    df_dsd = pd.read_csv(dsd_path)
    df_params = pd.read_csv(params_path)
    
    # 转换时间列
    df_dsd['Timestamp'] = pd.to_datetime(df_dsd['Timestamp'])
    df_params['Timestamp'] = pd.to_datetime(df_params['Timestamp'])
    
    return df_dsd, df_params


def extract_dsd_features(df_dsd):
    """
    从DSD数据中提取数浓度和速度
    
    根据研究方案，雨滴谱数据有64列：
    - 前32列：数浓度 N(D)
    - 后32列：下落速度 V(D)
    
    Returns:
        conc_array: (T, 32) 数浓度
        vel_array: (T, 32) 速度
    """
    data_cols = [col for col in df_dsd.columns if col.startswith('data')]
    dsd_data = df_dsd[data_cols].values  # (T, 64)
    
    # 分离数浓度和速度
    conc_array = dsd_data[:, :32]  # 前32列
    vel_array = dsd_data[:, 32:]   # 后32列
    
    return conc_array, vel_array


def create_sliding_window_samples(conc, vel, phys, rain_rate,
                                  seq_len=10, pred_len=5, stride=1,
                                  return_aux_phys=False):
    """
    滑动窗口生成样本
    
    Args:
        conc: 数浓度数组 (T, 32)
        vel: 速度数组 (T, 32)
        phys: 物理量数组 (T, 5)
        rain_rate: 雨强数组 (T,)
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        stride: 滑动步长
        
    Returns:
        X_conc, X_vel, X_phys, Y
    """
    T = len(rain_rate)
    samples_conc = []
    samples_vel = []
    samples_phys = []
    samples_label = []
    samples_aux = []
    
    for i in range(0, T - seq_len - pred_len + 1, stride):
        # 输入窗口
        input_conc = conc[i:i+seq_len]      # (10, 32)
        input_vel = vel[i:i+seq_len]        # (10, 32)
        input_phys = phys[i:i+seq_len]      # (10, 5)
        
        # 标签窗口：未来5分钟雨强
        label = rain_rate[i+seq_len:i+seq_len+pred_len]  # (5,)
        aux_label = phys[i+seq_len:i+seq_len+pred_len, 1:] if return_aux_phys else None
        
        # 过滤掉包含NaN的样本
        if (not np.any(np.isnan(input_conc)) and 
            not np.any(np.isnan(input_vel)) and
            not np.any(np.isnan(input_phys)) and
            not np.any(np.isnan(label)) and
            (not return_aux_phys or not np.any(np.isnan(aux_label)))):
            
            samples_conc.append(input_conc)
            samples_vel.append(input_vel)
            samples_phys.append(input_phys)
            samples_label.append(label)
            if return_aux_phys:
                samples_aux.append(aux_label)
    
    if len(samples_label) == 0:
        if return_aux_phys:
            return None, None, None, None, None
        return None, None, None, None
    
    X_conc = np.array(samples_conc)   # (N, 10, 32)
    X_vel = np.array(samples_vel)     # (N, 10, 32)
    X_phys = np.array(samples_phys)   # (N, 10, 5)
    Y = np.array(samples_label)       # (N, 5)
    if return_aux_phys:
        Y_aux = np.array(samples_aux)  # (N, 5, 4)
        return X_conc, X_vel, X_phys, Y, Y_aux

    return X_conc, X_vel, X_phys, Y


def split_by_events(df_params, train_ratio=0.8, val_ratio=0.1):
    """
    按降水事件划分数据集（严格时间顺序）
    
    Args:
        df_params: 参数DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        train_events, val_events, test_events: 每个都是事件列表，每个事件包含start_idx和end_idx
    """
    # 识别降水事件（非NaN的连续段）
    is_valid = ~df_params['RainRate'].isna()
    event_id = (is_valid != is_valid.shift()).cumsum()
    
    events = []
    for eid, group in df_params[is_valid].groupby(event_id):
        if len(group) > 0:
            events.append({
                'event_id': eid,
                'start_idx': group.index[0],
                'end_idx': group.index[-1],
                'length': len(group)
            })
    
    # 按时间顺序划分
    n_events = len(events)
    n_train = int(n_events * train_ratio)
    n_val = int(n_events * val_ratio)
    
    train_events = events[:n_train]
    val_events = events[n_train:n_train+n_val]
    test_events = events[n_train+n_val:]
    
    return train_events, val_events, test_events


def prepare_datasets(site_name, data_dir='data/图像preview3_Gap15_Len30',
                     seq_len=10, pred_len=5, stride=1,
                     train_ratio=0.8, val_ratio=0.1,
                     include_aux_phys=False):
    """
    准备训练/验证/测试数据集
    
    Returns:
        datasets: {'train': dataset, 'val': dataset, 'test': dataset}
        scaler: 特征缩放器
    """
    print(f"\nLoading data for {site_name}...")
    df_dsd, df_params = load_preprocessed_data(site_name, data_dir)
    
    # 提取特征
    conc, vel = extract_dsd_features(df_dsd)
    phys = df_params[['RainRate', 'Dm', 'LogNw', 'LWC', 'Z']].values
    rain_rate = df_params['RainRate'].values
    
    # 对数浓度进行对数变换
    conc = apply_log_transform(conc)
    
    # 按事件划分
    train_events, val_events, test_events = split_by_events(df_params, train_ratio, val_ratio)
    
    print(f"Train events: {len(train_events)} events")
    print(f"Val events: {len(val_events)} events")
    print(f"Test events: {len(test_events)} events")
    
    # 生成样本（逐事件生成，避免跨越事件边界）
    datasets = {}
    scaler = FeatureScaler()
    
    for split_name, events_list in [('train', train_events), ('val', val_events), ('test', test_events)]:
        if len(events_list) == 0:
            datasets[split_name] = None
            continue
        
        # 收集所有事件的样本
        all_X_conc = []
        all_X_vel = []
        all_X_phys = []
        all_Y = []
        all_Y_aux = []
        
        # 对每个事件单独生成样本
        for evt_idx, evt in enumerate(events_list):
            # 提取该事件的数据
            event_indices = list(range(evt['start_idx'], evt['end_idx'] + 1))
            event_conc = conc[event_indices]
            event_vel = vel[event_indices]
            event_phys = phys[event_indices]
            event_rain = rain_rate[event_indices]
            
            # 对该事件生成滑动窗口样本
            if include_aux_phys:
                X_conc, X_vel, X_phys, Y, Y_aux = create_sliding_window_samples(
                    event_conc, event_vel, event_phys, event_rain,
                    seq_len, pred_len, stride,
                    return_aux_phys=True
                )
            else:
                X_conc, X_vel, X_phys, Y = create_sliding_window_samples(
                    event_conc, event_vel, event_phys, event_rain,
                    seq_len, pred_len, stride
                )
            
            # 如果该事件生成了样本，则添加到总列表中
            if X_conc is not None and len(X_conc) > 0:
                all_X_conc.append(X_conc)
                all_X_vel.append(X_vel)
                all_X_phys.append(X_phys)
                all_Y.append(Y)
                if include_aux_phys:
                    all_Y_aux.append(Y_aux)
        
        # 合并所有事件的样本
        if len(all_X_conc) == 0:
            datasets[split_name] = None
            print(f"{split_name.capitalize()} dataset: 0 samples (no valid events)")
            continue
        
        X_conc = np.concatenate(all_X_conc, axis=0)
        X_vel = np.concatenate(all_X_vel, axis=0)
        X_phys = np.concatenate(all_X_phys, axis=0)
        Y = np.concatenate(all_Y, axis=0)
        Y_aux = np.concatenate(all_Y_aux, axis=0) if include_aux_phys else None
        
        # 训练集上拟合scaler
        if split_name == 'train':
            X_conc, X_vel, X_phys = scaler.fit_transform(X_conc, X_vel, X_phys)
        else:
            X_conc, X_vel, X_phys = scaler.transform(X_conc, X_vel, X_phys)
        
        datasets[split_name] = RainDropDataset(X_conc, X_vel, X_phys, Y, aux_labels=Y_aux)
        print(f"{split_name.capitalize()} dataset: {len(datasets[split_name])} samples from {len(all_X_conc)} events")
    
    return datasets, scaler, {'train_events': train_events, 'val_events': val_events, 'test_events': test_events}