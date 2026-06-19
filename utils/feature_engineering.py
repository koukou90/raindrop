import numpy as np
import pickle
import os


class FeatureScaler:
    """特征归一化和标准化工具类"""
    
    def __init__(self):
        self.conc_min = None
        self.conc_max = None
        self.vel_mean = None
        self.vel_std = None
        self.phys_mean = None
        self.phys_std = None
        
    def fit(self, conc_data, vel_data, phys_data):
        """
        在训练集上计算统计量
        
        Args:
            conc_data: 数浓度数据 (N, T, 32)
            vel_data: 速度数据 (N, T, 32)
            phys_data: 物理量数据 (N, T, 5)
        """
        # 对数浓度：计算 min/max for Min-Max Scaling
        self.conc_min = np.nanmin(conc_data, axis=(0, 1), keepdims=True)
        self.conc_max = np.nanmax(conc_data, axis=(0, 1), keepdims=True)
        
        # 速度：计算 mean/std for Z-Score Standardization
        self.vel_mean = np.nanmean(vel_data, axis=(0, 1), keepdims=True)
        self.vel_std = np.nanstd(vel_data, axis=(0, 1), keepdims=True)
        
        # 物理量：计算 mean/std for Z-Score Standardization
        self.phys_mean = np.nanmean(phys_data, axis=(0, 1), keepdims=True)
        self.phys_std = np.nanstd(phys_data, axis=(0, 1), keepdims=True)
        
        # 防止除零
        self.vel_std = np.where(self.vel_std == 0, 1.0, self.vel_std)
        self.phys_std = np.where(self.phys_std == 0, 1.0, self.phys_std)
        
        return self
    
    def transform(self, conc_data, vel_data, phys_data):
        """
        应用归一化/标准化变换
        
        Returns:
            conc_scaled, vel_scaled, phys_scaled
        """
        # 数浓度：Min-Max Scaling to [0, 1]
        conc_range = self.conc_max - self.conc_min
        # 防止除零：如果范围为零（所有值相同），设为1.0避免除零错误
        # 此时 conc_data - self.conc_min = 0，除以1.0后结果仍为0，符合预期
        conc_range = np.where(conc_range == 0, 1.0, conc_range)
        conc_scaled = (conc_data - self.conc_min) / conc_range
        
        # 速度：Z-Score Standardization
        vel_scaled = (vel_data - self.vel_mean) / self.vel_std
        
        # 物理量：Z-Score Standardization
        phys_scaled = (phys_data - self.phys_mean) / self.phys_std
        
        return conc_scaled, vel_scaled, phys_scaled
    
    def fit_transform(self, conc_data, vel_data, phys_data):
        """拟合并变换"""
        self.fit(conc_data, vel_data, phys_data)
        return self.transform(conc_data, vel_data, phys_data)
    
    def save(self, filepath):
        """保存scaler参数"""
        # 确保目录存在
        dir_path = os.path.dirname(filepath)
        if dir_path:  # 如果目录路径不为空
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'conc_min': self.conc_min,
                'conc_max': self.conc_max,
                'vel_mean': self.vel_mean,
                'vel_std': self.vel_std,
                'phys_mean': self.phys_mean,
                'phys_std': self.phys_std
            }, f)
    
    def load(self, filepath):
        """加载scaler参数"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            self.conc_min = params['conc_min']
            self.conc_max = params['conc_max']
            self.vel_mean = params['vel_mean']
            self.vel_std = params['vel_std']
            self.phys_mean = params['phys_mean']
            self.phys_std = params['phys_std']
        return self


def apply_log_transform(conc_data, epsilon=1e-6):
    """
    对数浓度进行对数变换
    
    Args:
        conc_data: 原始数浓度数据
        epsilon: 防止log(0)的小常数
        
    Returns:
        log_conc: 对数变换后的数据
    """
    return np.log10(conc_data + epsilon)


def inverse_log_transform(log_conc, epsilon=1e-6):
    """
    对数变换的逆操作
    
    Args:
        log_conc: 对数变换后的数据
        epsilon: 与正向变换时使用的小常数
        
    Returns:
        conc: 原始数浓度数据（确保非负）
    """
    result = np.power(10, log_conc) - epsilon
    # 确保结果非负（防止数值误差导致负值）
    return np.maximum(result, 0.0)