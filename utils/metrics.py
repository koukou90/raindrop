import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


def rmse(y_true, y_pred):
    """均方根误差"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """平均绝对误差"""
    return mean_absolute_error(y_true, y_pred)


def correlation_coefficient(y_true, y_pred):
    """皮尔逊相关系数"""
    if len(y_true) < 2:
        return 0.0
    corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return corr if not np.isnan(corr) else 0.0


def peak_mae(y_true, y_pred, threshold=20.0):
    """
    强降水峰值平均绝对误差
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        threshold: 强降水阈值 (mm/h)
    """
    mask = y_true >= threshold
    if np.sum(mask) == 0:
        return np.nan
    return mae(y_true[mask], y_pred[mask])


def mse(y_true, y_pred):
    """均方误差"""
    return mean_squared_error(y_true, y_pred)


def r_squared(y_true, y_pred):
    """
    决定系数 (R²)
    
    R² = 1 - (SS_res / SS_tot)
    其中 SS_res 是残差平方和，SS_tot 是总平方和
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else np.nan
    
    r2 = 1 - (ss_res / ss_tot)
    return r2 if not np.isnan(r2) else 0.0


def mape(y_true, y_pred):
    """
    平均绝对百分比误差 (MAPE)
    
    注意：当真实值为0时，MAPE会变为无穷大，因此需要处理零值情况
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 避免除以零：当真实值为0时，如果预测值也为0，则误差为0；否则跳过该样本
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan  # 所有真实值都为0，无法计算MAPE
    
    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return np.mean(percentage_errors)


def bias(y_true, y_pred):
    """
    偏差 (Bias)
    
    预测值的平均偏差，用于判断是否存在系统性高估或低估
    正值表示平均预测值高于真实值（高估），负值表示低估
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    return np.mean(y_pred - y_true)


def nse(y_true, y_pred):
    """
    Nash-Sutcliffe效率系数 (NSE)
    
    NSE = 1 - (SS_res / SS_tot)
    其中 SS_res 是残差平方和，SS_tot 是真实值相对于均值的平方和
    
    范围: (-∞, 1]
    - NSE = 1: 完美预测
    - NSE = 0: 模型预测与真实值均值相同
    - NSE < 0: 模型预测比简单使用均值更差
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else np.nan
    
    nse_value = 1 - (ss_res / ss_tot)
    return nse_value if not np.isnan(nse_value) else np.nan


def binary_classification_metrics(y_true, y_pred, threshold=20.0):
    """
    二分类评价指标：POD, FAR, CSI
    
    Args:
        y_true: 真实雨强
        y_pred: 预测雨强
        threshold: 强降水阈值 (mm/h)
        
    Returns:
        dict: {'POD': ..., 'FAR': ..., 'CSI': ...}
    """
    # 二值化
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # 混淆矩阵元素
    hits = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    misses = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    false_alarms = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    
    # POD (Probability of Detection)
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    
    # FAR (False Alarm Ratio)
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    
    # CSI (Critical Success Index)
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
    
    return {
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'Hits': hits,
        'Misses': misses,
        'FalseAlarms': false_alarms
    }


def compute_all_metrics(y_true, y_pred, threshold=20.0):
    """
    计算所有评价指标
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 回归指标
    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'CC': correlation_coefficient(y_true, y_pred),
        'R2': r_squared(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'Bias': bias(y_true, y_pred),
        'NSE': nse(y_true, y_pred),
        'Peak_MAE': peak_mae(y_true, y_pred, threshold)
    }
    
    # 分类指标
    # class_metrics = binary_classification_metrics(y_true, y_pred, threshold)
    # metrics.update(class_metrics)
    
    return metrics


def print_metrics(metrics, prefix=''):
    """
    格式化打印指标
    
    指标说明：
    - RMSE (均方根误差): 越小越好，衡量预测值与真实值的整体偏差
    - MAE (平均绝对误差): 越小越好，衡量预测值与真实值的平均偏差
    - MSE (均方误差): 越小越好，RMSE的平方，更强调大误差
    - CC (相关系数): 越大越好，范围[-1, 1]，衡量预测值与真实值的线性相关程度
    - R² (决定系数): 越大越好，范围通常[0, 1]，衡量模型解释方差的比例
    - MAPE (平均绝对百分比误差): 越小越好，相对误差，对低值敏感
    - Bias (偏差): 越接近0越好，正值表示高估，负值表示低估
    - NSE (Nash-Sutcliffe效率系数): 越大越好，范围(-∞, 1]，常用于水文/气象模型
    - Peak-MAE (强降水峰值MAE): 越小越好，衡量强降水事件(≥阈值)的预测误差
    - POD (命中率/检测率): 越大越好，范围[0, 1]，实际发生强降水时预测到的比例
    - FAR (误报率): 越小越好，范围[0, 1]，预测强降水但实际未发生的比例
    - CSI (临界成功指数): 越大越好，范围[0, 1]，综合衡量强降水的预测准确性
    """
    print(f"\n{prefix}Evaluation Metrics:")
    print(f"  RMSE: {metrics['RMSE']:.4f} (越小越好)")
    print(f"  MAE: {metrics['MAE']:.4f} (越小越好)")
    print(f"  MSE: {metrics['MSE']:.4f} (越小越好)")
    print(f"  CC: {metrics['CC']:.4f} (越大越好, 范围[-1,1])")
    print(f"  R2: {metrics['R2']:.4f} (越大越好, 范围通常[0,1])")
    if not np.isnan(metrics['MAPE']):
        print(f"  MAPE: {metrics['MAPE']:.4f}% (越小越好)")
    else:
        print(f"  MAPE: N/A (真实值全为0)")
    print(f"  Bias: {metrics['Bias']:.4f} (越接近0越好)")
    if not np.isnan(metrics['NSE']):
        print(f"  NSE: {metrics['NSE']:.4f} (越大越好, 范围(-inf,1])")
    else:
        print(f"  NSE: N/A (无法计算)")
    if not np.isnan(metrics['Peak_MAE']):
        print(f"  Peak-MAE: {metrics['Peak_MAE']:.4f} (越小越好)")
    else:
        print(f"  Peak-MAE: N/A (no heavy rain events)")
    # print(f"  POD: {metrics['POD']:.4f} (越大越好, 范围[0,1])")
    # print(f"  FAR: {metrics['FAR']:.4f} (越小越好, 范围[0,1])")
    # print(f"  CSI: {metrics['CSI']:.4f} (越大越好, 范围[0,1])")