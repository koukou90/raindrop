import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import re
import numpy as np
from pathlib import Path

# ==========================================
# 1. 配置 (Configuration)
# ==========================================

# --- 核心算法参数 ---
MERGE_GAP_MIN = 15      # 最大允许间歇 (分钟)
MIN_DURATION_MIN = 30   # 最小有效事件时长 (分钟)

# --- 功能开关 ---
PLOT_ENABLED = True     # True: 画图; False: 不画图

# --- 文件路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
param_file_path = str(DATA_DIR / 'W2129_卜里沟_参数.xlsx')
raw_dsd_file_path = str(DATA_DIR / 'W2129_卜里沟_雨滴谱.xlsx')

# --- 路径处理 ---
filename = os.path.splitext(os.path.basename(param_file_path))[0]
site_code = filename.split('_')[0]
site_name_map = {
    'W2127': 'W2127_Haichaoba',
    'W2128': 'W2128_Haichaoyinsi',
    'W2129': 'W2129_buligou'
}
site_name = site_name_map.get(site_code, site_code)

# 文件夹命名包含参数
folder_name = f"图像preview3_Gap{MERGE_GAP_MIN}_Len{MIN_DURATION_MIN}"
base_output_dir = DATA_DIR / folder_name / site_name
os.makedirs(base_output_dir, exist_ok=True)

# 保存路径
param_save_path = base_output_dir / f"{site_name}_params.csv"
raw_dsd_save_path = base_output_dir / f"{site_name}_dsd.csv"

# 图表设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 2. 数据读取与预处理函数
# ==========================================
def load_and_clean_data(filepath, file_type="Param"):
    print(f"Reading {file_type} file: {filepath} ...")
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        exit()

    def clean_datetime_string(s):
        if pd.isna(s):
            return s
        return re.sub(r'\s*(?:星期|周)[一二三四五六日天]\s*', ' ', str(s)).strip()

    time_col = df.columns[0]

    df[time_col] = df[time_col].apply(clean_datetime_string)
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col).sort_index()
    df = df[~df.index.duplicated(keep='first')]

    return df


# 读取文件
df_param = load_and_clean_data(param_file_path, "Parameter")
df_raw = load_and_clean_data(raw_dsd_file_path, "Raw DSD")

# 确定列名
param_cols = df_param.columns[:5]
rain_col_name = param_cols[0]
raw_cols = df_raw.columns[:]


# ==========================================
# 3. 核心算法：生成有效事件掩码
# ==========================================
def get_event_mask(rain_series, merge_gap, min_duration):
    series_filled = rain_series.fillna(0.0)
    is_raining = series_filled > 0

    # 合并间歇
    merged_mask = is_raining.copy()
    is_dry = ~is_raining
    dry_groups = (is_dry != is_dry.shift()).cumsum()

    for _, gap_data in is_dry.groupby(dry_groups):
        if gap_data.iloc[0] == True:
            if len(gap_data) <= merge_gap:
                merged_mask.loc[gap_data.index] = True

    # 过滤短事件
    event_groups = (merged_mask != merged_mask.shift()).cumsum()
    final_mask = merged_mask.copy()

    for _, event_data in merged_mask.groupby(event_groups):
        if event_data.iloc[0] == True:
            if len(event_data) < min_duration:
                final_mask.loc[event_data.index] = False

    return final_mask


# ==========================================
# 4. 主循环
# ==========================================
common_days = sorted(list(set(df_param.index.normalize()) & set(df_raw.index.normalize())))
print(f"Found {len(common_days)} common days. Settings: Gap<={MERGE_GAP_MIN}m, Len>={MIN_DURATION_MIN}m")

all_daily_params = []
all_daily_raw = []

for day in common_days:
    day_str = day.strftime('%Y-%m-%d')
    print(f"Processing: {day_str}")

    # 1. 构造标准的1440分钟索引
    full_time_idx = pd.date_range(start=day, end=day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq='1min')

    # 2. 准备数据
    try:
        daily_param = df_param.loc[day_str]
        if isinstance(daily_param, pd.Series):
            daily_param = daily_param.to_frame().T
        daily_param_full = daily_param.reindex(full_time_idx)[param_cols]
    except KeyError:
        continue

    try:
        daily_raw = df_raw.loc[day_str]
        if isinstance(daily_raw, pd.Series):
            daily_raw = daily_raw.to_frame().T
        daily_raw_full = daily_raw.reindex(full_time_idx)[raw_cols]
    except KeyError:
        continue

    # 3. 计算掩码
    original_rain = daily_param_full[rain_col_name]
    valid_mask = get_event_mask(original_rain, merge_gap=MERGE_GAP_MIN, min_duration=MIN_DURATION_MIN)

    # 4. 处理参数
    processed_params = daily_param_full.copy()
    for col in processed_params.columns:
        processed_params[col] = processed_params[col].fillna(0.0)
        processed_params.loc[~valid_mask, col] = np.nan

    # 5. 处理原始谱
    processed_raw = daily_raw_full.copy()
    processed_raw = processed_raw.fillna(0.0)
    processed_raw.loc[~valid_mask, :] = np.nan

    # 6. 收集数据
    export_params = processed_params.copy()
    export_params.insert(0, 'Timestamp', export_params.index)
    all_daily_params.append(export_params)

    export_raw = processed_raw.copy()
    export_raw.insert(0, 'Timestamp', export_raw.index)
    all_daily_raw.append(export_raw)

    # ==========================================
    # 7. 绘图 (严格锁定 00:00 - 24:00)
    # ==========================================
    if PLOT_ENABLED:
        filtered_rain = processed_params[rain_col_name]

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), sharex=True)
        fig.suptitle(f'{day_str} Processing (Gap<={MERGE_GAP_MIN}m, Len>={MIN_DURATION_MIN}m)', fontsize=14, y=0.96)

        # 上图：原始
        axes[0].plot(full_time_idx, original_rain, color='#1f77b4', linewidth=1.5, label='Original')
        axes[0].set_title('Raw Data', loc='left')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # 下图：处理后
        axes[1].plot(full_time_idx, filtered_rain, color='#d62728', linewidth=1.5, label='Processed')
        axes[1].set_title('Processed Data', loc='left')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle=':', alpha=0.6)

        # 强制设置X轴范围为 当日00:00 到 次日00:00
        x_min = day
        x_max = day + pd.Timedelta(days=1)
        axes[-1].set_xlim(x_min, x_max)

        # 刻度设置 (每2小时一个刻度)
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[-1].set_xlabel('Time (UTC/Local)', fontsize=12)

        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        plt.savefig(base_output_dir / f"{day_str}.png", dpi=100)
        plt.close(fig)


# ==========================================
# 5. 数据导出
# ==========================================
print("Saving files...")

if all_daily_params:
    # 保存参数
    final_params = pd.concat(all_daily_params, axis=0)
    final_params.columns = ['Timestamp', 'RainRate', 'Dm', 'LogNw', 'LWC', 'Z']
    print(f"Saving Params Matrix to {param_save_path} ... shape: {final_params.shape}")
    final_params.to_csv(param_save_path, index=False, encoding='utf-8')

    # 保存原始谱
    final_raw = pd.concat(all_daily_raw, axis=0)
    num_data_cols = final_raw.shape[1] - 1
    raw_col_names = ['Timestamp'] + [f"data{i+1}" for i in range(num_data_cols)]
    final_raw.columns = raw_col_names
    print(f"Saving Raw DSD Matrix to {raw_dsd_save_path} ... shape: {final_raw.shape}")
    final_raw.to_csv(raw_dsd_save_path, index=False, encoding='utf-8')

    print("All processing done!")
else:
    print("No valid data processed.")
