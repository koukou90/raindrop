import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import re
from pathlib import Path

# ==========================================
# 对比雨量计数据与雨滴谱的雨强数据
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# 文件路径
path_raindrop = str(DATA_DIR / 'W2129_卜里沟_参数.xlsx')           # 雨滴谱数据
path_station1 = str(DATA_DIR / '海潮音寺后山气象数据2020.csv')      # 气象站1
path_station2 = str(DATA_DIR / '三座窑气象数据2020.csv')            # 气象站2

raindrop_filename = os.path.splitext(os.path.basename(path_raindrop))[0]
# 提取站点代码（文件名第一部分，如 W2127）
site_code = raindrop_filename.split('_')[0]
# 站点名称映射（可根据实际情况调整）
site_name_map = {
    'W2127': 'W2127_Haichaoba',
    'W2128': 'W2128_Haichaoyinsi',
    'W2129': 'W2129_buligou'
}
site_name = site_name_map.get(site_code, site_code)  # 如果找不到映射，使用站点代码

# 输出路径（以站点名称命名子文件夹）
output_dir = DATA_DIR / '图像preview_雨量计对比' / site_name
os.makedirs(output_dir, exist_ok=True)

# 绘图设置 (英文显示)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 数据读取函数定义
# ==========================================


def clean_datetime_string(s):
    """(复用) 清洗雨滴谱Excel中的中文时间格式"""
    if pd.isna(s):
        return s
    # 移除 '星期X' 或 '周X'
    return re.sub(r'\s*(?:星期|周)[一二三四五六日天]\s*', ' ', str(s)).strip()


def load_raindrop_data(filepath):
    """读取雨滴谱Excel，只提取'雨强'"""
    print(f"Loading Raindrop data: {filepath} ...")
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

    # 处理时间列 (假设第1列)
    time_col = df.columns[0]
    df[time_col] = df[time_col].apply(clean_datetime_string)
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])

    # 设置索引并排序
    df = df.set_index(time_col).sort_index()
    # 去重
    df = df[~df.index.duplicated(keep='first')]

    # 提取雨强列 (根据你的描述，假设列名包含'雨强')
    # 如果列名固定，可以直接用 df['雨强']，这里为了健壮性进行模糊查找或按位置
    col_rain_rate = None
    for col in df.columns:
        if '雨强' in col:
            col_rain_rate = col
            break

    if col_rain_rate is None:
        # 如果找不到中文'雨强'，尝试按位置：第2列(索引1)
        col_rain_rate = df.columns[1]

    return df[[col_rain_rate]].rename(columns={col_rain_rate: 'Rain Rate'})


def load_station_data(filepath):
    """读取气象站CSV，只提取'雨量'"""
    print(f"Loading Station data: {filepath} ...")
    try:
        # CSV通常可能有中文编码，尝试 gb18030 (涵盖gbk) 或 utf-8
        try:
            df = pd.read_csv(filepath, encoding='gb18030')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='utf-8')

    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

    # 根据截图，时间列名为 'Row'，格式 '20200101 00:00'
    if 'Row' not in df.columns:
        print(f"Error: Column 'Row' not found in {filepath}")
        return None

    # 解析时间格式 '%Y%m%d %H:%M'
    df['Row'] = pd.to_datetime(df['Row'], format='%Y%m%d %H:%M', errors='coerce')
    df = df.dropna(subset=['Row'])

    # 设置索引
    df = df.set_index('Row').sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # 提取雨量列
    col_rain_amt = None
    for col in df.columns:
        if '雨量' in col:
            col_rain_amt = col
            break

    if col_rain_amt is None:
        print(f"Warning: '雨量' column not found in {filepath}. Using last column.")
        col_rain_amt = df.columns[-1]

    return df[[col_rain_amt]].rename(columns={col_rain_amt: 'Rain Amount'})


# ==========================================
# 3. 主处理流程
# ==========================================

# 3.1 读取三个文件
df_drop = load_raindrop_data(path_raindrop)
df_st1 = load_station_data(path_station1)  # 海潮音寺
df_st2 = load_station_data(path_station2)  # 三座窑

if any(d is None for d in [df_drop, df_st1, df_st2]):
    print("One or more files failed to load. Exiting.")
    exit()

# 3.2 提取日期并求交集 (Intersection)
days_drop = set(df_drop.index.normalize())
days_st1 = set(df_st1.index.normalize())
days_st2 = set(df_st2.index.normalize())

# 找出三个数据集都存在的日期
common_days = sorted(list(days_drop & days_st1 & days_st2))

print(f"\n--- Date Alignment ---")
print(f"Raindrop data days: {len(days_drop)}")
print(f"Station 1 data days: {len(days_st1)}")
print(f"Station 2 data days: {len(days_st2)}")
print(f"Common days (Intersection): {len(common_days)}")

if len(common_days) == 0:
    print("No common dates found among the three datasets.")
    exit()

# 3.3 循环绘图
print("\nStarting plotting loop...")

for day in common_days:
    day_str = day.strftime('%Y-%m-%d')
    print(f"Processing: {day_str}")

    # 构造当日完整的分钟级时间轴 (用于对齐X轴)
    full_time_idx = pd.date_range(start=day, end=day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq='1min')

    # 截取当日数据并重采样 (Reindex)
    # 1. 雨滴谱 (雨强)
    try:
        data_drop = df_drop.loc[day_str]
        # 防止只有一行数据变为Series
        if isinstance(data_drop, pd.Series):
            data_drop = data_drop.to_frame().T
        plot_drop = data_drop.reindex(full_time_idx)  # 缺失值自动填充NaN
    except KeyError:
        # 理论上求过交集不应该进这里，但为了保险
        continue

    # 2. 气象站1 (海潮音寺)
    try:
        data_st1 = df_st1.loc[day_str]
        if isinstance(data_st1, pd.Series):
            data_st1 = data_st1.to_frame().T
        plot_st1 = data_st1.reindex(full_time_idx)
    except KeyError:
        continue

    # 3. 气象站2 (三座窑)
    try:
        data_st2 = df_st2.loc[day_str]
        if isinstance(data_st2, pd.Series):
            data_st2 = data_st2.to_frame().T
        plot_st2 = data_st2.reindex(full_time_idx)
    except KeyError:
        continue

    # 绘制图形 (3行1列)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

    fig.suptitle(f'Rain Parameter Comparison - {day_str}', fontsize=16, y=0.95)

    # 子图1: 雨滴谱 - 雨强
    ax1 = axes[0]
    ax1.plot(plot_drop.index, plot_drop['Rain Rate'], color='#1f77b4', label='Disdrometer', linewidth=1.5)
    ax1.set_ylabel('Rain Rate (mm/h)', fontsize=10, fontweight='bold')
    ax1.set_title(f'W2127 Haichaoba - Rain Rate', loc='left', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    # ax1.legend(loc='upper right')

    # 子图2: 气象站1 - 雨量
    ax2 = axes[1]
    ax2.plot(plot_st1.index, plot_st1['Rain Amount'], color='#ff7f0e', label='Station', linewidth=1.5)
    ax2.set_ylabel('Rain Amount (mm)', fontsize=10, fontweight='bold')
    ax2.set_title(f'Haichaoyinsi (Station 1) - Rain Amount', loc='left', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 子图3: 气象站2 - 雨量
    ax3 = axes[2]
    ax3.plot(plot_st2.index, plot_st2['Rain Amount'], color='#2ca02c', label='Station', linewidth=1.5)
    ax3.set_ylabel('Rain Amount (mm)', fontsize=10, fontweight='bold')
    ax3.set_title(f'Sanzuoyao (Station 2) - Rain Amount', loc='left', fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # X轴格式化
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel('Time (UTC/Local)', fontsize=12)
    axes[-1].set_xlim([full_time_idx[0], full_time_idx[-1]])

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    # 保存图片
    save_filename = f"{day_str}.png"
    plt.savefig(output_dir / save_filename, dpi=150)
    plt.close(fig)

print("All plots saved successfully.")
