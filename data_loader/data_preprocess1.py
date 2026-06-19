import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import re
from pathlib import Path

# ==========================================
# 可视化雨滴谱5个物理量，包括雨强、加权直径、数浓度对数、含水量、反射率因子
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
file_path = str(DATA_DIR / 'W2129_卜里沟_参数.xlsx')  # 输入文件

filename = os.path.splitext(os.path.basename(file_path))[0]
# 提取站点代码（文件名第一部分，如 W2129）
site_code = filename.split('_')[0]
# 站点名称映射（可根据实际情况调整）
site_name_map = {
    'W2127': 'W2127_Haichaoba',
    'W2128': 'W2128_Haichaoyinsi',
    'W2129': 'W2129_buligou'
}
site_name = site_name_map.get(site_code, site_code)  # 如果找不到映射，使用站点代码

# 输出路径（以站点名称命名子文件夹）
output_dir = DATA_DIR / '图像preview' / site_name

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 图表全局设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 数据处理 (Data Processing)
# ==========================================
print(f"Reading file: {file_path} ...")
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    exit()


# 定义清洗时间的函数 (只定义一次)
def clean_datetime_string(s):
    """去除中文星期格式 (兼容 '星期' 和 '周')"""
    if pd.isna(s):
        return s
    # 正则：匹配空格 + (星期 或 周) + (一到日) + 空格，替换为空
    return re.sub(r'\s*(?:星期|周)[一二三四五六日天]\s*', ' ', str(s)).strip()


# 假设第一列始终为时间列 (根据你的数据结构)
time_col = df.columns[0]
print(f"Processing time column: '{time_col}'")

# 1. 清洗字符串 -> 2. 转datetime -> 3. 去除无效值
df[time_col] = df[time_col].apply(clean_datetime_string)
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

# 检查并移除无法解析的时间行
if df[time_col].isna().any():
    print(f"Warning: Removing {df[time_col].isna().sum()} invalid time rows.")
    df = df.dropna(subset=[time_col])

if df.empty:
    print("Error: No valid data left.")
    exit()

# 设置索引、排序、去重
df = df.set_index(time_col).sort_index()
df = df[~df.index.duplicated(keep='first')]

# 定义英文标签 (按顺序对应第1列到第5列数据)
english_labels = [
    'R (mm/h)',          # 雨强
    'D (mm)',            # 加权直径
    'Log(N)',            # 数浓度对数
    'LWC (kg/m³)',       # 含水量
    'Z (mm⁶/m³)'         # 反射率因子
]

# 校验列数
if len(df.columns) < 5:
    print("Error: Data columns are less than 5.")
    exit()

# ==========================================
# 3. 绘图循环 (Plotting Loop)
# ==========================================
unique_days = df.index.normalize().unique()
print(f"Found {len(unique_days)} days. Starting visualization...")

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for day in unique_days:
    day_str = day.strftime('%Y-%m-%d')
    print(f"Processing: {day_str}")

    # 获取当日数据 (使用loc切片)
    # 如果当日无数据(虽然unique_days里有，但为了保险)，跳过
    try:
        daily_data = df.loc[day_str]
        if isinstance(daily_data, pd.Series):
            daily_data = daily_data.to_frame().T
    except KeyError:
        continue

    # 构建完整的24小时时间轴 (1分钟分辨率)并重采样
    full_time_idx = pd.date_range(start=day, end=day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq='1min')
    plot_data = daily_data.reindex(full_time_idx)

    # 创建画布
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'{day_str}', fontsize=16, y=0.94)

    # 循环绘制5个子图
    # zip可以直接同时遍历：子图对象、数据列名、英文标签、颜色
    for ax, col_name, label, color in zip(axes, df.columns[:5], english_labels, colors):
        ax.plot(plot_data.index, plot_data[col_name], color=color, linewidth=1.2)
        ax.set_ylabel(label, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)

    # X轴格式化 (只设置最底下的一个)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel('Time (UTC/Local)', fontsize=12)
    axes[-1].set_xlim([full_time_idx[0], full_time_idx[-1]])

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    # 保存
    plt.savefig(output_dir / f"{day_str}.png", dpi=150)
    plt.close(fig)

print("All plots saved successfully.")
