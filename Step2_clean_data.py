import pandas as pd
import matplotlib.pyplot as plt
import os

# 0. 确保输出目录存在
os.makedirs('outputs', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. 读取数据
df = pd.read_excel('data/wind_speed_data.xlsx')
df.columns = df.columns.str.strip() # 去除列名可能存在的空格

# 2. 剔除 qc.7 标记为 'I' 的异常值（必须删！）
print("=== 原始数据形状 ===", df.shape)
df = df[df['qc.7'] != 'I'].copy()
print("=== 剔除异常值后形状 ===", df.shape)

# 3. 处理时间列（踩坑警告：你的 Hour 是 0100, 0200 这种格式，需要特殊处理）
# 先补零成4位字符串，然后提取前两位作为小时
df['Hour_str'] = df['Hour (PST)'].astype(str).str.zfill(4)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['Datetime'] = df['Date'] + pd.to_timedelta(df['Hour_str'].str[:2].astype(int), unit='h')

# 4. 设置为时间索引，并剔掉多余的列
df.set_index('Datetime', inplace=True)
drop_cols = ['Stn Id', 'Stn Name', 'CIMIS Region', 'Date', 'Hour (PST)', 'Jul', 'Hour_str'] + [f'qc.{i}' for i in range(1, 10)] + ['qc']
df_clean = df.drop(columns=drop_cols, errors='ignore')

# 5. 重采样！把“4天一测”强行变成“1小时一测”的连续数据（用线性插值补齐）
print("=== 重采样为1小时频率前 ===", df_clean.shape)
df_clean = df_clean.resample('1h').mean().interpolate()
print("=== 重采样为1小时频率后 ===", df_clean.shape)

# 6. 可视化1：全局风速趋势图（看宏观规律）
plt.figure(figsize=(15, 5))
plt.plot(df_clean.index, df_clean['Wind Speed (mph)'], label='Wind Speed')
plt.title('Wind Speed over Time (Full Dataset)')
plt.xlabel('Time')
plt.ylabel('Wind Speed (mph)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/wind_speed_trend_full.png')
plt.close() # 关掉画布，避免阻塞

# 7. 可视化2：局部放大图（只看前100个小时，看微观波动）
plt.figure(figsize=(12, 4))
plt.plot(df_clean['Wind Speed (mph)'].iloc[:100], marker='o', linestyle='-', color='orange')
plt.title('Wind Speed (First 100 Points)')
plt.xlabel('Time')
plt.ylabel('Wind Speed (mph)')
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/wind_speed_zoomed.png')
plt.close()

# 8. 保存清洗后的数据，方便下一步丢给模型
df_clean.to_csv('data/wind_speed_clean.csv')
print("\n✅ 数据清洗完成，已保存至 data/wind_speed_clean.csv")
print("\n=== 清洗后的前5行 ===")
print(df_clean.head())