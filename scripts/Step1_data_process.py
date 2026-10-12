import pandas as pd

# 确认一下你的 excel 到底放哪了，如果在 data 文件夹里，就这么写
file_path = 'data/wind_speed_data.xlsx'
df = pd.read_excel(file_path)

# 打印前5行
print("=== 前5行数据 ===")
print(df.head())

# 打印数据信息（看有没有缺失值）
print("\n=== 数据基础信息 ===")
df.info()

# 统计缺失值
print("\n=== 各列缺失值统计 ===")
print(df.isnull().sum())
