import pandas as pd
import openpyxl

# 读取 CSV 文件
csv_file = r'data\Event2012\open_set\20\20_es-False_ea-True_all_clustered_data.csv'
df = pd.read_csv(csv_file)

# 将数据保存为 XLS 文件
xls_file = r'data\Event2012\open_set\20\20_es-False_ea-True_all_clustered_data.xlsx'
df.to_excel(xls_file, index=False)

print(f"已将 {csv_file} 转换为 {xls_file}")