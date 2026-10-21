import pandas as pd
import pyarrow.parquet as pq

path = "/Users/skywalker/Documents/exe/episode_000000.parquet"

# 1. 查看 Parquet 元数据和字段类型
table = pq.read_table(path)
print("Schema:")
print(table.schema)

# 2. 加载数据
df = table.to_pandas()

print("\n数据形状（行数、列数）:")
print(df.shape)

print("\n所有字段:")
print(df.columns.tolist())

print("\n各字段的数据类型:")
print(df.dtypes)

# 3. 查看前 5 行
print("\n前 5 行:")
print(df.head().to_string(index=False))

# 4. 查看第一条样本
print("\n第一条样本:")
for key, value in df.iloc[0].items():
    print(f"{key}: {str(value)[:300]}")



df = pd.read_parquet(path)
df.to_csv("episode_000000.csv", index=False)

print("已导出 episode_000000.csv")