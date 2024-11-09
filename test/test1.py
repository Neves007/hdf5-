import pandas as pd

# 创建一个示例 DataFrame
data = {
    'column1': [1, 2, 3, 4],
    'column2': ['a', 'b', 'c', 'd'],
    'column3': [0.1, 0.2, 0.3, 0.4]
}
df = pd.DataFrame(data)

# 存储到 HDF5 文件的 group1 分组中
df.to_hdf('data.h5', key='/group1/my_data', mode='w')
# 从 HDF5 文件的 group1 中读取数据
df_loaded = pd.read_hdf('data.h5', key='/group1/my_data')
print(df_loaded)
