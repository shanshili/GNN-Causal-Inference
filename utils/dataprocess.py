import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
"""
BJ :
Label 1
sensor(NO) 453
pieces 8760

TJ :
Label 2
sensor(NO) 301
pieces 8760
"""
"""
数据对应坐标
"""
def get_data2(path):
    lat = []
    lon = []
    tem = []
    for file in tqdm(os.listdir(path)):  ##进度条
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        lat.append(df.loc[0,'lat'])
        lon.append(df.loc[0,'lon'])
        tem.append(df.loc[:,'TEM'].astype(str))
    data_location = np.transpose(np.vstack((lat, lon)))
    dff = pd.concat([pd.DataFrame({'{}'.format(index): labels}) for index, labels in enumerate(tem)], axis=1)
    return dff.fillna(0).values.T,data_location
"""
只有数据，没有对应坐标
"""
def get_data(path):
    dataset_list = []
    for file in tqdm(os.listdir(path)):  ##进度条
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        full_filename = os.path.basename(file_path) # 带后缀
        filename = full_filename.split('.')[0] # 不带后缀
        df['NO'] = int(re.findall(r"\d+", str(filename))[0])
        if os.path.basename(path) == 'BJ':
            df['label'] = 1
        else:
            df['label'] = 2
        dataset_list.append(df)
    df = pd.concat(dataset_list)
    return df

"""
normalization
"""
def normalization(data,error,adjust):
    a = data[error] - adjust
    data[error] = a
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

