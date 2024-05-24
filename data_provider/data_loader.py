import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from tqdm import tqdm
import json
warnings.filterwarnings('ignore')

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.data_path.endswith('.csv'):
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                            self.data_path))
        elif self.data_path.endswith('.parquet'):
            df_raw = pd.read_parquet(os.path.join(self.root_path,
                                            self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp = df_raw[['Date Time']][border1:border2]
        df_stamp['Date Time'] = pd.to_datetime(df_raw['Date Time'], format='%d.%m.%Y %H:%M:%S')
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['Date Time'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['Date Time'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['Date Time'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['Date Time'].apply(lambda row: row.hour, 1)
            # data_stamp = df_stamp.drop(['date'], axis=1).values
            data_stamp = df_stamp.drop(['Date Time'], axis=1).values
        elif self.timeenc == 1:
            # data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = time_features(pd.to_datetime(df_stamp['Date Time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_TGTSF(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, 
                 info_overhead=0, news_path=None, des_path=None,
                 news_pre_embed=True, des_pre_embed=False, add_date=True,
                 text_encoder=None, text_dim=384):
        # size [seq_len, label_len, pred_len]
        # info

        assert len(size) == 3

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.news_path = news_path
        self.des_path = des_path

        self.info_overhead = info_overhead
        self.news_pre_embed = news_pre_embed
        self.des_pre_embed = des_pre_embed
        self.add_date=add_date

        self.text_encoder = text_encoder
        self.text_dim = text_dim

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            print(df_raw.columns)
            df_data = df_raw[cols_data]
            self.cols_name=cols_data
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.cols_name=[self.target]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data = data[border1:border2]
        
        self.dates = df_raw['date'][border1+self.info_overhead:border2+self.info_overhead].values
        # print(self.des_pre_embed)
        if self.des_pre_embed:
            print('now using pre embedded description')
            if self.add_date:
                file_extension = '.npy'
                file_list = [f for f in os.listdir(os.path.join(self.root_path, self.des_path)) if f.endswith(file_extension)]
                self.des_data_dict = {}
                for d in tqdm(self.dates):
                    file_ = os.path.join(self.root_path, self.des_path, f"des-{d}.npy")
                    if os.path.exists(file_):
                        des = np.load(file_, allow_pickle=True)
                        des = torch.tensor(des.reshape(1,des.shape[0]))
                    else:
                        des = torch.zeros((1, self.text_dim))
                    self.des_data_dict[d] = des
            else:
                self.channel_description = torch.tensor(np.load(os.path.join(self.root_path, self.des_path+'.npy'), allow_pickle=True))
                self.channel_description = self.channel_description.unsqueeze(0)
        else:
            if self.des_path == 'None':
                self.channel_description = self.cols_name
                print('now using title as description')
                # print(self.channel_description)
            else:
                with open(os.path.join(self.root_path, self.des_path)) as f:
                    self.channel_description = f.readlines()
                print('now using '+self.des_path+' as description')
                # print(self.channel_description)
        if self.news_path != 'None':
            file_extension = '.npy'
            file_list = [f for f in os.listdir(os.path.join(self.root_path, self.news_path)) if f.endswith(file_extension)]
            self.news_data_dict = {}
            for d in tqdm(self.dates):
                file_ = os.path.join(self.root_path, self.news_path, f"News-{d}.npy")
                if os.path.exists(file_):
                    news = np.load(file_, allow_pickle=True)
                    news = torch.tensor(news.reshape(1,news.shape[0],news.shape[1]))
                else:
                    news = torch.zeros((1, 1, self.text_dim))
                self.news_data_dict[d] = news
            
            

    def __getitem__(self, index):
        lbw_begin = index
        lbw_end = lbw_begin + self.seq_len
        h_begin = lbw_end #- self.label_len 
        h_end = lbw_end + self.pred_len

        seq_x = self.data[lbw_begin:lbw_end]
        seq_y = self.data[h_begin:h_end]
        dates = self.dates[h_begin:h_end]


        if self.news_path != 'None':
            newss=[]
            dess=[]
            newss = [self.news_data_dict[d] for d in dates]
            max_news_number = max([_.shape[-2] for _ in newss])
            if self.des_pre_embed:
                if self.add_date:
                    dess = [self.des_data_dict[d] for d in dates]
                else:
                    dess = [self.channel_description for _ in dates]
            else:
                if self.add_date:
                    dess = [self._get_embedding([d+self.channel_description[i] for i in range(len(self.channel_description))]) for d in dates]
                else:
                    tmp_ = self._get_embedding(self.channel_description)
                    dess = [tmp_ for d in dates]

            for i in range(len(newss)):
                newss[i] = torch.nn.functional.pad(newss[i], (0, 0, 0, max_news_number - newss[i].shape[1])) # [l, n, d]
            news=torch.cat(newss, dim=0)
            des=torch.stack(dess, dim=0)
        else:
            news=torch.zeros((self.seq_len,1,self.text_dim))
            des=torch.zeros((self.seq_len,len(self.cols_name),self.text_dim))

        return seq_x, seq_y, news, des
    
    def _get_embedding(self, text):
        a =self.text_encoder.encode(text, convert_to_tensor=True)
        return a


    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


class Dataset_TGTSF_elec(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, 
                 info_overhead=0, news_path=None, des_path=None,
                 news_pre_embed=True, des_pre_embed=False, add_date=True,
                 text_encoder=None, text_dim=384):
        # size [seq_len, label_len, pred_len]
        # info

        assert len(size) == 3

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.news_path = news_path
        self.des_path = des_path

        self.info_overhead = info_overhead
        self.news_pre_embed = news_pre_embed
        self.des_pre_embed = des_pre_embed
        self.add_date=add_date

        self.text_encoder = text_encoder
        self.text_dim = text_dim

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:-2]
            # print(df_raw.columns)
            df_data = df_raw[cols_data]
            self.cols_name=cols_data
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.cols_name=[self.target]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data = data[border1:border2]
        # self.extra = df_raw[['dayofweek', 'holiday']][border1+self.info_overhead:border2+self.info_overhead].values
        
        self.dates = df_raw['dayofweek'][border1+self.info_overhead:border2+self.info_overhead].values
        self.holiday = df_raw['holiday'][border1+self.info_overhead:border2+self.info_overhead].values
        # print(self.des_pre_embed)

        self.news_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6, 'Holiday': 7, 'Weekday': 8, 'Weekend': 9}
        # file_list = [f for f in os.listdir(os.path.join(self.root_path, self.newspath)) if f.endswith('.npy')]
        desc_list = []
        for dayname in self.news_dict.keys():
            self.news_dict[dayname] = torch.tensor(np.load(os.path.join(self.root_path, self.news_path, f"News-{dayname}.npy"))).unsqueeze(0)
        for i in range(321):
            desc_list.append(torch.tensor(np.load(os.path.join(self.root_path, self.des_path, f"Channel-{i}.npy"))))
        self.desc_list = torch.stack(desc_list, dim=0)
        self.desc_list = self.desc_list.unsqueeze(0).repeat(self.pred_len, 1, 1)
            

    def __getitem__(self, index):
        lbw_begin = index
        lbw_end = lbw_begin + self.seq_len
        h_begin = lbw_end - self.label_len # 
        h_end = lbw_end + self.pred_len

        seq_x = self.data[lbw_begin:lbw_end]
        seq_y = self.data[h_begin:h_end]
        # dayofweeks = self.dates[lbw_begin:lbw_end]
        # holidays = self.holiday[lbw_begin:lbw_end]
        dayofweeks = self.dates[lbw_end:h_end]
        holidays = self.holiday[lbw_end:h_end]

        newss=[]

        for i in range(self.pred_len):
            news = [self.news_dict[dayofweeks[i]]]
            if holidays[i] == 1:
                news.append(self.news_dict['Holiday'])
            elif dayofweeks[i] in ['Saturday', 'Sunday']:
                news.append(self.news_dict['Weekend'])
            else:
                news.append(self.news_dict['Weekday'])
            news = torch.cat(news, dim=0)
            newss.append(news)
        news = torch.stack(newss, dim=0)

        des = self.desc_list
        
        # print(seq_x.shape, seq_y.shape, news.shape, des.shape)

        return seq_x, seq_y, news, des
    
    def _get_embedding(self, text):
        a =self.text_encoder.encode(text, convert_to_tensor=True)
        return a


    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_TGTSF_weather(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, 
                 info_overhead=0, news_path=None, des_path=None,
                 news_pre_embed=True, des_pre_embed=False, add_date=True,
                 text_encoder=None, text_dim=384, stride=24, control=False):
        # size [seq_len, label_len, pred_len]
        # info

        assert len(size) == 3

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.news_path = news_path
        self.des_path = des_path

        self.info_overhead = info_overhead
        self.news_pre_embed = news_pre_embed
        self.des_pre_embed = des_pre_embed
        self.add_date=add_date

        self.text_encoder = text_encoder
        self.text_dim = text_dim

        self.stride = stride

        self.control=control

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_parquet(os.path.join(self.root_path,
                                          self.data_path))

        df_raw['Date Time'] = pd.to_datetime(df_raw['Date Time'], format='%d.%m.%Y %H:%M:%S')
        df_raw['Date Time'] = df_raw['Date Time'].dt.strftime('%Y%m%d%H%M').astype(int)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        self.cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            # print(df_raw.columns)
            df_data = df_raw[cols_data]
            self.cols_name=cols_data
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.cols_name=[self.target]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data = data[border1:border2]

        self.time_stamp = df_raw['Date Time'][border1+self.info_overhead:border2+self.info_overhead].values
        if 'large' in self.news_path:
            with open(os.path.join(self.root_path, 'wm_messages_large_v1.json')) as f:
                self.time2msg1 = json.load(f)
            with open(os.path.join(self.root_path, 'wm_messages_large_v2.json')) as f:
                self.time2msg2 = json.load(f)
            with open(os.path.join(self.root_path, 'wm_messages_large_v3.json')) as f: 
                self.time2msg3 = json.load(f)

            with open(os.path.join(self.root_path, 'hashtable_large.json')) as f:
                self.msg2emb = json.load(f)

        else:
            with open(os.path.join(self.root_path, 'wm_messages_v1.json')) as f:
                self.time2msg1 = json.load(f)
            with open(os.path.join(self.root_path, 'wm_messages_v2.json')) as f:
                self.time2msg2 = json.load(f)
            with open(os.path.join(self.root_path, 'wm_messages_v3.json')) as f: 
                self.time2msg3 = json.load(f)

            with open(os.path.join(self.root_path, 'hashtable.json')) as f:
                self.msg2emb = json.load(f)
        
        self.time2msg = [self.time2msg1, self.time2msg2, self.time2msg3]
        msg2emb_={}

        for i in self.msg2emb.keys():
            _ = np.load(os.path.join(self.root_path, self.news_path, f"{self.msg2emb[i]}.npy"))
            _ = _[:self.text_dim]
            _ = normalize_l2(_)
            msg2emb_[self.msg2emb[i]] = torch.tensor(_) # {hash: emb}
        self.msg2emb=msg2emb_
        
        desc_list = []

        for i in range(len(cols_data)):
            _ = np.load(os.path.join(self.root_path, self.des_path, f"channel_{i}.npy"))
            _ = _[:self.text_dim]
            _ = normalize_l2(_)
            desc_list.append(torch.tensor(_))

        self.desc_list = torch.stack(desc_list, dim=0)
        self.desc_list = self.desc_list.unsqueeze(0).repeat(self.pred_len, 1, 1)

    def __getitem__(self, index):
        lbw_begin = index
        lbw_end = lbw_begin + self.seq_len
        h_begin = lbw_end
        h_end = lbw_end + self.pred_len

        seq_x = self.data[lbw_begin:lbw_end]
        seq_y = self.data[h_begin:h_end]

        time_stamps = self.time_stamp[h_begin:h_end:self.stride]

        newss=[]

        for i in time_stamps:
            news=[]
            date = str(i)[:8]
            time = int(str(i)[8:])
            if time<=559:
                time_stamp=date+'0000'
            elif time<=1159:
                time_stamp=date+'0600'
            elif time<=1759:
                time_stamp=date+'1200'
            else:
                time_stamp=date+'1800'

            # random pick 7 numbers from 0,1,2
            choices=np.random.choice(3, 7)
            for i in range(7):
                try:
                    chosen_ver = self.time2msg[choices[i]] # choose the msg version table
                    chosen_ver = chosen_ver[time_stamp] # choose the msg according to the time
                    chosen_hash = chosen_ver[i] # choose i-th sentence hash
                    news.append(self.msg2emb[chosen_hash])
                except (IndexError):
                    pass

            news = torch.stack(news, dim=0)

            newss.append(news)
        
        for i in range(len(newss)):
            newss[i] = torch.nn.functional.pad(newss[i], (0, 0, 0, 7 - newss[i].shape[0])) # [l, n, d]

        news=torch.stack(newss, dim=0)
        des=self.desc_list[::self.stride]
        if self.control:
            return seq_x, seq_y, news, des, time_stamps
        else:
            return seq_x, seq_y, news, des
    
    def _get_embedding(self, text):
        a =self.text_encoder.encode(text, convert_to_tensor=True)
        return a


    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)