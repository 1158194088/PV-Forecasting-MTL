import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler # ★ 引入 MinMaxScaler
from torch.utils.data import Dataset

class PVQuadTaskDataset(Dataset):
    def __init__(self, mtl_file, nwp_file, daily_file, seq_len=48, pred_len=24, train_ratio=0.7):
        self.seq_len = seq_len
        self.pred_len = pred_len

        df_mtl = pd.read_csv(mtl_file)
        df_mtl['Time'] = pd.to_datetime(df_mtl['Time'])
        df_nwp = pd.read_csv(nwp_file)
        df_nwp['time'] = pd.to_datetime(df_nwp['time'])
        
        df = pd.merge(df_mtl, df_nwp, left_on='Time', right_on='time', how='inner').fillna(0)
        df.drop(columns=['time'], inplace=True) 
        df = df.sort_values('Time').reset_index(drop=True)
        self.time_dt = df['Time']
        
        df_daily = pd.read_csv(daily_file)
        df_daily['date'] = pd.to_datetime(df_daily['date']).dt.strftime('%Y-%m-%d')
        unique_weathers = sorted(df_daily['weather'].dropna().unique())
        self.weather_mapping = {wt: idx for idx, wt in enumerate(unique_weathers)}
        self.weather_names_mapping = {idx: wt for wt, idx in self.weather_mapping.items()}
        self.num_classes = len(self.weather_mapping)
        self.daily_dict = dict(zip(df_daily['date'], df_daily['weather'].map(self.weather_mapping)))
        
        self.ramp_labels = torch.from_numpy(df['ramp_label_code'].values).long()

        dt = self.time_dt.dt
        self.time_feats = torch.from_numpy(np.stack([
            np.sin(2 * np.pi * dt.hour / 24), np.cos(2 * np.pi * dt.hour / 24),
            np.sin(2 * np.pi * dt.dayofyear / 365), np.cos(2 * np.pi * dt.dayofyear / 365),
            np.sin(2 * np.pi * dt.month / 12), np.cos(2 * np.pi * dt.month / 12)
        ], axis=1)).float()

        # =========================================================
        # =========================================================
        train_end_idx = int(len(df) * train_ratio)
        
        self.scaler_p = MinMaxScaler()
        self.scaler_p.fit(df[['Power']].values[:train_end_idx])
        self.power_values = torch.from_numpy(self.scaler_p.transform(df[['Power']].values)).float()
        
        self.scaler_dfi = MinMaxScaler()
        self.scaler_dfi.fit(df[['dfi']].values[:train_end_idx])
        self.dfi_values = torch.from_numpy(self.scaler_dfi.transform(df[['dfi']].values)).float()
        
        exclude_cols = ['Time', 'Power', 'dfi', 'ramp_rate', 'ramp_labels', 'ramp_label_code']
        feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
        
        self.scaler_nwp = StandardScaler()
        self.scaler_nwp.fit(df[feature_cols].values[:train_end_idx])
        self.nwp_data = torch.from_numpy(self.scaler_nwp.transform(df[feature_cols].values)).float()
        self.nwp_dim = self.nwp_data.shape[1]
        # =========================================================

        midnight_indices = df.index[df['Time'].dt.hour == 0].tolist()
        self.valid_x_begins = [pb - self.seq_len for pb in midnight_indices if (pb - self.seq_len) >= 0 and pb + self.pred_len <= len(df)]

    def __len__(self): return len(self.valid_x_begins)

    def __getitem__(self, idx):
        x_begin = self.valid_x_begins[idx]
        pred_begin = x_begin + self.seq_len
        ts_input = torch.cat([self.power_values[x_begin:pred_begin], self.time_feats[x_begin:pred_begin]], dim=1)
        nwp_input = torch.cat([self.nwp_data[pred_begin:pred_begin+self.pred_len], self.time_feats[pred_begin:pred_begin+self.pred_len]], dim=1)
        target_p = self.power_values[pred_begin:pred_begin + self.pred_len].squeeze(-1)
        target_dfi = self.dfi_values[pred_begin:pred_begin + self.pred_len].squeeze(-1)
        target_ramp = self.ramp_labels[pred_begin:pred_begin + self.pred_len]
        day_str = self.time_dt.iloc[pred_begin].strftime('%Y-%m-%d')
        daily_weather = self.daily_dict.get(day_str, 0) 
        
        return {
            'ts_input': ts_input, 'nwp_input': nwp_input, 
            'target_p': target_p, 'target_dfi': target_dfi, 
            'target_ramp': target_ramp, 'daily_weather': torch.tensor(daily_weather, dtype=torch.long)
        }