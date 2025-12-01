# import torch
# import os
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# import utils


# class Dataset_Airport_CV(Dataset):
#     def __init__(self, df, indices, size=None):
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         self.indices = indices
        
#         start_idx, end_idx = self.indices
        
#         if isinstance(df, pd.DataFrame):
#             self.data = df.iloc[start_idx:end_idx].values
#         else:
#             self.data = df[start_idx:end_idx]
            
#         self.data = torch.tensor(self.data, dtype=torch.float32)
            
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len 
#         r_end = r_begin + self.label_len + self.pred_len
        
#         seq_x = self.data[s_begin:s_end]
#         seq_y = self.data[r_begin:r_end]
#         return seq_x, seq_y

#     def __len__(self):
#         return len(self.data) - self.seq_len - self.pred_len + 1


# def load_dataset(args):
#     if os.path.exists(args.file_path):
#         df = pd.read_csv(args.file_path)
#         df['date'] = pd.to_datetime(df['date'])
        
#         df = df.sort_values('date')
        
#         if args.use_corona:
#             covid_start = pd.Timestamp('2020-01-01')
#             covid_end = pd.Timestamp('2023-12-31')
#             df = df[(df['date'] < covid_start) | (df['date'] > covid_end)]
#             df = df.reset_index(drop=True)
#             print("Expect Corona Period: 2020-01-01~2023-12-31")
#     else:
#         print(f"Error: csv file not found at {args.file_path}")
#         return None

#     # --- Train Index ---
#     train_end_date = f'{args.val_year - 1}-12-31'
    
#     # 데이터가 비어있지 않은지 확인
#     train_subset = df[df['date'] <= train_end_date]
#     if not train_subset.empty:
#         # Train은 0번부터 ~ (Validation 전년도 말)까지
#         train_end_idx = train_subset.index[-1] + 1
#         train_idx = [0, train_end_idx]
#     else:
#         # 혹시 전처리로 인해 Train 데이터가 아예 사라진 경우 예외처리
#         raise ValueError(f"No training data found before {args.val_year} after removing COVID period.")

#     # --- Validation Index ---
#     val_start_date = f'{args.val_year}-01-01'
#     val_end_date = f'{args.val_year}-12-31'
    
#     val_subset_start = df[df['date'] >= val_start_date]
#     val_subset_end = df[df['date'] <= val_end_date]
    
#     if not val_subset_start.empty and not val_subset_end.empty:
#         val_start_raw = val_subset_start.index[0]
#         # Val 끝 인덱스가 범위를 벗어나지 않도록 min 처리 혹은 로직 유지
#         # 보통 index[-1] + 1 은 슬라이싱 [start:end]를 위함
#         val_end_idx_candidate = df[df['date'] <= val_end_date].index
#         if len(val_end_idx_candidate) > 0:
#              val_end_raw = val_end_idx_candidate[-1] + 1
#         else:
#              val_end_raw = val_start_raw # 데이터 부족 시

#         val_idx = [val_start_raw - args.seq_len, val_end_raw]
#     else:
#         val_idx = [0, 0]

#     # --- Test Index ---
#     test_start_date = f'{args.test_year-1}-01-01'
#     test_end_date = f'{args.test_year}-12-31'
    
#     test_subset_start = df[df['date'] >= test_start_date]
    
#     if not test_subset_start.empty:
#         test_start_raw = test_subset_start.index[0]
#         test_end_idx_candidate = df[df['date'] <= test_end_date].index
        
#         if len(test_end_idx_candidate) > 0:
#             test_end_raw = test_end_idx_candidate[-1] + 1
#         else:
#             test_end_raw = len(df)

#         test_idx = [test_start_raw - args.seq_len, test_end_raw]
#     else:
#         test_idx = [0, 0]

#     # Dataset 생성
#     size_config = [args.seq_len, args.label_len, args.pred_len]
    
#     if args.target == 'both':
#         df = df[['arrival', 'departure']]
#     else:
#         df = df[[args.target]]
#     ndims = df.shape[-1]
    
#     train_set = Dataset_Airport_CV(df, train_idx, size_config)
#     val_set = Dataset_Airport_CV(df, val_idx, size_config)
#     test_set = Dataset_Airport_CV(df, test_idx, size_config)
    
#     train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
#     # 결과 반환
#     data_obj = {
#         "train_set": utils.inf_generator(train_dataloader), 
#         "val_set": utils.inf_generator(val_dataloader), 
#         "test_set": utils.inf_generator(test_dataloader),
#         "test_loader": test_dataloader,
#         "n_train_batches": len(train_dataloader),
#         "n_val_batches": len(val_dataloader),
#         "n_test_batches": len(test_dataloader),
#         "ndims": int(ndims),
#     }
    
#     return data_obj



import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import utils

class Dataset_Airport_CV(Dataset):
    def __init__(self, df, indices, size=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        start_idx, end_idx = indices
        
        # DataFrame 슬라이싱
        if isinstance(df, pd.DataFrame):
            self.data = df.iloc[start_idx:end_idx].values
        else:
            self.data = df[start_idx:end_idx]
            
        self.data = torch.tensor(self.data, dtype=torch.float32)
            
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1


def _get_range_idx(df, target_year, seq_len=0, end_year=None):
    """특정 연도(범위)에 해당하는 데이터 인덱스를 반환하는 헬퍼 함수"""
    start_date = f'{target_year}-01-01'
    if end_year is None:
        end_date = f'{target_year}-12-31'
    else:
        end_date = f'{end_year}-12-31'
        
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    indices = df.index[mask]
    
    if len(indices) > 0:
        # seq_len을 빼주어 앞부분 Context(Look-back window) 확보
        return [max(0, indices[0] - seq_len), indices[-1] + 1]
    return [0, 0]


def load_dataset(args, is_refit=False):
    # 1. Load CSV & Handle Gap (Corona)
    if os.path.exists(args.file_path):
        df = pd.read_csv(args.file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # [Gap Strategy] 2020~2022 삭제 (학습 방해), 2023 유지 (2024 예측용 Context)
        if not args.use_corona:
            covid_start = pd.Timestamp('2020-01-01')
            covid_end = pd.Timestamp('2022-12-31')
            df = df[(df['date'] < covid_start) | (df['date'] > covid_end)]
            df = df.reset_index(drop=True)
    else:
        print(f"Error: csv file not found at {args.file_path}")
        return None

    if args.target == 'both':
        data_df = df[['arrival', 'departure']]
    else:
        data_df = df[[args.target]]
    
    ndims = data_df.shape[-1]
    size_config = [args.seq_len, args.label_len, args.pred_len]

    # =========================================================
    # [3-Split Logic] Train / Valid (EarlyStop) / Eval (Score)
    # =========================================================
    # 변수 초기화
    train_idx, valid_idx, eval_idx = [0,0], [0,0], [0,0]
    
    if is_refit:
        # --- REFIT MODE ---
        # Eval (Final Test) : 2024 ~ 2025
        # Valid (Early Stop): 2019 (args.cv_end_year)
        # Train             : 2002 ~ 2018
        
        # 1. Eval (Test)
        eval_idx = _get_range_idx(df, args.test_year, args.seq_len, end_year=args.test_year+1)
        
        # 2. Valid (2019)
        refit_val_year = args.cv_end_year
        valid_idx = _get_range_idx(df, refit_val_year, args.seq_len)
        
        # 3. Train (~2018)
        train_end_year = refit_val_year - 1
        train_end_date = f'{train_end_year}-12-31'
        train_mask = df['date'] <= train_end_date
        
        if len(df.index[train_mask]) > 0:
            train_idx = [0, df.index[train_mask][-1] + 1]
        else:
            raise ValueError("Refit train data empty")

    else:
        # --- CV MODE (Nested) ---
        # Eval (Outer Valid) : args.val_year (e.g., 2015) -> Score
        # Valid (Inner Valid): args.val_year-1 (e.g., 2014) -> Early Stop
        # Train              : ~ args.val_year-2 (e.g., ~2013)
        
        # 1. Eval (Outer)
        eval_idx = _get_range_idx(df, args.val_year, args.seq_len)
        
        # 2. Valid (Inner)
        valid_idx = _get_range_idx(df, args.val_year - 1, args.seq_len)
        
        # 3. Train
        train_end_year = args.val_year - 2
        train_end_date = f'{train_end_year}-12-31'
        train_mask = df['date'] <= train_end_date
        
        if len(df.index[train_mask]) > 0:
            train_idx = [0, df.index[train_mask][-1] + 1]
        else:
            raise ValueError(f"CV Train data empty before {train_end_year}")

    # Dataset Creation
    train_set = Dataset_Airport_CV(data_df, train_idx, size_config)
    valid_set = Dataset_Airport_CV(data_df, valid_idx, size_config) if valid_idx != [0,0] else None
    eval_set = Dataset_Airport_CV(data_df, eval_idx, size_config) # Outer Valid or Final Test
    
    # Loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    valid_loader = None
    if valid_set:
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
        
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)
    
    return {
        "train_set": utils.inf_generator(train_loader),
        "valid_set": utils.inf_generator(valid_loader) if valid_loader else None, # Early Stop용
        "eval_set": utils.inf_generator(eval_loader), # 평가용
        "n_train_batches": len(train_loader),
        "n_valid_batches": len(valid_loader) if valid_loader else 0,
        "n_eval_batches": len(eval_loader),
        "ndims": ndims
    }