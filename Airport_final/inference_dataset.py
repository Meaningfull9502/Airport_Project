import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class Dataset_Airport_Inference(Dataset):
    def __init__(self, data_df, benchmark_df, date_series, indices, size=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        start_idx, end_idx = indices
        
        # 1. Data Conversion
        if isinstance(data_df, pd.DataFrame):
            self.data = data_df.iloc[start_idx:end_idx].values
        else:
            self.data = data_df[start_idx:end_idx]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        # 2. Benchmark (Slot)
        self.benchmark = None
        if benchmark_df is not None:
            if isinstance(benchmark_df, pd.DataFrame):
                self.bench_data = benchmark_df.iloc[start_idx:end_idx].values
            else:
                self.bench_data = benchmark_df[start_idx:end_idx]
            self.bench_data = torch.tensor(self.bench_data, dtype=torch.float32)
            self.benchmark = True
            
        # 3. Date Info (전체 날짜 정보 저장)
        self.date_series = pd.to_datetime(date_series.iloc[start_idx:end_idx]).reset_index(drop=True)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        # [중요] 윈도우의 시작 날짜(Time Step 0)를 반환
        start_date_item = self.date_series.iloc[r_begin]
        start_date_str = start_date_item.strftime('%Y-%m-%d')

        if self.benchmark:
            seq_bench = self.bench_data[r_begin:r_end]
            # 4개의 값을 반환: 입력, 정답, 벤치마크, 시작날짜
            return seq_x, seq_y, seq_bench, start_date_str
            
        return seq_x, seq_y, start_date_str
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1


def _get_range_idx(df, start_year, end_year, seq_len=0):
    start_date = f'{start_year}-01-01'
    end_date = f'{end_year}-12-31'
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    indices = df.index[mask]
    if len(indices) > 0:
        return [max(0, indices[0] - seq_len), indices[-1] + 1]
    return [0, 0]


def load_inference_dataset(args):
    print(f">> [Dataset] Loading 2024~2025 Data with Dates...")

    if os.path.exists(args.file_path):
        df = pd.read_csv(args.file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.fillna(0)
    else:
        print(f"Error: csv file not found at {args.file_path}")
        return None

    # [이벤트 날짜 추출] holiday_type != 0 인 날짜들
    event_df = df[df['holiday_type'] != 0]
    event_dates = set(event_df['date'].dt.strftime('%Y-%m-%d').values)
    print(f"   - Identified {len(event_dates)} Event Days (Holiday != 0)")

    if args.target == 'arrival':
        target_cols = ['arrival']
        bench_cols = ['slot_arrival']
    elif args.target == 'departure':
        target_cols = ['departure']
        bench_cols = ['slot_departure']
    elif args.target == 'both':
        target_cols = ['arrival', 'departure']
        bench_cols = ['slot_arrival', 'slot_departure']
    else:
        target_cols = [args.target]
        bench_cols = []
    
    if args.target == 'both' or args.use_covar:
        target_cols.append('holiday_type')
        
    data_df = df[target_cols]
    bench_df = df[bench_cols]
    date_series = df['date'] # 날짜 컬럼 전달

    ndims = data_df.shape[-1]
    size_config = [args.seq_len, args.label_len, args.pred_len, ndims]

    test_idx = _get_range_idx(df, 2024, 2025, seq_len=args.seq_len)

    dataset = Dataset_Airport_Inference(data_df, bench_df, date_series, test_idx, size_config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    return {
        "test_loader": loader,
        "ndims": ndims,
        "event_dates": event_dates # 이벤트 날짜 집합 반환
    }