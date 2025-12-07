import argparse
import os
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
import ast
import holidays  # pip install holidays 필요
from models import LSTM, NBeats, TSMixer, PatchTST, iTransformer, TimeXer, TSMixer_Covar

# =============================================================================
# Argument Parser
# =============================================================================
parser = argparse.ArgumentParser('Real-World Future Prediction with Smart Holiday')

# 1. Basic Settings
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--file_path', type=str, default='dataset/airport_flight_with_holiday.csv')
parser.add_argument('--output_dir', type=str, default='predictions')

# 2. Target & Model Selection
parser.add_argument('--target', type=str, default='arrival', choices=['arrival', 'departure', 'both'])
parser.add_argument('--model', type=str, default='patchtst')
parser.add_argument('--use_covar', action='store_true') 

# 3. Model Params (Placeholders)
parser.add_argument('--seq_len', type=int, default=56)
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--pred_len', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--patch_len', type=int, default=16) 
parser.add_argument('--factor', type=int, default=5)
parser.add_argument('--norm_mode', type=str, default='revin')

args = parser.parse_args()

# =============================================================================
# Load Best Params
# =============================================================================
def load_best_model_params(args):
    target_map = {
        'arrival': 'arrival_models_comparison.csv',
        'departure': 'departure_models_comparison.csv',
        'both': 'both_models_comparison.csv'
    }
    csv_path = os.path.join('best_models', target_map.get(args.target))
    
    if not os.path.exists(csv_path):
        print(f"[Warning] Config file not found. Using args.")
        return args
    
    try:
        df = pd.read_csv(csv_path)
        row = df[df['Model'] == args.model]
        if row.empty: return args
        
        param_str = row.iloc[0]['Best_Params']
        def dummy_device(type='cuda', index=None): return type
        params_dict = eval(param_str, {'device': dummy_device})
        
        ignore = ['gpu', 'file_path', 'output_dir']
        print(f">> [Auto-Config] Loaded Params for '{args.model}'")
        for k, v in params_dict.items():
            if k not in ignore: setattr(args, k, v)
        if hasattr(args, 'patch_len'): args.stride = args.patch_len
    except: pass
    return args

args = load_best_model_params(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Smart Holiday Mark Generator
# =============================================================================
def get_future_holiday_marks(start_date, pred_len, device):
    """
    미래 날짜에 대한 공변량(Holiday Mark)을 생성합니다.
    규칙: 평일(0), 휴일/주말(1), 설날(2), 추석(3)
    로직: 샌드위치 데이(명절과 휴일 사이 평일)는 해당 명절로 통합
    """
    # 1. 미래 날짜 리스트 생성
    future_dates = [start_date + timedelta(days=i) for i in range(pred_len)]
    
    # 2. 공휴일 정보 로드 (필요한 연도만)
    years = list(set([d.year for d in future_dates]))
    kr_holidays = holidays.KR(years=years)
    
    raw_marks = []
    
    # 3. 기본 분류 (0, 1, 2, 3)
    for dt in future_dates:
        is_holiday = dt in kr_holidays
        holiday_name = kr_holidays.get(dt, "")
        
        val = 0
        # 우선순위: 설/추석 > 일반휴일/주말 > 평일
        if is_holiday and '설날' in holiday_name:
            val = 2
        elif is_holiday and '추석' in holiday_name:
            val = 3
        elif is_holiday or dt.weekday() >= 5: # 주말(5,6) 포함
            val = 1
        else:
            val = 0
        raw_marks.append(val)
        
    # 4. 샌드위치 데이 처리 (Bridging Logic)
    # 로직: 평일(0) 양옆에 [명절(2,3)]과 [비평일(1,2,3)]이 있으면 명절로 변경
    final_marks = raw_marks.copy()
    n = len(final_marks)
    
    # 샌드위치 처리는 윈도우 내부에서만 수행 (경계값 처리는 생략하거나 윈도우 확장 필요하나 여기선 내부만)
    for i in range(n):
        if final_marks[i] == 0:
            prev = final_marks[i-1] if i > 0 else 0
            next_val = final_marks[i+1] if i < n-1 else 0
            
            # Case 1: [명절] - [평일] - [휴일/명절] -> 평일을 앞 명절로 통합
            if (prev in [2, 3]) and (next_val != 0):
                final_marks[i] = prev
            # Case 2: [휴일/명절] - [평일] - [명절] -> 평일을 뒤 명절로 통합
            elif (next_val in [2, 3]) and (prev != 0):
                final_marks[i] = next_val
    
    # 5. Tensor 변환 [1, pred_len]
    return torch.tensor(final_marks, dtype=torch.float32).unsqueeze(0).to(device)

# =============================================================================
# Prepare Real-World Input Data
# =============================================================================
def get_latest_input(args):
    if not os.path.exists(args.file_path): raise FileNotFoundError(f"No file: {args.file_path}")
    df = pd.read_csv(args.file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    last_date = df['date'].iloc[-1]
    print(f">> Data Ends At: {last_date.strftime('%Y-%m-%d')}")
    
    input_df = df.iloc[-args.seq_len:].copy()
    
    if args.target == 'arrival': cols = ['arrival']
    elif args.target == 'departure': cols = ['departure']
    else: cols = ['arrival', 'departure']
    
    if args.use_covar: cols.append('holiday_type')
        
    data_values = input_df[cols].values
    input_tensor = torch.tensor(data_values, dtype=torch.float32).unsqueeze(0).to(device)
    
    return input_tensor, last_date

# =============================================================================
# Main Logic
# =============================================================================
def get_model_instance(args):
    if args.target == 'both': args.ndims = 2
    else: args.ndims = 1
    
    model_dict = {
        'lstm': LSTM, 'nbeats': NBeats, 'tsmixer': TSMixer, 
        'patchtst': PatchTST, 'itransformer': iTransformer, 
        'timexer': TimeXer, 'tsmixer_covar': TSMixer_Covar,
    }
    return model_dict[args.model].Model(args).to(device)


def main():
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f" REAL FUTURE PREDICTION | Target: {args.target} | Steps: {args.pred_len}")
    print(f"{'='*60}")

    # 1. 데이터 준비
    try:
        input_x, last_date = get_latest_input(args)
    except Exception as e:
        print(f"[Error] Data prep: {e}"); return

    # 2. 모델 로드
    try:
        model = get_model_instance(args)
        ckpt_path = f'best_models/{args.model}/1.pth'

        if not os.path.exists(ckpt_path):
            print(f"[Error] Checkpoint not found: {ckpt_path}"); return
            
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f">> Model Loaded: {ckpt_path}")
    except Exception as e:
        print(f"[Error] Model load: {e}"); return

    # 3. 예측
    with torch.no_grad():
        if args.target == 'both' or args.use_covar:
            # 입력 데이터 분리
            x_feat = input_x[:, :, :-1] # Feature
            x_mark = input_x[:, :, -1]  # Past Holiday
            
            # 미래 날짜에 대한 공휴일 마킹 생성 (샌드위치 로직 포함)
            future_start_date = last_date + timedelta(days=1)
            future_mark = get_future_holiday_marks(future_start_date, args.pred_len, device)
            
            # 모델 입력 (과거 특징, 과거 마크, 미래 마크)
            output = model(x_feat, x_mark, future_mark)
        else:
            output = model(input_x)

    # 4. 결과 저장
    preds = output.cpu().numpy().squeeze(0)
    
    future_dates = [last_date + timedelta(days=i+1) for i in range(args.pred_len)]
    results = []
    
    for i, date in enumerate(future_dates):
        d_str = date.strftime('%Y-%m-%d')
        if args.target == 'both':
            results.append({'Date': d_str, 'Type': 'Arrival', 'Value': round(preds[i, 0], 2)})
            results.append({'Date': d_str, 'Type': 'Departure', 'Value': round(preds[i, 1], 2)})
        else:
            val = preds[i, 0] if preds.ndim > 1 else preds[i]
            results.append({'Date': d_str, 'Type': args.target.capitalize(), 'Value': round(float(val), 2)})

    df_res = pd.DataFrame(results)
    
    # Pivot
    if args.target == 'both':
        df_final = df_res.pivot(index='Date', columns='Type', values='Value').reset_index()
    else:
        df_final = df_res[['Date', 'Value']]
        df_final.columns = ['Date', f'Predicted_{args.target}']
    
    save_name = f"{args.target}_{args.model}_{args.pred_len}_{last_date.strftime('%Y%m%d')}.csv"
    save_path = os.path.join(args.output_dir, save_name)
    df_final.to_csv(save_path, index=False)
    print(f"\n>> Saved: {save_path}")


if __name__ == '__main__':
    main()