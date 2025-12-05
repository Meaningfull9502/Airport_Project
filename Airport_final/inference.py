import argparse
import os
import csv
import torch
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from vis_utils import visualize_holiday_performance
from inference_dataset import load_inference_dataset
from models import LSTM, NBeats, TSMixer, PatchTST, iTransformer, TimeXer, TSMixer_Covar
# =============================================================================
# Helper: Load Best Params from CSV
# =============================================================================
def load_best_model_params(args):
    target_map = {
        'arrival': 'arrival_models_comparison.csv',
        'departure': 'departure_models_comparison.csv',
        'both': 'both_models_comparison.csv'
    }
    
    csv_filename = target_map.get(args.target)
    if args.use_dbloss:
        csv_path = os.path.join(f'logs/{args.target}_{args.pred_len}_{args.use_covar}_{args.use_dbloss}', csv_filename)
    else:
        csv_path = os.path.join(f'logs/{args.target}_{args.pred_len}_{args.use_covar}', csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"[Warning] Model list file not found: {csv_path}. Using default args.")
        return args
    
    try:
        df = pd.read_csv(csv_path)
        row = df[df['Model'] == args.model]
        if row.empty:
            print(f"[Warning] Model '{args.model}' not found in CSV. Using default args.")
            return args
        
        param_str = row.iloc[0]['Best_Params']
        def dummy_device(type='cuda', index=None): return type
        params_dict = eval(param_str, {'device': dummy_device})
        
        ignore_keys = ['gpu', 'device', 'batch_size', 'file_path', 'output_dir', 'checkpoint_path', 'test_year', 'cv_start_year', 'cv_end_year']
        
        print(f">> [Auto-Config] Loading params for '{args.model}' from {csv_filename}...")
        for k, v in params_dict.items():
            if k not in ignore_keys:
                setattr(args, k, v)
        
        if hasattr(args, 'patch_len'): args.stride = args.patch_len
        print(">> Parameters updated.\n")
        
    except Exception as e:
        print(f"[Error] Param load failed: {e}")
        
    return args


def get_model(args):
    model_dict = {
        'lstm': LSTM, 'nbeats': NBeats, 'tsmixer': TSMixer, 
        'patchtst': PatchTST, 'itransformer': iTransformer, 
        'timexer': TimeXer, 'tsmixer_covar': TSMixer_Covar,
    }
    
    return model_dict[args.model].Model(args).to(args.device)
# =============================================================================
# Argument Parser
# =============================================================================
parser = argparse.ArgumentParser('Airport Inference: Combined Report')

# 1. Environment
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--file_path', type=str, default='dataset/airport_flight_with_holiday.csv')
parser.add_argument('--output_dir', type=str, default='inference_results')

# 2. Target & Model
parser.add_argument('--target', type=str, default='arrival', choices=['arrival', 'departure', 'both'])
parser.add_argument('--model', type=str, default='patchtst')
parser.add_argument('--use_covar', action='store_true')
parser.add_argument('--use_dbloss', action='store_true')

# 3. Model Params (Default placeholders)
parser.add_argument('--seq_len', type=int, default=56)
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--pred_len', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--patch_len', type=int, default=16) 
parser.add_argument('--factor', type=int, default=5)
parser.add_argument('--norm_mode', type=str, default='revin')

args = parser.parse_args()

# Candidate Models
if args.target == 'both':
    candidate_models = ['timexer', 'tsmixer_covar', 'lstm', 'itransformer']
else:
    if args.use_covar:
        candidate_models = ['lstm', 'patchtst', 'tsmixer_covar']
    else:
        candidate_models = ['lstm', 'patchtst', 'nbeats', 'tsmixer']
    
# Load Params & Set Device
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================================================================
# Main Logic
# =============================================================================
def main():
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f" INFERENCE | Target: {args.target} | Model: {args.model}")
    print(f" Logic: Single CSV Report (Scope column separated)")
    print(f"{'='*60}")

    # 1. Dataset Load
    data_obj = load_inference_dataset(args)
    if data_obj is None: return
    
    raw_ndims = data_obj['ndims']
    if args.use_covar: args.ndims = raw_ndims - 1
    else: args.ndims = raw_ndims
    
    test_loader = data_obj['test_loader']
    event_dates_set = data_obj['event_dates'] 

    # 2. Model Load
    try:
        model = get_model(args)
        if args.use_dbloss:
            args.checkpoint_path = f'model_states/{args.target}_{args.pred_len}_{args.use_covar}_{args.use_dbloss}/{args.model}/1.pth'
        else:
            args.checkpoint_path = f'model_states/{args.target}_{args.pred_len}_{args.use_covar}/{args.model}/1.pth'
        
        if not os.path.exists(args.checkpoint_path):
             print(f"[Error] Checkpoint not found: {args.checkpoint_path}")
             return

        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f">> Model Loaded: {args.checkpoint_path}")
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        return

    # 3. Data Collection (Flattening)
    all_records = []
    print(">> Processing Inference...")
    
    with torch.no_grad():
        for batch_x, batch_y, batch_bench, start_dates in tqdm(test_loader):
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            
            # Inference
            if args.target == 'both' or args.use_covar:
                output = model(batch_x[:,:,:-1], batch_x[:,:,-1], batch_y[:,:,-1])
                true_y = batch_y[:,:,:-1]
            else:
                output = model(batch_x)
                true_y = batch_y
            
            p_np = output.cpu().numpy()
            t_np = true_y.cpu().numpy()
            s_np = batch_bench.cpu().numpy()

            batch_size = p_np.shape[0]
            for b in range(batch_size):
                start_dt = pd.Timestamp(start_dates[b])
                
                for step in range(args.pred_len):
                    target_date = start_dt + pd.Timedelta(days=step)
                    date_str = target_date.strftime('%Y-%m-%d')
                    
                    if args.target == 'both':
                        all_records.append({'Date': date_str, 'Target_Type': 'Arrival', 'My_Pred': p_np[b, step, 0], 'True': t_np[b, step, 0], 'Slot': s_np[b, step, 0]})
                        all_records.append({'Date': date_str, 'Target_Type': 'Departure', 'My_Pred': p_np[b, step, 1], 'True': t_np[b, step, 1], 'Slot': s_np[b, step, 1]})
                    else:
                        all_records.append({'Date': date_str, 'Target_Type': args.target, 'My_Pred': p_np[b, step, 0], 'True': t_np[b, step, 0], 'Slot': s_np[b, step, 0]})

    # 4. DataFrame Creation
    df_all = pd.DataFrame(all_records)
    
    # 5. Data Split
    df_model_eval = df_all.copy()
    df_comparison = df_all[df_all['Slot'] != 0].copy()
    
    print(f">> Total Predictions (Full Range): {len(df_model_eval)}")
    print(f">> Comparison Valid Predictions (Slot!=0): {len(df_comparison)}")

    # 6. Metric Calculation Function
    def calc_metrics_df(df, scope_name, mode='comparison'):
        results = []
        unique_targets = df['Target_Type'].unique()
        if len(unique_targets) == 0: return []
        
        for tgt in unique_targets:
            sub = df[df['Target_Type'] == tgt]
            if len(sub) == 0: continue
            
            # Comparison 모드라면 Slot이 없는 데이터는 평가에서 제외
            if mode == 'comparison':
                sub = sub[sub['Slot'] != 0]
                if len(sub) == 0: continue # 비교할 데이터가 없으면 스킵

            # Metric 계산
            rmse_my = np.sqrt(((sub['My_Pred'] - sub['True'])**2).mean())
            mae_my = (sub['My_Pred'] - sub['True']).abs().mean()
            
            if mode == 'comparison':
                rmse_slot = np.sqrt(((sub['Slot'] - sub['True'])**2).mean())
                mae_slot = (sub['Slot'] - sub['True']).abs().mean()
                diff_rmse = rmse_slot - rmse_my
                diff_mae = mae_slot - mae_my
            else:
                rmse_slot = np.nan; mae_slot = np.nan; diff_rmse = np.nan; diff_mae = np.nan
            
            results.append({
                'Target_Type': tgt, 'Scope': scope_name,
                'RMSE_My': rmse_my, 'RMSE_Slot': rmse_slot,
                'MAE_My': mae_my, 'MAE_Slot': mae_slot,
                'Diff_RMSE': diff_rmse, 'Diff_MAE': diff_mae,
                'Count': len(sub)
            })
            
        return results

    # 7. 리포트 생성 및 저장 함수 (콘솔 출력은 분리, 저장은 통합)
    def generate_and_save(df_source, suffix, mode='comparison'):
        
        # (1) 전체 기간 계산
        res_all = calc_metrics_df(df_source, "All_Period", mode=mode)
        
        # (2) 이벤트 기간 계산
        df_evt = df_source[df_source['Date'].isin(event_dates_set)]
        res_evt = calc_metrics_df(df_evt, "Event_Dates", mode=mode)
        
        # 결과 합치기
        final_res = res_all + res_evt
        if not final_res: return

        # ---- [콘솔 출력: 분리해서 보여줌] ----
        def print_table(res_list, title):
            print(f"\n[{suffix} - {title} Summary ({mode})]")
            header = f"{'Target':<10} | {'RMSE_My':<10} | {'RMSE_Slot':<10} | {'Diff_RMSE':<10} | {'MAE_My':<10} | {'MAE_Slot':<10} | {'Diff_MAE':<10}"
            print(header)
            print("-" * 90)
            for r in res_list:
                s_rmse = f"{r['RMSE_Slot']:.2f}" if not np.isnan(r['RMSE_Slot']) else "-"
                d_rmse = f"{r['Diff_RMSE']:.2f}" if not np.isnan(r['Diff_RMSE']) else "-"
                s_mae = f"{r['MAE_Slot']:.2f}" if not np.isnan(r['MAE_Slot']) else "-"
                d_mae = f"{r['Diff_MAE']:.2f}" if not np.isnan(r['Diff_MAE']) else "-"
                print(f"{r['Target_Type']:<10} | {r['RMSE_My']:<10.2f} | {s_rmse:<10} | {d_rmse:<10} | {r['MAE_My']:<10.2f} | {s_mae:<10} | {d_mae:<10} ")

        print_table(res_all, "All Period")
        print_table(res_evt, "Event Dates")

        # ---- [CSV 저장: 하나의 파일에 통합 저장] ----
        df_res = pd.DataFrame(final_res)
        df_res['Model'] = args.model
        df_res['Evaluation_Mode'] = mode
        df_res['Use_Covar'] = args.use_covar
        df_res['Use_DBloss'] = args.use_dbloss
        if 'Checkpoint' not in df_res.columns: df_res['Checkpoint'] = os.path.basename(args.checkpoint_path)
        
        # 컬럼 순서 정리
        cols = ['Model', 'Evaluation_Mode', 'Target_Type', 'Scope', 'Use_Covar', 'Use_DBloss', 'RMSE_My', 'RMSE_Slot', 'MAE_My', 'MAE_Slot', 'Diff_RMSE', 'Diff_MAE', 'Checkpoint']
        df_res = df_res[cols]
        
        save_name = f"{suffix}_{args.target}_{args.pred_len}.csv"
        save_path = os.path.join(args.output_dir, save_name)
        
        # 파일이 있으면 이어서 쓰기 (Append), 없으면 새로 쓰기
        if os.path.exists(save_path):
            df_res.to_csv(save_path, mode='a', header=False, index=False)
        else:
            df_res.to_csv(save_path, index=False)
            
        print(f"\n>> All results saved to: {save_path}")

    # 실행 1: Model Full Evaluation (전체)
    generate_and_save(df_model_eval, "total", mode="model_only")
    # 실행 2: Comparison (Slot 데이터가 있는 날짜까지만)
    generate_and_save(df_comparison, "total", mode="comparison")
    # 시각화
    visualize_holiday_performance(model, test_loader, args, save_dir="holiday_plots")


if __name__ == '__main__':
    for model in candidate_models:
        args.model = model
        args = load_best_model_params(args)
        main()