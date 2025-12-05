import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =============================================================================
# 1. 시각화 대상 명절 기간 정의
# =============================================================================
HOLIDAY_PERIODS = {
    "2024_Seollal": ["2024-02-09", "2024-02-12"],  
    "2024_Chuseok": ["2024-09-14", "2024-09-18"],  
    "2025_Seollal": ["2025-01-25", "2025-02-02"],  
    "2025_Chuseok": ["2025-10-03", "2025-10-12"],
}

def check_if_in_period(date_str):
    """주어진 날짜가 명절 기간에 포함되는지 확인하고, 해당 명절 이름 반환"""
    # 날짜 형식 변환 (YYYY-MM-DD)
    try:
        dt = pd.to_datetime(date_str)
    except:
        return None

    for name, (start, end) in HOLIDAY_PERIODS.items():
        if pd.to_datetime(start) <= dt <= pd.to_datetime(end):
            return name
    return None

# =============================================================================
# 2. Metric 계산 함수
# =============================================================================
def calc_metrics(true, pred):
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true - pred))
    
    # MAPE (0으로 나누기 방지)
    true_safe = np.where(true == 0, 1e-6, true)
    mape = np.mean(np.abs((true - pred) / true_safe)) * 100
    
    return rmse, mae, mape

# =============================================================================
# 3. Main Visualization Function
# =============================================================================
def visualize_holiday_performance(model, test_loader, args, save_dir="vis_results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\n>> [Visualization] Calculating metrics & Generating optimized plots...")
    
    model.eval()
    
    # ---------------------------------------------------------
    # Step 1: 전체 데이터 수집 (Global Metrics 계산용)
    # ---------------------------------------------------------
    all_preds, all_trues, all_slots = [], [], []
    batch_data_list = [] 
    
    with torch.no_grad():
        for batch_x, batch_y, batch_bench, batch_dates in test_loader:
            bx = batch_x.to(args.device)
            if args.use_covar:
                output = model(bx[:,:,:-1], bx[:,:,-1], batch_y[:,:,-1].to(args.device))
                true_y = batch_y[:,:,:-1]
            else:
                output = model(bx)
                true_y = batch_y
            
            p_np = output.cpu().numpy()
            t_np = true_y.cpu().numpy()
            s_np = batch_bench.cpu().numpy()
            
            all_preds.append(p_np)
            all_trues.append(t_np)
            all_slots.append(s_np)
            
            batch_data_list.append({
                'pred': p_np, 'true': t_np, 'slot': s_np, 'dates': batch_dates
            })

    full_pred = np.concatenate(all_preds, axis=0)
    full_true = np.concatenate(all_trues, axis=0)
    full_slot = np.concatenate(all_slots, axis=0)
    
    # Global Metrics Calculation
    global_metrics = {}
    target_cols = ['Arrival', 'Departure'] if args.target == 'both' else [args.target.capitalize()]
    
    for i, col in enumerate(target_cols):
        idx = i if args.target == 'both' else 0
        f_p = full_pred[:, :, idx].flatten()
        f_t = full_true[:, :, idx].flatten()
        f_s = full_slot[:, :, idx].flatten()
        
        valid_mask = f_s != 0
        if np.sum(valid_mask) > 0:
            g_rmse_my, _, _ = calc_metrics(f_t[valid_mask], f_p[valid_mask])
            g_rmse_sl, _, _ = calc_metrics(f_t[valid_mask], f_s[valid_mask])
            global_metrics[col] = {'rmse_my': g_rmse_my, 'rmse_sl': g_rmse_sl, 'diff_rmse': g_rmse_sl-g_rmse_my}
        else:
            global_metrics[col] = {'rmse_my': 0, 'rmse_sl': 0}

    # ---------------------------------------------------------
    # Step 2: 명절 기간 시각화 (Y축 스케일링 & X축 라벨링 최적화)
    # ---------------------------------------------------------
    plotted_periods = set()
    
    for batch_data in batch_data_list:
        pred_np = batch_data['pred']
        true_np = batch_data['true']
        slot_np = batch_data['slot']
        batch_dates = batch_data['dates']
        
        batch_size = pred_np.shape[0]
        for b in range(batch_size):
            start_date_str = batch_dates[b] if isinstance(batch_dates, tuple) else batch_dates[b]
            start_dt = pd.to_datetime(start_date_str)
            dates_list = [start_dt + timedelta(days=i) for i in range(args.pred_len)]
            dates_str_list = [d.strftime('%Y-%m-%d') for d in dates_list]
            
            mid_date = dates_str_list[len(dates_str_list)//2]
            period_name = check_if_in_period(mid_date)
            
            if period_name is None: continue
            if period_name in plotted_periods: continue
            plotted_periods.add(period_name)
            
            num_plots = 2 if args.target == 'both' else 1
            # 가로 16, 세로 6 * plots (와이드 비율 유지)
            fig, axes = plt.subplots(num_plots, 1, figsize=(16, 6 * num_plots))
            if num_plots == 1: axes = [axes]
            
            # 명절 날짜
            h_start_str, h_end_str = HOLIDAY_PERIODS[period_name]
            h_start_dt = pd.to_datetime(h_start_str)
            h_end_dt = pd.to_datetime(h_end_str)
            
            for i, col_name in enumerate(target_cols):
                ax = axes[i]
                idx = i if args.target == 'both' else 0
                
                y_pred = pred_np[b, :, idx]
                y_true = true_np[b, :, idx]
                y_slot = slot_np[b, :, idx]
                
                # Local Metrics
                l_rmse_my, l_mae_my, l_mape_my = calc_metrics(y_true, y_pred)
                l_rmse_sl, l_mae_sl, l_mape_sl = calc_metrics(y_true, y_slot)
                g_stats = global_metrics[col_name]
                
                # 통계 텍스트
                stats_txt = (
                    f"[{col_name} Performance Comparison]\n"
                    f"─────────────────────────────────────\n"
                    f"● GLOBAL (All Dates)\n"
                    f"  RMSE: (My) {g_stats['rmse_my']:,.0f} VS (Slot) {g_stats['rmse_sl']:,.0f}\n"
                    f"  Diff: {g_stats['diff_rmse']:+,.0f}\n"
                    f"─────────────────────────────────────\n"
                    f"● LOCAL ({period_name})\n"
                    f"  RMSE: (My) {l_rmse_my:,.0f} VS (Slot) {l_rmse_sl:,.0f}\n"
                    f"  Diff: {l_rmse_sl - l_rmse_my:+,.0f}"
                )
                
                x_axis = np.arange(len(dates_str_list))
                
                # [X축 설정] 날짜 띄엄띄엄 표시 (가독성 확보)
                # 데이터 길이에 따라 간격 자동 설정 (최대 10~12개 틱만 표시되도록)
                n_ticks = len(dates_str_list)
                step = max(1, n_ticks // 10) 
                
                # 배경 하이라이트
                holiday_indices = []
                for d_idx, d_str in enumerate(dates_str_list):
                    curr_dt = pd.to_datetime(d_str)
                    if h_start_dt <= curr_dt <= h_end_dt:
                        holiday_indices.append(d_idx)
                        
                # [Y축 스케일링] 실제 값 범위 + 상단 여유 공간(Headroom)
                # 0부터 시작하지 않고 min 값 주변에서 시작
                data_min = min(np.min(y_true), np.min(y_pred), np.min(y_slot))
                data_max = max(np.max(y_true), np.max(y_pred), np.max(y_slot))
                data_range = data_max - data_min
                
                # 하단은 조금만 여유, 상단은 텍스트 박스 고려해 많이 여유
                y_bottom = data_min - (data_range * 0.1) 
                y_top = data_max + (data_range * 0.45) # 45% 여유 (박스 공간)
                
                ax.set_ylim(bottom=y_bottom, top=y_top)

                if holiday_indices:
                    ax.axvspan(holiday_indices[0]-0.5, holiday_indices[-1]+0.5, 
                               color='gold', alpha=0.2, label='Holiday Period')
                    
                    # 명절 이름 (그래프 상단에 표시)
                    mid_h_idx = (holiday_indices[0] + holiday_indices[-1]) / 2
                    text_y_pos = data_max + (data_range * 0.05) # 데이터 바로 위에
                    ax.text(mid_h_idx, text_y_pos, period_name.replace('_', ' '), 
                            ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkorange')

                # 그래프 그리기
                ax.plot(x_axis, y_true, label='Actual', color='black', marker='o', markersize=4, linewidth=2, zorder=3)
                ax.plot(x_axis, y_slot, label='Slot Algo', color='dodgerblue', linestyle='--', marker='s', markersize=4, alpha=0.8)
                ax.plot(x_axis, y_pred, label=f'{args.model} (My)', color='crimson', linestyle='-', marker='^', markersize=6, linewidth=2.5)
                
                # 값 텍스트 (간격을 두어 표시, 데이터가 많으면)
                text_step = max(1, len(y_true) // 15)
                for j in range(0, len(y_true), text_step):
                    val = y_true[j]
                    ax.text(j, val, f"{int(val):,}", ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # 통계 박스 (좌측 상단, 그래프와 겹치지 않게 Headroom 활용)
                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
                ax.text(0.01, 0.96, stats_txt, transform=ax.transAxes, fontsize=10, 
                        fontfamily='monospace', verticalalignment='top', bbox=props)
                
                ax.set_title(f"{period_name} Forecast - {col_name}", fontsize=16, fontweight='bold')
                
                # X축 틱 설정
                ax.set_xticks(x_axis[::step])
                xtick_labels = [dates_str_list[i] for i in range(0, n_ticks, step)]
                ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=10)
                
                ax.set_ylabel("Passengers")
                ax.legend(loc='upper right', frameon=True, fancybox=True)
                ax.grid(True, alpha=0.4, linestyle='--')
            
            plt.tight_layout()
            
            file_name = f"{period_name}_{args.target}_{args.model}_{args.pred_len}_{args.use_covar}_{args.use_dbloss}.png"
            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"   -> Saved: {save_path}")

    print(">> [Visualization] Done.\n")