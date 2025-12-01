import os
import random
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def save_checkpoint(state, save, ID):
	if not os.path.exists(save):
		os.makedirs(save)
	filename = os.path.join(save, f'{ID}.pth')
	torch.save(state, filename)
 
 
def get_logger(name, logpath, level=logging.INFO, console=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    log_dir = os.path.dirname(logpath)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(logpath, mode='w')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        
    return logger


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_next_batch(dataloader):
	# Make the union of all time points and perform normalization across the whole dataset
	data_dict = dataloader.__next__()
	
	return data_dict


def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()


def save_vis_plot(model, raw_loader, args, target, experimentID, epoch):
    # 1. 마지막 배치 가져오기
    last_batch = None
    for batch in raw_loader:
        last_batch = batch
    
    if isinstance(last_batch, dict):
        temp_x = last_batch['x']
        temp_y = last_batch['y']
    elif isinstance(last_batch, list) or isinstance(last_batch, tuple):
        temp_x, temp_y = last_batch
        
    # 2. 모델 추론을 위한 입력 준비 (배치 전체)
    vis_input = temp_x[-1].unsqueeze(0).clone().detach().to(args.device)
    
    # 결과 저장 폴더 생성
    save_dir = f"results/{experimentID}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 3. 모델 추론 (한 번만 수행하여 모든 feature에 대한 예측값 확보)
    model.eval()
    with torch.no_grad():
        outputs = model(vis_input)
    
    # 4. 첫 번째(0)와 두 번째(1) 칼럼에 대해 각각 시각화 반복
    # 데이터의 feature 차원 수 확인 (batch, seq_len, features)
    num_features = temp_x.shape[-1]
    
    column_names = ['arrival', 'departure']
    if target == 'arrival':
        target_columns = [0]
    elif target == 'departure':
        target_columns = [1]
    
    for col_idx in target_columns:
        # 데이터에 해당 칼럼이 존재하는지 확인
        if col_idx >= num_features:
            print(f"Warning: 데이터의 Feature 수가 {num_features}개이므로 {col_idx+1}번째 칼럼을 시각화할 수 없습니다.")
            continue

        # --- 해당 칼럼(col_idx) 데이터 추출 ---
        vis_history_seq = temp_x[-1, :, col_idx].detach().cpu().numpy() 
        vis_true_seq = temp_y[-1, :, col_idx].detach().cpu().numpy()
        pred_seq = outputs[0, :, col_idx].detach().cpu().numpy()

        # 시각화 설정
        viz_len = 7  # 보여줄 과거 데이터 길이
        full_len = len(vis_history_seq) # 전체 시퀀스 길이
        
        # 시작 인덱스 계산
        start_idx = max(0, full_len - viz_len)
        
        # 1. History 데이터 자르기 (그래프용)
        plot_history_y = vis_history_seq[start_idx:]
        plot_history_x = list(range(start_idx, full_len))
        
        # 2. Future 데이터 연결 (연결점은 그대로 유지)
        true_connected = np.concatenate([vis_history_seq[-1:], vis_true_seq[-args.pred_len:]])
        pred_connected = np.concatenate([vis_history_seq[-1:], pred_seq[-args.pred_len:]])
        
        # 3. Future X축 설정 (History의 끝인 full_len - 1 부터 시작)
        plot_future_x = list(range(full_len - 1, full_len + args.pred_len))

        # --- 그래프 그리기 ---
        plt.figure(figsize=(14, 7))
        
        # History Plot
        plt.plot(plot_history_x, plot_history_y, label='History', color='gray', alpha=0.5, marker='o', markersize=3)
        
        # Ground Truth Plot
        plt.plot(plot_future_x, true_connected, label='Ground Truth', color='blue', marker='o', markersize=4)
        
        # Prediction Plot
        plt.plot(plot_future_x, pred_connected, label=f'{args.model} (Pred)', color='red', linestyle='--', marker='o', markersize=4)
        
        # 값 텍스트 표시 (Future 구간만)
        for i, (x, y) in enumerate(zip(plot_future_x, true_connected)):
            plt.text(x, y, f'{y:.0f}', fontsize=8, color='blue', ha='center', va='bottom', fontweight='bold')

        for i, (x, y) in enumerate(zip(plot_future_x, pred_connected)):
            if i == 0: continue # 연결점 중복 표시 방지
            plt.text(x, y, f'{y:.0f}', fontsize=8, color='red', ha='center', va='top')

        # 제목에 칼럼 정보 추가
        plt.title(f"Forecast Result (Epoch {epoch}) - {column_names[col_idx]})")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 파일명에 col0, col1 구분 추가
        save_path = f"{save_dir}/{column_names[col_idx]}_{args.model}_Epoch{epoch}.png"
        plt.savefig(save_path)
        plt.close()
        

def calculate_metrics(true, pred):
    """메트릭 계산 헬퍼 함수"""
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true - pred))
    
    # MAPE calculation (prevent division by zero)
    true_safe = np.where(true == 0, 1e-6, true) 
    mape = np.mean(np.abs((true - pred) / true_safe)) * 100
    
    return mse, rmse, mae, mape

def save_slop_plot(model, raw_loader, args, target, experimentID, epoch):
    # 1. SLOP 데이터 및 날짜 정의 (고정된 7일 데이터)
    slop_dates = ['25.07.25', '25.07.26', '25.07.27', '25.07.28', '25.07.29', '25.07.30', '25.07.31']
    slop_arrival = np.array([104355, 106893, 108139, 105142, 101979, 111863, 112079])
    slop_departure = np.array([125333, 125479, 124038, 125028, 115311, 122402, 113592])
    
    slop_len = len(slop_dates)  # 7일

    # 2. 마지막 배치 데이터 추출
    last_batch = None
    for batch in raw_loader:
        last_batch = batch
    
    if isinstance(last_batch, dict):
        temp_x = last_batch['x']
        temp_y = last_batch['y']
    elif isinstance(last_batch, (list, tuple)):
        temp_x, temp_y = last_batch

    # 3. 모델 예측 (한 번만 수행)
    vis_input = temp_x[-1].unsqueeze(0).clone().detach().to(args.device)
    
    if not os.path.exists(f"results/{experimentID}/"):
        os.makedirs(f"results/{experimentID}/")
        
    model.eval()
    with torch.no_grad():
        outputs = model(vis_input)

    # 4. 첫 번째(0)와 두 번째(1) 칼럼에 대해 반복 처리
    num_features = temp_x.shape[-1]
    column_names = ['arrival', 'departure']
    if target == 'arrival':
        target_columns = [0]
    elif target == 'departure':
        target_columns = [1]

    for col_idx in target_columns:
        if target == 'arrival':
            y_label_str = "Arrival Passengers"
            slop_data_ref = slop_arrival
        else:
            y_label_str = "Departure Passengers"
            slop_data_ref = slop_departure
            
        if col_idx >= num_features:
            print(f"Warning: Feature 수가 {num_features}개여서 Col {col_idx}를 시각화할 수 없습니다.")
            continue

        # --- 데이터 슬라이싱 ---
        vis_history_seq = temp_x[-1, :, col_idx].detach().cpu().numpy()
        vis_true_seq = temp_y[-1, :, col_idx].detach().cpu().numpy()
        pred_seq = outputs[0, :, col_idx].detach().cpu().numpy()
        
        slop_data = slop_data_ref 

        # --- 데이터 구간 설정 ---
        pred_len = args.pred_len
        
        # 전체 예측 구간 데이터
        true_full = vis_true_seq[-pred_len:]
        pred_full = pred_seq[-pred_len:]
        
        # 메트릭 계산 및 SLOP 비교를 위한 "마지막 7일" 데이터 슬라이싱
        compare_len = min(pred_len, slop_len)
        
        true_last_7 = true_full[-compare_len:]
        pred_last_7 = pred_full[-compare_len:]
        slop_last_7 = slop_data[-compare_len:] 

        # 메트릭 계산 (마지막 7일 기준)
        my_mse, my_rmse, my_mae, my_mape = calculate_metrics(true_last_7, pred_last_7)
        slop_mse, slop_rmse, slop_mae, slop_mape = calculate_metrics(true_last_7, slop_last_7)

        # 텍스트 박스 내용
        my_stats_text = (
            f"[ {args.model} - Last {compare_len} Days]\n"
            f"MSE : {my_mse:,.0f}\n"
            f"RMSE: {my_rmse:,.0f}\n"
            f"MAE : {my_mae:,.0f}\n"
            f"MAPE: {my_mape:.2f}%"
        )
        slop_stats_text = (
            f"[ SLOP - Last {compare_len} Days]\n"
            f"MSE : {slop_mse:,.0f}\n"
            f"RMSE: {slop_rmse:,.0f}\n"
            f"MAE : {slop_mae:,.0f}\n"
            f"MAPE: {slop_mape:.2f}%"
        )

        # 5. Plotting 좌표 설정
        full_len = len(vis_history_seq)
        
        # History X축 (14일치)
        viz_hist_len = 14
        hist_start_idx = max(0, full_len - viz_hist_len)
        x_history = np.arange(hist_start_idx, full_len)
        y_history = vis_history_seq[hist_start_idx:]
        
        # Future X축
        x_future = np.arange(full_len, full_len + pred_len)
        
        # 연결선
        x_future_connected = np.concatenate(([full_len-1], x_future))
        true_connected = np.concatenate(([vis_history_seq[-1]], true_full))
        pred_connected = np.concatenate(([vis_history_seq[-1]], pred_full))

        # SLOP X축 (맨 뒤 배치)
        slop_start_x = (full_len + pred_len) - slop_len
        x_slop = np.arange(slop_start_x, full_len + pred_len)
        
        # 6. 그리기
        plt.figure(figsize=(16, 8))
        
        plt.plot(x_history, y_history, label='History', color='gray', alpha=0.5, linewidth=1.5)
        plt.plot(x_future_connected, true_connected, label='Ground Truth', color='black', linewidth=2, marker='o', markersize=4)
        plt.plot(x_future_connected, pred_connected, label=f'{args.model}', color='red', linestyle='--', marker='x', markersize=6, linewidth=2)
        
        # SLOP 그래프
        plt.plot(x_slop, slop_data, label='SLOP', color='green', linestyle='-.', marker='s', markersize=5, linewidth=2)

        # 기준선
        plt.axvline(x=full_len - 1, color='blue', linestyle=':', alpha=0.5)

        # 7. X축 날짜 라벨링
        xticks_pos = []
        xticklabels = []
        all_x = list(range(hist_start_idx, full_len + pred_len))
        
        for x in all_x:
            if x >= slop_start_x:
                date_idx = x - slop_start_x
                if 0 <= date_idx < slop_len:
                    xticks_pos.append(x)
                    xticklabels.append(slop_dates[date_idx])
            elif (x - hist_start_idx) % 2 == 0:
                xticks_pos.append(x)
                xticklabels.append(str(x))

        plt.xticks(xticks_pos, xticklabels, rotation=45, fontsize=10, fontweight='bold')

        # 8. 수치 텍스트 표시
        # (A) GT & Prediction
        for i, x in enumerate(x_future):
            y_t = true_full[i]
            y_p = pred_full[i]
            
            plt.text(x, y_t, f'{int(y_t):,}', fontsize=8, color='black', ha='center', va='bottom', fontweight='bold')
            
            if y_p < y_t:
                va_pos = 'top'
                offset = - (y_t - y_p) * 0.05
            else:
                va_pos = 'bottom'
                offset = (y_p - y_t) * 0.05
                
            xy_text_pos = y_p + offset if va_pos == 'bottom' else y_p - offset
            plt.text(x, xy_text_pos, f'{int(y_p):,}', fontsize=8, color='red', ha='center', va=va_pos, fontweight='bold')

        # (B) SLOP 텍스트
        for i, x in enumerate(x_slop):
            y_s = slop_data[i]
            plt.text(x, y_s, f'{int(y_s):,}', fontsize=8, color='green', ha='center', va='top', 
                     fontweight='bold', bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.6))

        # 통계 박스
        plt.text(0.02, 0.95, my_stats_text, transform=plt.gca().transAxes, fontsize=11, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.5, edgecolor='red'))

        plt.text(0.02, 0.78, slop_stats_text, transform=plt.gca().transAxes, fontsize=11, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.5, edgecolor='green'))

        plt.title(f"Comparison: {args.model} vs SLOP ({column_names[col_idx]})", fontsize=16)
        plt.ylabel(y_label_str)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = f"results/{experimentID}/Metrics_Compare_{column_names[col_idx]}_Epoch{epoch}.png"
        plt.savefig(save_path)
        plt.close()
    