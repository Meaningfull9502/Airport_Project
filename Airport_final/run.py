import argparse
import os
import itertools
import datetime
import csv
from tqdm import tqdm
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
from dataset import *
from utils import *
from evaluation import *
from models import LSTM, NBeats, TSMixer, PatchTST, iTransformer, TimeXer, TSMixer_Covar
# =============================================================================
# Dynamic Hyperparameter Generator
# =============================================================================
def get_valid_combinations(model_name):
    combinations = []
    
    # 1. Base Parameters (Common)
    # 기본 파라미터 그리드 정의
    lrs = [1e-3, 1e-4, 1e-5]
    seq_lens = [56, 91, 365]
    d_models = [64, 128, 256, 512]
    n_layers = [2, 4, 6]
    
    # 2. Model Categories
    # 모델 성격에 따른 분류
    # patch_based: patch_len이 필요한 모델
    patch_models = ['patchtst', 'timexer'] 
    # transformer_based: n_heads가 필요한 모델
    transformer_models = ['patchtst', 'itransformer', 'timexer'] 

    # 3. Iterate Base Parameters
    # itertools.product를 사용하여 기본 파라미터 조합 생성
    for lr, sl, dm, nl in itertools.product(lrs, seq_lens, d_models, n_layers):
        base_config = {
            'lr': lr,
            'seq_len': sl,
            'd_model': dm,
            'n_layers': nl
        }
        
        # 현재 설정에서 파생될 수 있는 옵션 리스트 초기화
        current_candidates = [base_config]
        
        # 4. Handle n_heads (Transformer Logic)
        if model_name in transformer_models:
            new_candidates = []
            
            # d_model 크기에 따른 n_heads 후보군 설정
            if dm == 64:
                heads_options = [2, 4, 8]
            else:
                heads_options = [2, 4, 8, 16]
            
            for config in current_candidates:
                for h in heads_options:
                    # 유효성 검사: d_model은 n_heads로 나누어 떨어져야 함
                    if dm % h == 0:
                        temp = config.copy()
                        temp['n_heads'] = h
                        new_candidates.append(temp)
            
            current_candidates = new_candidates

        # 5. Handle patch_len (Patch Logic)
        if model_name in patch_models:
            new_candidates = []
            patch_options = [7, 14]
                
            for config in current_candidates:
                for p in patch_options:
                    temp = config.copy()
                    temp['patch_len'] = p
                    new_candidates.append(temp)
            
            current_candidates = new_candidates
            
        # 최종적으로 생성된 후보군을 결과 리스트에 추가
        combinations.extend(current_candidates)

    return combinations
# =============================================================================
# Argument Parser
# =============================================================================
parser = argparse.ArgumentParser('AirPort TS Forecasting Grand Search')

# 1. Environment
parser.add_argument('--gpu', type=str, default='0')

# 2. Data Strategy
parser.add_argument('--target', type=str, default='arrival', choices=['arrival', 'departure', 'both'])
parser.add_argument('--cv_start_year', type=int, default=2017)
parser.add_argument('--cv_end_year', type=int, default=2019)
parser.add_argument('--test_year', type=int, default=2024)
parser.add_argument('--use_corona', action='store_true')

# 3. Training Params
parser.add_argument('--seq_len', type=int, default=28)
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--pred_len', type=int, default=56)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)

# 4. Model Params
parser.add_argument('--model', type=str, default='patchtst')
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--factor', type=int, default=5)
parser.add_argument('--norm_mode', type=str, default='revin')
parser.add_argument('--use_covar', action='store_true')
parser.add_argument('--use_dbloss', action='store_true')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.stride = args.patch_len 
# =============================================================================
# Main Functions
# =============================================================================
def get_model(args):
    full_model_dict = {
        'lstm': LSTM, 'nbeats': NBeats, 'tsmixer': TSMixer, 'patchtst': PatchTST, 
        'itransformer': iTransformer, 'timexer': TimeXer, 'tsmixer_covar': TSMixer_Covar,
    }
    return full_model_dict[args.model].Model(args).to(args.device)


def train_one_epoch(model, optimizer, data_obj, args):
    model.train()
    total_loss = []
    for _ in range(data_obj["n_train_batches"]):
        optimizer.zero_grad()
        batch_dict = get_next_batch(data_obj["train_set"])
        if args.target == 'both':
            res = compute_both_losses(model, batch_dict, args)
        else:
            if args.use_covar:
                res = compute_both_losses(model, batch_dict, args)
            else:
                res = compute_loss(model, batch_dict, args)
        res["loss"].backward()
        optimizer.step()
        total_loss.append(res["loss"].item())
    return torch.tensor(np.mean(total_loss))


def main():
    SEEDS = [1]
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    args.file_path = 'dataset/airport_flight_with_holiday.csv'

    # Candidate Models
    if args.target == 'both':
        candidate_models = ['timexer', 'tsmixer_covar', 'lstm', 'itransformer']
    else:
        if args.use_covar:
            candidate_models = ['lstm', 'patchtst', 'tsmixer_covar']
        else:
            candidate_models = ['lstm', 'patchtst', 'nbeats', 'tsmixer']
    
    # Target Directory
    if args.use_dbloss:
        target_log_dir = os.path.join("logs", f'{args.target}_{args.pred_len}_{args.use_covar}_{args.use_dbloss}')
    else:
        target_log_dir = os.path.join("logs", f'{args.target}_{args.pred_len}_{args.use_covar}')
    if not os.path.exists(target_log_dir): os.makedirs(target_log_dir, exist_ok=True)
    
    # Grand Summary Logger
    summary_path = os.path.join(target_log_dir, f"Grand_Summary_{timestamp}.log")
    summary_logger = get_logger("grand_summary", summary_path, console=True)
    
    summary_logger.info(f"{'='*60}")
    summary_logger.info(f" GRAND SEARCH START | Target: {args.target}")
    summary_logger.info(f" Candidate Models: {candidate_models}")
    summary_logger.info(f"{'='*60}\n")

    # =========================================================================
    # Loop 0: Model Loop (Grand Loop)
    # =========================================================================
    for model_name in candidate_models:
        args.model = model_name
        
        # [Dynamic Params Generator]
        param_combinations = get_valid_combinations(args.model)
        
        # 모델별 로그 경로
        model_log_dir = os.path.join(target_log_dir, args.model)
        detail_log_dir = os.path.join(model_log_dir, "details")
        if not os.path.exists(detail_log_dir): os.makedirs(detail_log_dir, exist_ok=True)
        if not os.path.exists("model_states/"): os.makedirs("model_states/")

        summary_logger.info(f"\n{'='*60}")
        summary_logger.info(f" >>> STARTING MODEL: {args.model.upper()}")
        summary_logger.info(f" >>> Total Combinations: {len(param_combinations)}")
        summary_logger.info(f"{'='*60}")
        
        best_global_rmse = float('inf')
        best_config = None
        # =====================================================================
        # Loop 1: Grid Search (Config)
        # =====================================================================
        for p_idx, params in enumerate(param_combinations):
            
            # 파라미터 적용
            for k, v in params.items():
                setattr(args, k, v)
            
            seed_scores = []
            summary_logger.info(f"   [Config {p_idx+1}/{len(param_combinations)}] Params: {params}")
            # =================================================================
            # Loop 2: Seed Loop
            # =================================================================
            for current_seed in SEEDS:
                fix_seed(current_seed)
                args.seed = current_seed
                
                # Detail Logger (상세 로그 파일)
                detail_filename = f"Detail_Cfg{p_idx+1}_Seed{current_seed}.log"
                detail_path = os.path.join(detail_log_dir, detail_filename)
                detail_logger = get_logger(f"detail_{args.model}_{p_idx}_{current_seed}", detail_path, console=False)
                
                detail_logger.info(f"DETAILED LOG | Model: {args.model} | Seed: {current_seed}")
                detail_logger.info(f"FULL CONFIG: {vars(args)}")
                # =============================================================
                # Loop 3: Nested CV
                # =============================================================
                cv_years = range(args.cv_start_year, args.cv_end_year + 1)
                fold_rmses = []
                
                for val_year in cv_years:
                    experimentID = int(SystemRandom().random()*100000)
                    args.val_year = val_year
                    
                    detail_logger.info(f"\n{'-'*20} [Fold: {val_year}] ExpID: {experimentID} {'-'*20}")
                    
                    # Data Load
                    data_obj = load_dataset(args, is_refit=False)
                    args.ndims = data_obj['ndims']
                    
                    model = get_model(args)
                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    
                    best_inner_val_loss = float('inf')
                    patience_counter = 0
                    best_iter = 0
                    test_res = None
                    
                    # [Pbar]
                    pbar = tqdm(range(args.epochs), desc=f"M:{args.model} Cfg:{p_idx+1} S:{current_seed} F:{val_year}", leave=False)
                    
                    for epoch in pbar:
                        itr = epoch
                        train_loss_val = train_one_epoch(model, optimizer, data_obj, args)
                        train_res = {"loss": train_loss_val} 

                        model.eval()
                        with torch.no_grad():
                            val_res = evaluation(model, data_obj["valid_set"], data_obj["n_valid_batches"], args)
                        
                        if val_res['loss'] < best_inner_val_loss:
                            best_inner_val_loss = val_res['loss']
                            best_iter = epoch
                            patience_counter = 0
                            with torch.no_grad():
                                test_res = evaluation(model, data_obj["eval_set"], data_obj["n_eval_batches"], args)
                        else:
                            patience_counter += 1
                        
                        pbar.set_postfix({
                            'Tr': f"{train_res['loss'].item():.4f}",
                            'Val': f"{val_res['loss'].item():.4f}",
                            'Best': f"{best_inner_val_loss:.4f}",
                            'Pat': f"{patience_counter}/{args.patience}"
                        })

                        # [Log to File]
                        logger = detail_logger
                        logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
                        logger.info("Train - Loss: {:.2f}".format(train_res["loss"].item()))
                        logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}%".format(
                            val_res["loss"].item(), val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100
                        ))
                        if (test_res != None):
                            logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}%".format(
                                best_iter, test_res["loss"], test_res["mse"], test_res["rmse"], test_res["mae"], test_res["mape"]*100
                            ))
                        
                        if patience_counter >= args.patience:
                            detail_logger.info(f"Early Stopping at Epoch {epoch}")
                            break
                    
                    if test_res is None:
                         with torch.no_grad():
                            test_res = evaluation(model, data_obj["eval_set"], data_obj["n_eval_batches"], args)
                    
                    fold_rmses.append(test_res['rmse'])
                    detail_logger.info(f"Fold {val_year} Best RMSE: {test_res['rmse']:.4f}")

                seed_avg = np.mean(fold_rmses)
                seed_scores.append(seed_avg)
                
                close_logger(detail_logger) 
                
            # Config Average
            global_avg_rmse = np.mean(seed_scores)
            summary_logger.info(f"      -> Score: {global_avg_rmse:.4f}")
            
            if global_avg_rmse < best_global_rmse:
                best_global_rmse = global_avg_rmse
                best_config = vars(args).copy()
                summary_logger.info(f"      [!] New Best for {args.model}")
        # =====================================================================
        # PHASE 2: FINAL REFIT (For This Model's Champion)
        # =====================================================================
        summary_logger.info(f"\n   >>> REFIT CHAMPION ({args.model})")
        
        # Best Config 적용
        for k, v in best_config.items():
            setattr(args, k, v)
        if hasattr(args, 'patch_len'): args.stride = args.patch_len 
            
        final_test_results = {'rmse': [], 'mae': [], 'mape': []}
        champion_log_dir = os.path.join(model_log_dir, "details")
        
        for current_seed in SEEDS:
            experimentID = int(SystemRandom().random()*100000)
            fix_seed(current_seed)
            args.seed = current_seed
            
            refit_filename = f"Refit_Champion_{args.model}_Seed{current_seed}.log"
            refit_path = os.path.join(champion_log_dir, refit_filename)
            detail_logger = get_logger(f"refit_{args.model}_{current_seed}", refit_path, console=False)
            detail_logger.info(f"REFIT START | Model: {args.model} | Seed: {current_seed}")
            detail_logger.info(f"FULL CONFIG: {best_config}")
            
            summary_logger.info(f"   > Refitting Seed {current_seed}...")
            
            data_obj_refit = load_dataset(args, is_refit=True)
            model = get_model(args)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            
            best_refit_val_loss = float('inf')
            patience_counter = 0
            best_iter = 0
            test_res = None
            
            pbar = tqdm(range(args.epochs), desc=f"Refit {args.model} Seed {current_seed}", leave=True)
            
            for epoch in pbar:
                itr = epoch
                train_loss_val = train_one_epoch(model, optimizer, data_obj_refit, args)
                train_res = {"loss": train_loss_val}

                model.eval()
                with torch.no_grad():
                    val_res = evaluation(model, data_obj_refit["valid_set"], data_obj_refit["n_valid_batches"], args)
                
                if val_res['loss'] < best_refit_val_loss:
                    best_refit_val_loss = val_res['loss']
                    best_iter = epoch
                    patience_counter = 0
                    
                    with torch.no_grad():
                        test_res = evaluation(model, data_obj_refit["eval_set"], data_obj_refit["n_eval_batches"], args)
                else:
                    patience_counter += 1
                
                pbar.set_postfix({
                    'Tr': f"{train_res['loss'].item():.2f}",
                    'Val': f"{val_res['loss'].item():.2f}",
                    'Best': f"{best_refit_val_loss:.2f}",
                    'Pat': f"{patience_counter}"
                })

                logger = detail_logger 
                logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
                logger.info("Train - Loss: {:.2f}".format(train_res["loss"].item()))
                logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}%".format(
                    val_res["loss"].item(), val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100
                ))
                if (test_res != None):
                    logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}%".format(
                        best_iter, test_res["loss"], test_res["mse"], test_res["rmse"], test_res["mae"], test_res["mape"]*100
                    ))

                if patience_counter >= args.patience:
                    detail_logger.info("Early stopping!")
                    break
                
            final_test_results['rmse'].append(test_res['rmse'])
            final_test_results['mae'].append(test_res['mae'])
            final_test_results['mape'].append(test_res['mape'])
            
            if args.use_dbloss:
                model_save_dir = os.path.join("model_states", f'{args.target}_{args.pred_len}_{args.use_covar}_{args.use_dbloss}', args.model)
            else:
                model_save_dir = os.path.join("model_states", f'{args.target}_{args.pred_len}_{args.use_covar}', args.model)
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir, exist_ok=True)
            save_checkpoint(model.state_dict(), model_save_dir, f'{current_seed}')
            close_logger(detail_logger)

        # ---------------------------------------------------------------------
        # [Leaderboard Save]
        # ---------------------------------------------------------------------
        mean_rmse = np.mean(final_test_results['rmse'])
        std_rmse = np.std(final_test_results['rmse'])
        mean_mae = np.mean(final_test_results['mae'])
        mean_mape = np.mean(final_test_results['mape'])

        summary_logger.info(f"\n[DONE] {args.model} Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} \n{final_test_results['rmse']}")
        summary_logger.info(f"{'='*60}\n")

        # Append to Leaderboard
        comp_log_path = os.path.join(target_log_dir, f"{args.target}_models_comparison.csv")
        file_exists = os.path.isfile(comp_log_path)
        
        with open(comp_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Model', 'Target', 'RMSE_Mean', 'RMSE_Std', 'MAE_Mean', 'MAPE_Mean', 'Best_Params'])
            
            writer.writerow([
                args.model,
                args.target,
                f"{mean_rmse:.4f}",
                f"{std_rmse:.4f}",
                f"{mean_mae:.4f}",
                f"{mean_mape:.4f}",
                str(best_config)
            ])
        
        print(f"[Leaderboard] {args.model} result added to {comp_log_path}")

    print("\n[ALL DONE] Grand Search Finished.")
#===========================================================================
if __name__ == '__main__':
    main()
