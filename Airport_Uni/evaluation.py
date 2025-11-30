import torch
import numpy as np
from tqdm import tqdm
from utils import *
from DBLoss import *


def metric(pred, true):
    mse = torch.mean((pred - true) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true))
    mape = torch.mean(torch.abs((pred - true) / (true + 1e-7)))
    
    return mse, rmse, mae, mape


def compute_all_losses(model, batch, args):
    batch_x, batch_y = batch
    
    batch_x = batch_x.to(args.device)
    batch_y = batch_y.to(args.device)
    
    true_y = batch_y[:, -args.pred_len:]

    preds = model(batch_x) 
    
    if preds.shape != true_y.shape:
        preds = preds[:, -args.pred_len:]
    
    mse, rmse, mae, mape = metric(preds, true_y)

    results = {}
    if args.use_dbloss:
        criterion = DBLoss(alpha=0.1, beta=0.1)
        loss = criterion(preds, true_y)
        results["loss"] = loss
    else:
        results["loss"] = mse
        
    results["mse"] = mse.item()
    results["rmse"] = rmse.item()
    results["mae"] = mae.item()
    results["mape"] = mape.item()
    
    return results


def evaluation(model, dataloader, n_dataloader, args):
    model.eval()
    
    total_loss = []
    total_mse = []
    total_rmse = []
    total_mae = []
    total_mape = []
    
    with torch.no_grad():
        for _ in range(n_dataloader):
            batch = get_next_batch(dataloader)
            eval_res = compute_all_losses(model, batch, args)

            total_loss.append(eval_res['loss'])
            total_mse.append(eval_res['mse'])
            total_rmse.append(eval_res['rmse'])
            total_mae.append(eval_res['mae'])
            total_mape.append(eval_res['mape'])

    results = {}
    results["loss"] = sum(total_loss) / n_dataloader
    results["mse"] = np.mean(total_mse)
    results["rmse"] = np.mean(total_rmse)
    results["mae"] = np.mean(total_mae)
    results["mape"] = np.mean(total_mape)

    return results