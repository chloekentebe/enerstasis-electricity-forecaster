# Used for building train/val/test splits for training tft
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
import json
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error)
from model import TemporalFusionTransformer
import time
from collections import Counter
import os

directory = Path("processed_data")
file = directory / "main_dataset.csv"
main = pd.read_csv(file)

impor_dir = Path("notebooks")
f = impor_dir / "xgboost_feature_importance.csv"
df_import = pd.read_csv(f)
top_feats = df_import.sort_values("importance", ascending=False).head(100)["feature"].tolist()
keep = ["ontario_demand_mw", "season_fall", "phev_registration", "total_large_load_mw"]
main_trimmed = main[top_feats + keep].copy()
print(main_trimmed)

# chronological split
train_set = main[main["timestamp"] < "2022-12-31"] 
val_set = main[(main["timestamp"] >= "2023-01-01") & (main["timestamp"] < "2025-01-01")] 
test_set = main[main["timestamp"] >= "2025-01-01"] 

train_set = train_set[top_feats + keep]
val_set = val_set[top_feats + keep]
test_set = test_set[top_feats + keep]

features = top_feats + keep
# USING TRAIN SET MEAN AND STD for z-score normalization
f_mean = train_set[features].mean()
f_std = train_set[features].std()

train_set[features] = (train_set[features] - f_mean) / f_std
val_set[features] = (val_set[features] - f_mean) / f_std
test_set[features] = (test_set[features] - f_mean) / f_std
print(f"f_mean: {f_mean}, f_std:{f_std}")
print(f"f_mean length: {len(f_mean)} f_std length: {len(f_std)}")

#checkpoint_dir = Path("primary_model/lucky_optimized_final_checkpoints") #777
checkpoint_dir = Path("primary_model/last_lucky_optimized_checkpoints") #777
checkpoint_dir.mkdir(exist_ok=True)

class ElectricityDemandDataset(Dataset):
    # Divides an hourly dataframe into (168h past, 24h future) windows for training.
    def __init__(self, df, past_feature_names, future_feature_names, target_col,
                 encoder_len=168, decoder_len=24, stride=24):
        self.df = df.reset_index(drop=True)
        self.past_feature_names = past_feature_names
        self.future_feature_names = future_feature_names
        self.target_col = target_col
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.valid_starting_inds = list(range(0,len(df) - encoder_len - decoder_len + 1, stride))
    
    def __len__(self):
        return len(self.valid_starting_inds)
    
    def __getitem__(self, ind):
        # Calculates slice of past and future data
        i = self.valid_starting_inds[ind]
        past_slice = self.df.iloc[i:i + self.encoder_len] # 168 rows of data
        future_slice = self.df.iloc[i + self.encoder_len:i + self.encoder_len + self.decoder_len] # 24 rows of data
        past_feature_dict = {
            n: torch.tensor(past_slice[n].values, dtype=torch.float32).unsqueeze(-1)
            for n in self.past_feature_names
        }
        future_feature_dict = {
            n: torch.tensor(future_slice[n].values, dtype=torch.float32).unsqueeze(-1)
            for n in self.future_feature_names
        }
        target = torch.tensor(future_slice[self.target_col].values, dtype=torch.float32)
        return target, future_feature_dict, past_feature_dict

def classify_feature_names(df, target_col, exclude_columns=None):
    """
    Divides dataframe into past inputs and known future inputs based on naming conventions.
    """
    exclude_columns = set(exclude_columns or []) | {target_col}
    columns = [col for col in df.columns if col not in exclude_columns]
    fut_columns = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
        "is_weekend","is_holiday", "season_fall", "season_spring", "season_summer", "season_winter",
        "phev_registration", "bev_registration", "total_large_load_mw", "embedded_solar_mw"
    ]
    past_columns = [col for col in columns if col not in fut_columns]
    return fut_columns, past_columns

# loss function for quantile predictions (0.1, 0.5, 0.9 as used in paper)
def quant_loss(target, predictions, quantiles=(0.1, 0.5, 0.9), t_weight=2.0, h_weights=None):
    # adds a final dimension with size 1 in order to compare
    loss_list = []
    for num, qt in enumerate(quantiles):
         # subtract forecast (B, 24) from target (B, 24)
         errors = target - predictions[..., num]
         loss = torch.max((qt - 1)*errors, qt*errors)
         weight = t_weight if qt != 0.5 else 1.0
         if h_weights is not None:
              loss = loss * h_weights.unsqueeze(0)
         # pinball (quantile) loss --> model is more penalized when q=0.9 vs q=0.1
         loss_list.append(weight * loss)
    # averaging is smoother for backpropogation
    return torch.stack(loss_list, dim=-1).mean() # (B, 24, 3)

def train_for_one_epoch(device, model, tr_loader, optimizer, tail_weight, hour_weights, max_grad_norm):
    model.train()
    total_loss = 0.0
    for target, fut_dict, p_dict in tr_loader:
        target = target.to(device)
        p_dict = {k: v.to(device) for k, v in p_dict.items()}
        fut_dict = {k: v.to(device) for k, v in fut_dict.items()}
        optimizer.zero_grad()
        forecast, _ = model(p_dict, fut_dict)
        #loss = F.mse_loss(forecast, target)
        loss = quant_loss(target, forecast, quantiles=(0.1, 0.5, 0.9), t_weight=tail_weight, h_weights=hour_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm) # address some oscillations in val loss
        optimizer.step()
        total_loss += loss.item() * target.size(0)
    return total_loss / len(tr_loader.dataset)

def get_val_loss(device, model, val_loader, tail_weight, hour_weights):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for target, fut_dict, p_dict in val_loader:
            target = target.to(device)
            p_dict = {k: v.to(device) for k, v in p_dict.items()}
            fut_dict = {k: v.to(device) for k, v in fut_dict.items()}
            forecast, _ = model(p_dict, fut_dict)
            loss = quant_loss(target, forecast, quantiles=(0.1, 0.5, 0.9), t_weight=tail_weight, h_weights=hour_weights)
            total_loss += loss.item() * target.size(0)
    model.train()
    return total_loss / len(val_loader.dataset)

# objective function for optuna (metrics to optimize)
def objective(trial, train_ds, val_ds, past_feature_names, future_feature_names, device, max_epochs=40):
    bs = trial.suggest_categorical("bs", [32])
    lr = trial.suggest_float("lr", 0.0001, 0.0007, log=True)
    hs = trial.suggest_categorical("hs", [64])
    dropout = trial.suggest_float("dropout", 0.45, 0.6)
    n_heads = trial.suggest_categorical("n_heads", [4])
    num_layers = trial.suggest_categorical("num_layers", [1])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
    tail_weight = trial.suggest_float("tail_weight", 1.2, 1.3)
    hour_weight = trial.suggest_float("hour_weight", 1.05, 1.1)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [1.0])

    tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    model = TemporalFusionTransformer(past_feature_names, future_feature_names, input_size=hs,
                                      hidden_size=hs, n_heads=n_heads, dropout=dropout, num_layers=num_layers).to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    naming = f"final_trial{trial.number:02d}_h{hs}_lr{lr:.2e}_bs{bs}_do{dropout:.2f}_heads{n_heads}_layers{num_layers}_wd{weight_decay:.2e}_gn{max_grad_norm}"
    best_val_loss = float("inf")
    loss_hist = {"train_loss": [], "val_loss": []}
    
    hour_weights = torch.ones(24, device=device)
    hour_weights[4:] = hour_weight

    for epoch in range(max_epochs):
        str = time.time()
        tr_loss = train_for_one_epoch(device, model, tr_loader, optimizer, tail_weight, hour_weights, max_grad_norm)
        elapsed = time.time() - str
        print(f"1 epoch at ~100 features and stride=24: {elapsed/60:1f} minutes")
        curr_val_loss = get_val_loss(device, model, val_loader, tail_weight, hour_weights)
        scheduler.step(curr_val_loss) # decay lr udinr gtraining
        #print(f"current LR: {optimizer.param_groups[0]['lr']}")
        loss_hist["val_loss"].append(curr_val_loss)
        loss_hist["train_loss"].append(tr_loss)

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "val_loss": curr_val_loss, "config": trial.params, "feature_means": f_mean.to_dict(),
            "feature_stds": f_std.to_dict()
        }, checkpoint_dir / f"{naming}_epoch{epoch:02d}.pt")
        
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "val_loss": curr_val_loss, "config": trial.params,
        }, checkpoint_dir / f"{naming}_epoch{epoch:02d}_best.pt")
        
        trial.report(curr_val_loss, epoch)
        if trial.should_prune():
            with open(checkpoint_dir/f"history_{naming}.json", "w") as fl:
                json.dump(loss_hist, fl)
            raise optuna.TrialPruned()
    with open(checkpoint_dir/f"history_{naming}.json", "w") as fl:
                    json.dump(loss_hist, fl)
    return best_val_loss

def calc_r2(fc, tg):
    residual_ss = ((tg - fc) ** 2).sum() # unexplained variance
    total_ss = ((tg - tg.mean()) ** 2).sum()
    return (1 - (residual_ss/total_ss)).item()

@torch.no_grad()
def test_evaluation(model, loader, d_mean, d_std, device):
    model.eval()
    # must collection all predictions and ground truths in order to calculate mae, mape, rmse, and r2
    a_forecasts_mw = []
    a_gts_mw = [] # ground truth
    a_encoder_weights = []
    a_decoder_weights = []
    a_attention_weights = []

    for target, fut_dict, p_dict in loader:
        target = target.to(device)
        p_dict = {k: v.to(device) for k, v in p_dict.items()}
        fut_dict = {k: v.to(device) for k, v in fut_dict.items()}
        
        n_forecast, intp = model(p_dict, fut_dict) # (B, 24, 3)
        forecast_mw = (n_forecast * d_std) + d_mean
        gt_mw = (target * d_std) + d_mean # (B, 24)

        a_forecasts_mw.append(forecast_mw.cpu())
        a_gts_mw.append(gt_mw.cpu())
        a_encoder_weights.append(intp["encoder_vsn_weights"].cpu())
        a_decoder_weights.append(intp["decoder_vsn_weights"].cpu())
        a_attention_weights.append(intp["attention_weights"].cpu())

    # concatenate along the first dimension (rows)) --> column count is 24 (for each hour) so it matches for all
    forecasts_mw = torch.cat(a_forecasts_mw, dim=0).squeeze(-1)
    gts_mw = torch.cat(a_gts_mw, dim=0)
    encoder_weights = torch.cat(a_encoder_weights, dim=0)
    decoder_weights = torch.cat(a_decoder_weights, dim=0)
    attention_weights = torch.cat(a_attention_weights, dim=0)

    # extract quantile = 0.5
    med_forecast = forecasts_mw[...,1]

    # l1 loss is mean absolute error
    mae = F.l1_loss(med_forecast, gts_mw).item()
    rmse = torch.sqrt(F.mse_loss(med_forecast, gts_mw)).item()
    mape = (torch.abs((med_forecast - gts_mw)/gts_mw).mean() * 100).item()
    r2 = calc_r2(med_forecast, gts_mw)

    # quantile coverage (values that are inside  0.1 to 0.9 quantile)
    first, third = forecasts_mw[..., 0], forecasts_mw[..., 2]
    coverage = ((gts_mw >= first) & (gts_mw <= third)).float().mean().item()
    crossing_qs = (forecasts_mw[...,0] > forecasts_mw[..., 2]).float().mean()

    # create list for plotting later in order to compare it with xgboost performance
    r2_everyh = [calc_r2(med_forecast[:,hour], gts_mw[:, hour]) for hour in range(24)]
    mae_everyh = [F.l1_loss(med_forecast[:, hour], gts_mw[:, hour]).item() for hour in range(24)]
    metrics = {"MAPE": mape, "RMSE": rmse,"MAE": mae, "R2": r2, "MAE_H": mae_everyh, "R2_H": r2_everyh,
               "coverage_80": coverage, "crossing_quantiles": crossing_qs}

    return metrics, {"f_mw": forecasts_mw, "t_mw": gts_mw, "median_forecast": med_forecast,
                     "encoder_weights": encoder_weights, "decoder_weights": decoder_weights,
                     "attention_weights": attention_weights}

def tft_feat_importance(encoder_weights, attention_weights, encoder_length=168):
    # encoder weights contain 1 weight for each feature and each hour that has been observed (in the past)
    # attention weights contain 1 for each forecast hour and observed hour
    attention_encoder = attention_weights[:,:,:encoder_length]
    critical_feats = torch.bmm(attention_encoder, encoder_weights) # (64, 24, 168) matrix multiplied by (64, 168, 101) to get (64, 24, 101)
    result = critical_feats.mean(dim=0) # becomes (24, 101)
    return result

if __name__ == "__main__":
    os.environ["PYTROCH_ENABLE_MPS_FALLBACK"] = "1" # to address potential NotImplementedErrors
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device}")

    #ex_columns=["demand_lag_1", "demand_rolling24_mean", "timestamp"]
    ex_columns=["timestamp"]
    future_feature_names, past_feature_names = classify_feature_names(main_trimmed, target_col="ontario_demand_mw", exclude_columns=ex_columns)
    print(f"Past feature names: {past_feature_names} of length {len(past_feature_names)}")
    print(f"Future feature names: {future_feature_names} of length {len(future_feature_names)}")

    train_ds = ElectricityDemandDataset(
        df=train_set,
        past_feature_names=past_feature_names,
        future_feature_names=future_feature_names,
        target_col="ontario_demand_mw", stride=24
    )
    val_ds = ElectricityDemandDataset(
        df=val_set,
        past_feature_names=past_feature_names,
        future_feature_names=future_feature_names,
        target_col="ontario_demand_mw", stride=24
    )
    test_ds = ElectricityDemandDataset(
        df=test_set,
        past_feature_names=past_feature_names,
        future_feature_names=future_feature_names,
        target_col="ontario_demand_mw", stride=24
    )
    print(len(train_ds), len(val_ds), len(test_ds))
    
    study = optuna.create_study(
        direction="minimize",
        # halts trials  that are peforming worse than median
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )
    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds, past_feature_names, future_feature_names, device),
        n_trials=3, timeout=60*60*5
    )
    print("BEST TRIAL:", study.best_trial.number, study.best_trial.params)

    # PERFORMANCE EVALUATION BELOW
    d_mean = f_mean["ontario_demand_mw"]
    d_std = f_std["ontario_demand_mw"]
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)
    cp = torch.load("/Users/chloekentebe/enerstasis-electricity-forecaster/primary_model/lucky_optimized_final_checkpoints/final_trial21_h64_lr6.75e-04_bs32_do0.46_heads4_layers1_wd9.47e-05_gn1.0_epoch28.pt", device)
    hs = 64
    model = TemporalFusionTransformer(
        past_feature_names, future_feature_names,
        input_size=hs, hidden_size=hs,
        n_heads=cp["config"]["n_heads"], dropout=cp["config"]["dropout"], num_layers=1
    ).to(device)
    model.load_state_dict(cp["model_state"])
    metrics, results = test_evaluation(model, test_loader, d_mean, d_std, device)
    print(metrics)
    print(f"Fraction of predictions with cross quantiles: {metrics['crossing_quantiles']:.3f}")

    importance = tft_feat_importance(results["encoder_weights"], results["attention_weights"])

    # dataframe for top 20 features per forecast hour
    t20_h = {}
    for h in range(24):
        h_imp = importance[h]
        inds = h_imp.argsort(descending=True)[:20]
        t20_h[f"hour_{h+1}"] = [past_feature_names[ind] for ind in inds]
    top20_feats = pd.DataFrame(t20_h)
    top20_feats.index = [f"rank_{n+1}" for n in range(20)]
    print(top20_feats)
    top20_feats.to_csv("top20_feats_ph_aug4_quant_2.csv")
    
    f_occurences = Counter()
    for h_list in t20_h.values():
        f_occurences.update(h_list)
    table = pd.DataFrame(f_occurences.most_common(20), columns=["feature", "hours_in_top20"])
    table["percentage_of_24_hours"] = (table["hours_in_top20"]/24 * 100).round(1)
    table.index = range(1, len(table)+1)
    table.to_csv("top20_presence_table_aug4_quant_2.csv")