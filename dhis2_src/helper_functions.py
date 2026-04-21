import numpy as np
import pandas as pd
import torch


from .inference_model_exp import LSTM_EIR, LSTM_Incidence
from .inference_sequence_creator import create_causal_sequences, create_sequences_assymetric


log_transform = lambda x: np.log(x + 1e-8)
inverse_log_transform = lambda x: np.exp(x) - 1e-8


# ---------------- Helpers ----------------

def load_models(model_eir_path, model_inc_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_eir = LSTM_EIR(input_size=1, architecture=[256, 128, 64, 32])
    model_eir.load_state_dict(torch.load(model_eir_path, map_location=device))
    model_eir.to(device)
    model_eir.eval()

    model_inc = LSTM_Incidence(input_size=2, architecture=[200, 100, 50])
    model_inc.load_state_dict(torch.load(model_inc_path, map_location=device))
    model_inc.to(device)
    model_inc.eval()

    return model_eir, model_inc, device

def preprocess_data(df):
    if not pd.api.types.is_numeric_dtype(df['prev_true']):
        return None, False  
    
    has_true_values = {'EIR_true', 'incall'}.issubset(df.columns)

    if has_true_values:
        df_scaled = df[['prev_true', 'EIR_true', 'incall']].apply(log_transform)
    else:
        df_scaled = df[['prev_true']].apply(log_transform)
    
    return df_scaled, has_true_values

def adjust_trailing_zero_prevalence(df, prevalence_column='prev_true', min_val=0.0003, max_val=0.001, seed=None):
    df = df.copy()
    zeros_mask = df[prevalence_column] == 0
    num_zeros = zeros_mask.sum()
    if num_zeros > 0:
        rng = np.random.default_rng(seed)
        random_values = rng.uniform(min_val, max_val, size=num_zeros)
        df.loc[zeros_mask, prevalence_column] = random_values
    return df


def generate_predictions_per_run(data, selected_runs, run_column, window_size, _model_eir, _model_inc, _device, has_true_values):
    run_results = {}

    for run in selected_runs:
        run_data = data[data[run_column] == run]
        if run_data.empty:
            continue

        scaled_data, _ = preprocess_data(run_data)
        if scaled_data is None:
            continue

        X_eir_scaled, y_eir = create_sequences_assymetric(scaled_data, window_size)
        if len(X_eir_scaled) == 0:
            continue

        X_eir_scaled = X_eir_scaled.to(_device)
        with torch.no_grad():
            eir_preds_scaled = _model_eir(X_eir_scaled.unsqueeze(-1))
            eir_preds_unscaled = inverse_log_transform(eir_preds_scaled.cpu().numpy())

        prev_series_scaled = scaled_data['prev_true'].values[:len(eir_preds_scaled)]

        inc_input_df_scaled = pd.DataFrame({
            'prev_true': prev_series_scaled,
            'EIR_true': scaled_data['EIR_true'].values[:len(eir_preds_scaled)] if 'EIR_true' in scaled_data.columns else eir_preds_scaled[:, 0].cpu().numpy()
        })

        X_inc_input, _ = create_causal_sequences(inc_input_df_scaled, window_size, features=['prev_true', 'EIR_true'])
        X_inc_input = X_inc_input.to(_device)

        with torch.no_grad():
            inc_preds_scaled = _model_inc(X_inc_input)
            inc_preds_unscaled = inverse_log_transform(inc_preds_scaled.cpu().numpy())

        run_results[run] = {
            "eir_preds_scaled": eir_preds_scaled.cpu().numpy(),
            "eir_preds_unscaled": eir_preds_unscaled,
            "inc_preds_unscaled": inc_preds_unscaled,
            "scaled_data": scaled_data,
            "original_data": run_data,
            "y_eir_true": y_eir.cpu().numpy() if y_eir is not None else None
        }

    return run_results
