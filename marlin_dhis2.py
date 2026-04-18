from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io
import time

from dhis2_src.helper_functions import (
    load_models,
    preprocess_data,
    adjust_trailing_zero_prevalence,
    generate_predictions_per_run,
)

app = FastAPI(title="MARLIN Inference API")

# Loading models
WINDOW_SIZE = 10  # Don't change this

MODEL_EIR_PATH = (
    "src/trained_model/causal/LSTM_EIR_4_layers_new_assymetric_9500runs_prevall.pth"
)
MODEL_INC_PATH = (
    "src/trained_model/causal/LSTM_Incidence_3_layers_15000run_causal_W10.pth"
)

model_eir, model_inc, device = load_models(MODEL_EIR_PATH, MODEL_INC_PATH)


# Health Check
@app.get("/")
def health():
    return {"status": "MARLIN ready 🐟"}


# Prediction Endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    run_column: str = "run",  # This corresponds to a Geographic Unit, say Region, District...
    time_column: str = "t",  # This corresponds to Time, say Month ...
    prevalence_column: str = "prev_true",  # The actual timeseries of prevalence
):
    """
    Upload a CSV or Parquet file containing prevalence data.
    Returns inferred EIR and Incidence per run.
    """

    start_time = time.time()

    # What i think is probable to Load file ---- Modify as necessary to take in JSON or other format
    try:
        if file.filename is None:
            raise HTTPException(status_code=400, detail="Filename is missing")
        if file.filename.endswith(".csv"):
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".parquet"):
            contents = await file.read()
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # Validating required inputs are available (You can take this off if you are sure they will alwys be available)
    required_cols = {run_column, time_column, prevalence_column}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {required_cols - set(df.columns)}",
        )

    runs = df[run_column].unique().tolist()

    # Preprocess
    df = adjust_trailing_zero_prevalence(
        df, prevalence_column=prevalence_column, seed=42
    )

    df_scaled, has_true_values = preprocess_data(df)

    if df_scaled is None:
        raise HTTPException(status_code=400, detail="Preprocessing failed")

    # Predicting
    run_results = generate_predictions_per_run(
        df, runs, run_column, WINDOW_SIZE, model_eir, model_inc, device, has_true_values
    )

    # Output
    output = {}

    for run, result in run_results.items():
        output[run] = {
            "eir_pred": result["eir_preds_unscaled"][:, 0].tolist(),
            "inc_pred": result["inc_preds_unscaled"][:, 0].tolist(),
        }

    return {
        "n_runs": len(output),
        "runtime_seconds": round(time.time() - start_time, 3),
        "predictions": output,
    }
