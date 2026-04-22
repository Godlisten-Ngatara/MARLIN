#EIR and Incidence Prediction Analysis

This project aims to provide tool for estimating/reconstructing malaria incidence and transmission intensity from ANC timeseries prevalence data through emulation of mechanistic Malaria transmission model. This is a simplified version which includes helper functions for data processing and model prediction

## Folder Structure

```
project_root/
│── dhis2_src                           # Source code/functions directory
│   ├── helper_functions.py                # Contains functions for preprocessing...and other utils
│   ├── inference_sequence_creator.py  # Creates birectional sequences of tensors
│   ├── inference_model_exp.py      # Defines PyTorch model and training
│   ├── _init_.py                   # Initialises python app
│── test/                # Unit tests
│   ├── test_data         # Contains a thousand test runs across different transmission intensities
│── marlin_dhis2/         # Version compatible with DHIS2 Endpoint creation using FAST API
│── requirements.txt      # Dependencies
│── README.md             # Project overview and instructions
│── marlin.py             #Version with streamlit UI



##Installtion and Set-up

- Install Python of Desired Version - This project uses Python 3.12.6
- Clone repo and navigate to project root directory using command prompt (cd /path/to/MARLIN)
- Install Dependecies with "pip install -r requirements.txt" (create python virtual environment in the project directory before installing dependencies if desired. That is the standard practice)
- Run the ANC_Emulator_PyTorch.ipynb notebook or other variants for the training process
- Model Weights and plots are saved in src and plots folder respectively. This can be modified in the respective function creating the plot
- To run the dashboard, while still in the project root (in command prompt), run "streamlit run emulator.py" or the other variants

### Main Components

 **Sequence Creation**
   - Processes time-series data to generate input-output pairs suitable for model training, handling padding for initial time steps.

 **Machine Learning Model**
   - Implements an LSTM-based architecture to predict malaria metrics (`EIR_true`, `incall`) using simulation data.
