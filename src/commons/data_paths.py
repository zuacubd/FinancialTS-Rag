import os
import sys
import torch


# Root directory (data, models, and output)
root_dir="/Users/zia-mac/Documents/projects/data/financial_data"

# Data root
data_root = os.path.join(root_dir, "fin_seer")

# Models root
models_root = os.path.join(root_dir, "models")

# Output root (contains preprocessed data and results)
# Currently it is located in the code directory, please change it according to your storage strategy)
output_root="/Users/zia-mac/Documents/projects/information-retrieval/FinSeer"

# Train, validation, and test directory
train_data_dir = os.path.join(data_root, "train")
valid_data_dir = os.path.join(data_root, "valid")
test_data_dir = os.path.join(data_root, "test")

# Models path
stock_llm_dir = os.path.join(models_root, "StockLLM")
def get_model_dir(model_name):
    if model_name == "StockLLM":
       return stock_llm_dir

# Output path
output_dir = os.path.join(output_root, "output")

# Preprocessed path
preprocessed_dir = os.path.join(output_dir, "preprocessed")

#Parameters


# GPU or CPU or MPS device
if torch.backends.mps.is_available():
    device= torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
