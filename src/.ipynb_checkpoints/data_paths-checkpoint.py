import os
import sys
import torch


#data paths
data_root="/Users/zia-mac/Documents/projects/data/financial_data/fin_seer"
output_root="/Users/zia-mac/Documents/projects/information-retrieval/FinSeer"

train_data_path = os.path.join(data_root, "train")
valid_data_path = os.path.join(data_root, "valid")
test_data_path = os.path.join(data_root, "test")

#model path
# https://huggingface.co/TheFinAI/FinSeer
# https://huggingface.co/TheFinAI/StockLLM

#output path
output_path = os.path.join(output_root, "output")


#Parameters
if torch.backends.mps.is_available():
    device= torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")