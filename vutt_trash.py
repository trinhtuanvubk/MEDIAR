import torch
import os
from train_tools.data_utils.transforms import get_pred_transforms
from train_tools import *
from train_tools.models import MEDIARFormer
from core.MEDIAR import Predictor, EnsemblePredictor


model_path2 = "./from_phase2.pth"
weights2 = torch.load(model_path2, map_location="cpu")
model_args = {
    "classes": 3,
    "decoder_channels": [1024, 512, 256, 128, 64],
    "decoder_pab_channels": 256,
    "encoder_name": 'mit_b5',
    "in_channels": 3
}

model2 = MEDIARFormer(**model_args)
model2.load_state_dict(weights2, strict=False)

model2 = model2.to(torch.device('cuda'))
model2.eval()

# print(model2)

output_path = "results"

# pred_transforms = get_pred_transforms()
# img_path = "Datasets/Public/images/000_img.tif"
# img_data = pred_transforms(img_path)
# img_data = img_data.unsqueeze(0)
# print(img_data.shape)
# img_data = img_data.to(torch.device("cuda"))


predictor = Predictor(model2, "cuda:0", "Datasets/Public/test_", output_path, algo_params={"use_tta": False})
_ = predictor.conduct_prediction()
