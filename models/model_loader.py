import torch
from .arq_sfanet_2_e5 import SFANet  # Asegúrate de que está en PYTHONPATH

def load_model(weights_path, device, num_classes=12, in_channels=11, cat_num_categories=15, cat_emb_dim=4):
    model = SFANet(
        in_channels=in_channels,
        cat_num_categories=cat_num_categories,
        cat_emb_dim=cat_emb_dim,
        num_classes=num_classes,
        pretrained=False
    )
    if weights_path is None:
        return model
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model