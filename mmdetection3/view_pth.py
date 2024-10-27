   import torch

   checkpoint = torch.load(, map_location='cpu')
   print("Checkpoint keys:", checkpoint.keys())
   print("Model state dict keys:", checkpoint['state_dict'].keys())