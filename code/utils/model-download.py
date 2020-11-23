import torchvision.models as models
import torch
import os


resnet18 = models.resnet18(pretrained=True)

full_fn = os.path.join("..","anns","resnet-18.pth")

torch.save(resnet18, full_fn)