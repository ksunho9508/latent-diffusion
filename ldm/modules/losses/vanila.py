
import torch
from torch import nn 
from taming.modules.losses.lpips import LPIPS 

def l1(x, y):
    return torch.abs(x-y).mean()

def l2(x, y):
    return torch.pow((x-y), 2).mean()

class Vanila(nn.Module):
    def __init__(self, pixelloss_weight=1.0,perceptual_weight=1.0, perceptual_loss="lpips",
                 pixel_loss="l1"):
        super().__init__() 
        assert perceptual_loss in ["lpips", "clips", "dists"]
        assert pixel_loss in ["l1", "l2"] 
        self.pixel_weight = pixelloss_weight
        if perceptual_loss == "lpips":
            print(f"{self.__class__.__name__}: Running with LPIPS.")
            self.perceptual_loss = LPIPS().eval()
        else:
            raise ValueError(f"Unknown perceptual loss: >> {perceptual_loss} <<")
        self.perceptual_weight = perceptual_weight

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2    
 
    def forward(self, inputs, reconstructions,
                 split="train"):  
        rec_loss = self.pixel_loss(inputs, reconstructions) 
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions).mean()
        else:
            p_loss = torch.tensor([0.0])
  
        loss = rec_loss + self.perceptual_weight * p_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),  
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean(), 
                } 
        return loss, log