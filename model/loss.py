import torch
import torch.nn.functional as F
import utils
from model.yolo_loss import YoloLoss, YoloDistLoss, LineLoss


def sigmoid_BCE_loss(y_input, y_target):
    return F.binary_cross_entropy_with_logits(y_input, y_target)

def MSE(y_input, y_target):
    return F.mse_loss(y_input, y_target.float())



