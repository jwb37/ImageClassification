# Note: Expected image input size: 224x224

import torch.nn as nn

from .BaseModel import BaseModel
from torchvision.models import vgg16
from Params import Params


ClassificationLoss = nn.CrossEntropyLoss()


class VGG16_Model(BaseModel):
    def __init__(self):
        self.net = vgg16(num_classes=Params.NumClasses)
        self.n_classes = Params.NumClasses
        self.optimizer = Params.create_optimizer(self.net.parameters())

    def wrap_result(self, result, use_aux):
        return {'final': result}

    def calculate_losses(self, result, targets, use_aux=False):
        loss_record = dict()
        loss_record['final'] = ClassificationLoss( result['final'], targets )
        return loss_record
