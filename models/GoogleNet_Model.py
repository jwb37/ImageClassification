import torch.nn as nn

from .BaseModel import BaseModel
from networks.GoogleNet import GoogleNet
from Params import Params


ClassificationLoss = nn.CrossEntropyLoss()
AuxClassifierWeight = 0.3


class GoogleNet_Model(BaseModel):
    def __init__(self):
        self.net = GoogleNet(Params.InChannels, Params.NumClasses, Params.CropSize)
        self.n_classes = Params.NumClasses
        self.optimizer = Params.create_optimizer(self.net.parameters())

    def calculate_losses(self, result, targets, use_aux=True):
        loss_record = dict()  
        loss_record['final'] = ClassificationLoss( result['final'], targets )

        if use_aux:
            loss_record['aux1'] = ClassificationLoss( result['aux1'], targets )
            loss_record['aux2'] = ClassificationLoss( result['aux2'], targets )
            loss_record['final'] += AuxClassifierWeight * (loss_record['aux1'] + loss_record['aux2'])

        return loss_record
