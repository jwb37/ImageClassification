import torch.nn as nn

from .BaseModel import BaseModel
from networks.Inceptionv3 import Inceptionv3_Net
from Params import Params


ClassificationLoss = nn.CrossEntropyLoss()
AuxClassifierWeight = 0.3


class Inceptionv3_Model(BaseModel):
    def __init__(self, in_channels, n_classes, img_size):
        in_channels, n_classes, img_size = Params.InChannels, Params.NumClasses, Params.CropSize

        assert(in_channels == 3)
        assert(img_size == 299)
        # For this network, no flexibility in number of input channels or image size are allowed

        self.n_classes = n_classes

        if Params.isTrue('ContinueTrain'):
            # Skip initializing weights (saves time)
            self.net = Inceptionv3_Net(n_classes, init_weights=False)
        else:
            self.net = Inceptionv3_Net(n_classes)

        self.optimizer = Params.create_optimizer(self.net.parameters())


    def wrap_result(self, result, use_aux):
        if use_aux:
            return { 'final': result[0], 'aux': result[1] }
        else:
            return { 'final': result }


    def test(self, images, targets, use_aux=False):
        # The prebuilt Inception model in torchvision does not support using
        # auxiliary classifiers during evaluation. We force them off here
        return super().test(images, targets, False)


    def calculate_losses(self, result, targets, use_aux=True):
        loss_record = dict()  
        loss_record['final'] = ClassificationLoss( result['final'], targets )

        if use_aux:
            loss_record['aux'] = ClassificationLoss( result['aux'], targets )
            loss_record['final'] += AuxClassifierWeight * loss_record['aux']

        return loss_record
