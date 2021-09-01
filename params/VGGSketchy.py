import torch

# Directory to store checkpoints/results in under checkpoints folder
ModelName = 'VGG16_Sketchy'


# Model Parameters
ModelType = 'VGG16'
BatchNorm = True


# Training parameters
NumEpochs = 30
BatchSize = 16
CropSize = 224
ContinueTrain = True

create_optimizer = lambda model_params: torch.optim.SGD(model_params, lr=0.01)

UseScheduler = True
create_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, NumEpochs // 3, gamma=0.1)


# Paths and Outputs
CheckpointFreq = 1
CheckpointDir = './checkpoints'


# Dataset
DatasetType = 'Subdir'
NumClasses = 125
TrainingSet = './imgs/Sketchy/train'
TestSet = './imgs/Sketchy/test'
ValidationSet = './imgs/Sketchy/valid'
