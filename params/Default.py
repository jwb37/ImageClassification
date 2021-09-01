import torch

# Directory to store checkpoints/results in under checkpoints folder
ModelName = 'GoogleNet'


# Model Parameters
ModelType = 'GoogleNet'
BatchNorm = True
UseSqueezeExcitation = False


# Training parameters
NumEpochs = 60
BatchSize = 16
CropSize = 256
ContinueTrain = False

create_optimizer = lambda model_params: torch.optim.SGD(model_params, lr=0.01)

UseScheduler = True
create_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, NumEpochs // 3, gamma=0.1)


# Paths and Outputs
CheckpointFreq = 10
CheckpointDir = './checkpoints'

# Dataset
DatasetType = 'Textfile'
NumClasses = 196
ImageDir = './imgs/cars'
TextFileDir = './set_files'
