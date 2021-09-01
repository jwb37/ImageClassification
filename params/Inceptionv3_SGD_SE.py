import torch

# Directory to store checkpoints/results in under checkpoints folder
ModelName = 'Inceptionv3_SGD_SE'


# Model Parameters
ModelType = 'Inceptionv3'
BatchNorm = True
UseSqueezeExcitation = True


# Training parameters
NumEpochs = 60
BatchSize = 32
CropSize = 299
ContinueTrain = True


create_optimizer = lambda model_params: torch.optim.SGD(model_params, lr=0.045, momentum=0.9)

UseScheduler = True
create_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)


# Paths and Outputs
CheckpointFreq = 1
CheckpointDir = './checkpoints'


# Dataset
DatasetType = 'Textfile'
NumClasses = 196
ImageDir = './imgs/cars'
TextFileDir = './set_files'
