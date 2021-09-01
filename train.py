from Params import Params
from dataset_classes import create_dataset
from models import create_model


import time
import os
import sys
import os.path as path
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self):
        self.load_datasets()
        self.create_model()

        if Params.isTrue('ContinueTrain'):
            self.prepare_continued_training()
        else:
            self.start_epoch = 0


    def load_datasets(self):
        self.training_set = create_dataset('train')
        self.validation_set = create_dataset('valid')

        self.train_dl = torch.utils.data.DataLoader(
            self.training_set,
            batch_size = Params.BatchSize,
            shuffle = True,
            num_workers = 6
        )

        self.valid_dl = torch.utils.data.DataLoader(
            self.validation_set,
            batch_size = 256,
            shuffle = True,
            num_workers = 6
        )


    def prepare_logs(self):
        os.makedirs( path.join(Params.CheckpointDir, Params.ModelName), exist_ok=True )

        self.logs = {
            'train': path.join( Params.CheckpointDir, Params.ModelName, 'training.log' ),
            'valid': path.join( Params.CheckpointDir, Params.ModelName, 'validation.log' )
        }

        # Do not clear log files if we are continuing training a previous model
        if not Params.isTrue('ContinueTrain'):
            for log_filename in self.logs.values():
                print( f"Clearing log file {log_filename}" )
                if path.exists(log_filename):
                    os.remove(log_filename)


    def prepare_continued_training(self):
        # Find latest model trained so far

        model_path = path.join( Params.CheckpointDir, Params.ModelName )
        # Thought about using a regular expression, but just a splice and startswith/endswith should suffice (it's not like we're expecting malformed filenames)
        saved_models = [ (int(filename[6:-3]),filename) for filename in os.listdir(model_path) if filename.endswith('.pt') and filename.startswith('epoch_') ]
        saved_models.sort(key = lambda t: t[0], reverse=True)

        print( f"Loading model at epoch {saved_models[0][0]}. Filename is {saved_models[0][1]}" )
        self.start_epoch = saved_models[0][0]
        self.model.load( path.join(model_path, saved_models[0][1]) )

        # Get Learning Rate scheduler up to appropriate point
        for k in range(self.start_epoch):
            self.scheduler.step()


    def create_model(self):
        self.model = create_model(Params.ModelType)
        self.model.cuda()

        if Params.UseScheduler:
            self.scheduler = Params.create_scheduler(self.model.optimizer)


    def run_validation(self, epoch):
        print("Validation")
        accuracy_record = { 'top1': 0, 'top5': 0 }
        valid_loss_total = 0.0

        with torch.no_grad():
            self.model.eval()
            for iter, data in enumerate(tqdm(self.valid_dl)):
                images = data['image'].cuda()
                targets = data['category'].cuda()

                # Use auxiliary classifiers, because we want to be able to compare
                # the validation loss to the training loss
                batch_loss_record, batch_accuracy = self.model.test(images, targets, use_aux=True)

                valid_loss_total += batch_loss_record['final'].item() * len(targets)
                for attribute in ['top1', 'top5']:
                    accuracy_record[attribute] += batch_accuracy[attribute]

        # Save validation results
        denom = len(self.validation_set)
        perf_string = f"Loss {valid_loss_total / denom:.3f} " \
            f"Top1 {100 * accuracy_record['top1'] / denom:.5f} " \
            f"Top5 {100 * accuracy_record['top5'] / denom:.5f}\n"

        with open(self.logs['valid'], 'a') as log_file:
            log_file.write( f"Epoch {epoch+1} TotalIter {self.total_iter} {perf_string}" )
        print( perf_string )


    def inner_training_loop(self, epoch):
        train_log_buffer = []

        print("Training")
        self.model.train()
        for iter, data in enumerate(tqdm(self.train_dl)):
            images = data['image'].cuda()
            targets = data['category'].cuda()
            train_loss_record = self.model.training_step(images, targets)
            self.total_iter += len(targets)

            train_log_buffer.append(
                f"Epoch {epoch+1} Iter {iter} TotalIter {self.total_iter} " \
                f"Loss {train_loss_record['final'].item():.3f}\n"
            )

        # Save training losses for one epoch in one go (keeps logs still valid in case of interruption mid-training)
        with open(self.logs['train'], 'a') as log_file:
            log_file.writelines(train_log_buffer)


    def train(self):
        self.prepare_logs()

        # Initializes to zero, unless we're continuing training
        self.total_iter = self.start_epoch * len(self.training_set)

        for epoch in range(self.start_epoch, Params.NumEpochs):

            print( f"\nEpoch {epoch+1} out of {Params.NumEpochs}:" )
            tic = time.perf_counter()

            self.inner_training_loop(epoch)
            self.run_validation(epoch)

            if Params.UseScheduler:
                self.scheduler.step()

            # Save checkpoint
            if ((epoch+1) % Params.CheckpointFreq) == 0:
                filename = path.join(Params.CheckpointDir, Params.ModelName, f"epoch_{epoch+1}.pt")
                print( f"Saving checkpoint to {filename}" )
                self.model.save(filename)

            toc = time.perf_counter() - tic
            print( f"Epoch took {toc:.1f} seconds. Estimated remaining time {toc*(Params.NumEpochs - epoch - 1)/60:.1f} minutes" )

        # Save final model
        filename = path.join(Params.CheckpointDir, Params.ModelName, f"final.pt")
        self.model.save(filename)



if __name__ == "__main__":
    t = Trainer()
    t.train()
