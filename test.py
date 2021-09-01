from Params import Params
from dataset_classes import create_dataset
from models import create_model

import time
import os
import sys
import os.path as path
import torch
from tqdm import tqdm


class Tester:
    def __init__(self):
        self.create_dataset()
        self.load_model()

    def create_dataset(self):
        self.test_set = create_dataset('test')

        self.test_dl = torch.utils.data.DataLoader(
            self.test_set,
            batch_size = 256,
            shuffle = True,
            num_workers = 6
        )

    def prepare_log(self):
        self.log_filename = path.join( Params.CheckpointDir, Params.ModelName, 'test_results.log' )

        print( f"Clearing log file {self.log_filename}" )
        if path.exists(self.log_filename):
            os.remove(self.log_filename)

    def load_model(self):
        self.model = create_model(Params.ModelType)
        self.model.cuda()

        model_path = path.join( Params.CheckpointDir, Params.ModelName )

        if path.exists( path.join(model_path, 'final.pt') ):
            latest_file = 'final.pt'
        else:
            # Find saved model with the latest epoch
            saved_models = [ (int(filename[6:-3]),filename) for filename in os.listdir(model_path) if filename.endswith('.pt') and filename.startswith('epoch_') ]
            saved_models.sort(key = lambda t: t[0], reverse=True)
            latest_file = saved_models[0][1]

        print( f"Loading model {latest_file}" )
        self.model.load( path.join(model_path, latest_file) )

    def run_test(self):
        print("Testing")
        self.prepare_log()

        accuracy_record = { 'top1': 0, 'top5': 0 }
        valid_loss_total = 0.0
        with torch.no_grad():
            self.model.eval()
            for iter, data in enumerate(tqdm(self.test_dl)):
                images = data['image'].cuda()
                targets = data['category'].cuda()

                batch_loss_record, batch_accuracy = self.model.test(images, targets)
                valid_loss_total += batch_loss_record['final'].item() * len(targets)

                for attribute in ['top1', 'top5']:
                    accuracy_record[attribute] += batch_accuracy[attribute]

        # Save test results
        result_string =  f"Top1 {100 * accuracy_record['top1'] / len(self.test_set):.5f} "
        result_string += f"Top5 {100 * accuracy_record['top5'] / len(self.test_set):.5f}\n"

        with open(self.log_filename, 'w') as log_file:
            log_file.write(result_string)
        print(result_string)

if __name__ == '__main__':
    tester = Tester()
    tester.run_test()
