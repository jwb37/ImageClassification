import torch
from sklearn.metrics import top_k_accuracy_score

from Params import Params


class BaseModel:
    def __init__(self):
        pass


    def cuda(self):
        self.net.cuda()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def to(self, device):
        self.net.to(device)


    def training_step(self, images, targets):
        self.optimizer.zero_grad()

        result = self.net.forward(images)
        result = self.wrap_result(result, True)
        loss_record = self.calculate_losses(result, targets)
        loss_record['final'].backward()

        if Params.isTrue('ClipGradients'):
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), Params.ClipGradientThreshold)

        self.optimizer.step()
        return loss_record


    # Other methods expect result to be in the form of a dict with result['final'] representing the final classification tensor
    # This method can be used by derived classes to prepare the raw result returned by the network into this form.
    def wrap_result(self, result, use_aux):
        return result


    def test(self, images, targets, use_aux=False):
        result = self.net.forward(images)
        result = self.wrap_result(result, use_aux)
        loss_record = self.calculate_losses(result, targets, use_aux)

        # Accuracy counts
        model_prediction = result['final'].cpu().numpy()
        targets_np = targets.cpu().numpy()
        accuracy = {
            'top1': top_k_accuracy_score(targets_np, model_prediction, k=1, normalize=False, labels=range(self.n_classes)),
            'top5': top_k_accuracy_score(targets_np, model_prediction, k=5, normalize=False, labels=range(self.n_classes))
        }

        return [loss_record, accuracy]


    def save(self, filename):
        save_state = dict()
        save_state['model'] = self.net.state_dict()
        save_state['optimizer'] = self.optimizer.state_dict()
        torch.save( save_state, filename )


    def load(self, filename):
        save_state = torch.load( filename )
        self.net.load_state_dict(save_state['model'])
        self.optimizer.load_state_dict(save_state['optimizer'])
