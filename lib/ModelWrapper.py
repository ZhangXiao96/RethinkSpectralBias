import torch
import torch.nn as nn
import numpy as np


class ModelWrapper(object):

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        acc = correct / targets.size(0)
        return loss.item(), acc, correct

    def eval_all(self, test_loader):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                loss, correct = self.eval_on_batch(inputs, targets)
                total += targets.size(0)
                test_loss += loss
                test_correct += correct
            test_loss /= (batch_idx+1)
            test_acc = test_correct / total
        return test_loss, test_acc

    def predict_line_fft(self, test_loader, h, nb_interpolation, max_num=500):
        self.model.eval()
        prob_list = []
        steps = np.arange(-h, h, 2*h/nb_interpolation)
        with torch.no_grad():
            nb_x = 0
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                origin_inputs = inputs.data.cpu().numpy()
                nb_x += len(origin_inputs)
                direction = np.random.standard_normal(origin_inputs.shape)
                normed_direction = direction.reshape(len(origin_inputs), -1)
                normed_direction /= np.linalg.norm(normed_direction, ord=2, axis=-1, keepdims=True)
                normed_direction = np.reshape(normed_direction, newshape=origin_inputs.shape)
                results = []
                for step in steps:
                    pert_inputs = step * normed_direction + origin_inputs
                    pert_inputs = torch.from_numpy(pert_inputs).to(self.device, dtype=torch.float)
                    # result = nn.Softmax(-1)(self.model(pert_inputs)).data.cpu().numpy()
                    result = self.model(pert_inputs).data.cpu().numpy()
                    results.append(result)
                results = np.array(results).transpose(1, 2, 0)
                A = np.abs(np.fft.rfft(results, axis=-1))
                # A = np.mean(A, axis=-2, keepdims=False)
                # he_prob = A / np.sum(A, axis=-1, keepdims=True)
                # prob_list.append(he_prob)
                prob_list.append(A)
                if nb_x >= max_num:
                    break
            As = np.concatenate(prob_list, axis=0)
        return As

    def predict_adv_line_fft(self, test_loader, h, nb_interpolation, max_num=500):
        self.model.eval()
        prob_list = []
        steps = np.arange(-h, h, 2*h/nb_interpolation)
        nb_x = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad = True
            outputs = self.model(inputs)
            _, init_pred = outputs.max(1)
            loss = self.criterion(outputs, init_pred)
            self.model.zero_grad()
            loss.backward()
            direction = inputs.grad.data.cpu().numpy()
            origin_inputs = inputs.data.cpu().numpy()
            nb_x += len(origin_inputs)

            normed_direction = direction.reshape(len(origin_inputs), -1)
            normed_direction /= np.linalg.norm(normed_direction, ord=2, axis=-1, keepdims=True)
            normed_direction = np.reshape(normed_direction, newshape=origin_inputs.shape)
            self.model.zero_grad()
            results = []
            for step in steps:
                pert_inputs = step * normed_direction + origin_inputs
                pert_inputs = torch.from_numpy(pert_inputs).to(self.device, dtype=torch.float)
                # result = nn.Softmax(-1)(self.model(pert_inputs)).data.cpu().numpy()
                result = self.model(pert_inputs).data.cpu().numpy()
                results.append(result)
            results = np.array(results).transpose(1, 2, 0)
            A = np.abs(np.fft.rfft(results, axis=-1))
            # A = np.mean(A, axis=-2, keepdims=False)
            # he_prob = A / np.sum(A, axis=-1, keepdims=True)
            # prob_list.append(he_prob)
            prob_list.append(A)
            if nb_x >= max_num:
                break
        As = np.concatenate(prob_list, axis=0)
        return As


    def eval_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        return loss.item(), correct

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()



