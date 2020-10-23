import os

import numpy as np
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from point_gcn.models.net import MultiLayerGCN
from point_gcn.runner.runner import Runner
from point_gcn.tools.utils import import_class


class TrainRunner(Runner):
    def __init__(self, args):
        super(TrainRunner, self).__init__(args)
        # loss
        self.loss = nn.CrossEntropyLoss().to(self.output_dev)

    def load_dataset(self):
        feeder_class = import_class(self.args.dataset)
        feeder = feeder_class(
            self.args.data_path, num_points=self.args.num_points,
            k=self.args.knn, phase='train'
        )
        self.num_classes = feeder.num_classes
        self.shape_names = feeder.shape_names
        train_data = DataLoader(
            dataset=feeder,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=8
        )
        self.dataset['train'] = train_data
        self.print_log(f'Train data loaded: {len(feeder)} samples.')

        if self.args.eval_model:
            feeder = feeder_class(
                self.args.data_path, num_points=self.args.num_points,
                k=self.args.knn, phase='test'
            )
            test_data = DataLoader(
                dataset=feeder,
                batch_size=self.args.test_batch_size,
                shuffle=False,
                num_workers=8
            )
            self.dataset['test'] = test_data
            self.print_log(f'Test data loaded: {len(feeder)} samples.')

    def load_model(self):
        model = MultiLayerGCN(
            dropout=self.args.dropout, num_classes=self.num_classes
        )
        self.model = model.to(self.output_dev)

    def initialize_model(self):
        if self.args.weights is not None:
            self.load_model_weights(
                self.model,
                self.args.weights,
                self.args.ignore
            )
            self.load_optimizer_weights(self.optimizer, self.args.weights)
            self.load_scheduler_weights(self.scheduler, self.args.weights)

    def run(self):
        best_epoch = -1
        best_acc = 0.0
        for epoch in range(self.epoch, self.args.num_epochs):
            self._train_model(epoch)
            eval_model = self.args.eval_model and (
                    ((epoch + 1) % self.args.eval_interval == 0) or
                    (epoch + 1 == self.args.num_classifier_epochs))
            if eval_model:
                acc = self._eval_model(epoch)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                self.print_log(
                    'Best accuracy: {:.2f}%, best model: model{}.pt'.format(
                        best_acc * 100.0, best_epoch + 1
                    ))

    def _train_model(self, epoch):
        self.print_log(f'Train Epoch: {epoch + 1}')
        self.model.train()

        loader = self.dataset['train']
        loss_values = []

        self.record_time()
        timer = dict(data=0.0, model=0.0, statistic=0.0)
        for batch_id, (x, adj, label) in enumerate(loader):
            # get data
            x = x.float().to(self.output_dev)
            adj = adj.float().to(self.output_dev)
            label = label.long().to(self.output_dev)
            timer['data'] += self.tick()

            # forward
            pred = self.model(adj, x)
            loss = self.loss(pred, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            timer['model'] += self.tick()

            # statistic
            loss_values.append(loss.item())
            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, lr: {:.5f}'.format(
                        batch_id + 1, len(loader), loss.item(),
                        self.optimizer.param_groups[0]['lr']
                    ))
            timer['statistic'] += self.tick()
        self.scheduler.step()

        mean_loss = np.mean(loss_values)
        self.print_log('Mean training loss: {:.4f}.'.format(mean_loss))
        self.print_log(
            'Time consumption: [Data] {:.1f} min, [Model] {:.1f} min'.format(
                timer['data'] / 60.0, timer['model'] / 60.0
            ))

        if self.args.save_model and (epoch + 1) % self.args.save_interval == 0:
            model_path = os.path.join(
                self.model_path, 'model{}.pt'.format(epoch + 1)
            )
            self.save_weights(
                epoch, self.model, self.optimizer, self.scheduler, model_path
            )

        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('train/classifier_loss', mean_loss, epoch)

    def _eval_model(self, epoch):
        self.print_log(f'Eval Epoch: {epoch + 1}')
        self.model.eval()

        loader = self.dataset['test']
        loss_values = []
        pred_scores = []
        true_scores = []

        for batch_id, (x, adj, label) in enumerate(loader):
            # get data
            x = x.float().to(self.output_dev)
            adj = adj.float().to(self.output_dev)
            label = label.long().to(self.output_dev)

            # forward
            y = self.model(adj, x)
            loss = self.loss(y, label)

            # statistic
            loss_values.append(loss.item())
            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}'.format(
                        batch_id + 1, len(loader), loss.item()
                    ))
            pred = y.max(dim=1)[1]
            pred_scores.append(pred.data.cpu().numpy())
            true_scores.append(label.data.cpu().numpy())
        pred_scores = np.concatenate(pred_scores)
        true_scores = np.concatenate(true_scores)

        mean_loss = np.mean(loss_values)
        overall_acc = accuracy_score(true_scores, pred_scores)
        avg_class_acc = balanced_accuracy_score(true_scores, pred_scores)
        self.print_log('Mean testing loss: {:.4f}.'.format(mean_loss))
        self.print_log('Overall accuracy: {:.2f}%'.format(overall_acc * 100.0))
        self.print_log(
            'Average class accuracy: {:.2f}%'.format(avg_class_acc * 100.0)
        )

        if self.args.show_details:
            self.print_log('Detailed results:')
            report = classification_report(
                true_scores,
                pred_scores,
                target_names=self.shape_names,
                digits=4
            )
            self.print_log(report, print_time=False)

        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('test/loss', mean_loss, epoch)
                writer.add_scalar('test/overall_accuracy', overall_acc, epoch)
                writer.add_scalar(
                    'test/average_class_accuracy', avg_class_acc, epoch
                )

        return overall_acc
