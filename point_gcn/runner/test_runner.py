import numpy as np
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from torch.utils.data.dataloader import DataLoader

from point_gcn.models.net import MultiLayerGCN
from point_gcn.runner.runner import Runner
from point_gcn.tools.utils import import_class


class TestRunner(Runner):
    def __init__(self, args):
        super(TestRunner, self).__init__(args)
        # loss
        self.loss = nn.CrossEntropyLoss().to(self.output_dev)

    def load_dataset(self):
        feeder_class = import_class(self.args.dataset)
        feeder = feeder_class(
            self.args.data_path, num_points=self.args.num_points,
            k=self.args.knn, phase='test'
        )
        self.num_classes = feeder.num_classes
        self.shape_names = feeder.shape_names
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
        if self.args.weights is None:
            raise ValueError('Please appoint --weights.')
        self.load_model_weights(
            self.model,
            self.args.weights,
            self.args.ignore
        )
        self.load_optimizer_weights(self.optimizer, self.args.weights)
        self.load_scheduler_weights(self.scheduler, self.args.weights)

    def run(self):
        self._eval_model()

    def _eval_model(self):
        self.print_log('Eval Model:')
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
