"""this class build and run a trainer by a configuration"""
import os
import sys
import shutil
import datetime
import traceback

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.radam import RAdam
# from torch.optim import Adam as RAdam
# from torch.optim import SGD as RAdam
from utils.center_loss import CenterLoss
from utils.metrics.segment_metrics import eval_metrics
from utils.metrics.metrics import accuracy
from utils.generals import make_batch
import torch
import torch.nn as nn
from torch.autograd.function import Function

EMO_DICT = {0: "ne", 1: "an", 2: "di", 3: "fe", 4: "ha", 5: "sa", 6: "su"}

class SparseCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(SparseCenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        self.sparse_centerloss = SparseCenterLossFunction.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.centers.data.t())

    def forward(self, feat, A, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.sparse_centerloss(feat, A, label, self.centers, batch_size_tensor)
        return loss

class SparseCenterLossFunction(Function):
    @staticmethod
    def forward(ctx, feature, A, label, centers, batch_size):
        ctx.save_for_backward(feature, A, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (A * (feature - centers_batch).pow(2)).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, A, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = feature - centers_batch
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        # A gradient
        grad_A = diff.pow(2) / 2.0 / batch_size

        counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), - A * diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return grad_output * A * diff / batch_size, grad_output * grad_A, None, grad_centers, None

class Trainer(object):
    """base class for trainers"""

    def __init__(self):
        pass

class FER2013Trainer(Trainer):
    """for classification task"""

    def __init__(self, model, train_set, val_set, test_set, configs):
        super().__init__()
        print("Start trainer..")
        print(configs)

        # load config
        self._configs = configs
        self._lr = self._configs["lr"]
        self._batch_size = self._configs["batch_size"]
        self._momentum = self._configs["momentum"]
        self._weight_decay = self._configs["weight_decay"]
        self._distributed = self._configs["distributed"]
        self._num_workers = self._configs["num_workers"]
        self._device = torch.device(self._configs["device"])
        self._max_epoch_num = self._configs["max_epoch_num"]
        self._max_plateau_count = self._configs["max_plateau_count"]

        # load dataloader and model
        self._train_set = train_set
        self._val_set = val_set
        self._test_set = test_set
        self._model = model(
            in_channels=configs["in_channels"],
            num_classes=configs["num_classes"],
        )

        # self._model.fc = nn.Linear(512, 7)
        # self._model.fc = nn.Linear(2, 7)
        # self._model.fc = nn.Linear(256, 7)
        self._model = self._model.to(self._device)

        if self._distributed == 1:
            torch.distributed.init_process_group(backend="nccl")
            self._model = nn.parallel.DistributedDataParallel(
                self._model, find_unused_parameters=True
            )
            self._train_loader = DataLoader(
                self._train_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=True,
                worker_init_fn=lambda x: np.random.seed(x),
            )
            self._val_loader = DataLoader(
                self._val_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=lambda x: np.random.seed(x),
            )

            self._test_loader = DataLoader(
                self._test_set,
                batch_size=1,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=lambda x: np.random.seed(x),
            )
        else:
            self._train_loader = DataLoader(
                self._train_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=True,
            )
            self._val_loader = DataLoader(
                self._val_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
            )
            self._test_loader = DataLoader(
                self._test_set,
                batch_size=1,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
            )

        # define loss function (criterion) and optimizer
        class_weights = [
            1.02660468,
            9.40661861,
            1.00104606,
            0.56843877,
            0.84912748,
            1.29337298,
            0.82603942,
        ]
        class_weights = torch.FloatTensor(np.array(class_weights))
        if self._configs["weighted_loss"] == 0:
            self._criterion = nn.CrossEntropyLoss().to(self._device)
        else:
            self._criterion = nn.CrossEntropyLoss(class_weights).to(self._device)

        self._optimizer = RAdam(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            # params=self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,


        )

        # for center loss
        # self._criterion_cent = CenterLoss(num_classes=7, feat_dim=2, use_gpu=True)
        self._criterion_cent =SparseCenterLoss(7, 2048).to(self._device)
        #self._optimizer_cent = RAdam(
        #    self._criterion_cent.parameters(), lr=self._lr
        #)
        # DACL used this: 
        #torch.optim.SGD(criterion['center'].parameters(), cfg['alpha']) alpha=?
        self._optimizer_cent = torch.optim.SGD(
            self._criterion_cent.parameters()
        )


        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            patience=self._configs['plateau_patience'],
            # min_lr=1e-10,
            verbose=True
        )
        #not same
        self._center_scheduler =ReduceLROnPlateau(
            self._optimizer_cent,
            patience=self._configs['plateau_patience'],
            # min_lr=1e-10,
            verbose=True
        )

        # # ''' TODO set step size equal to configs
        # self._scheduler = StepLR(self._optimizer, step_size=self._configs["steplr"])
        # self._center_scheduler = StepLR(
        #     self._optimizer_cent, step_size=self._configs["csteplr"]
        # )
        # # '''

        # training info
        self._start_time = datetime.datetime.now()
        self._start_time = self._start_time.replace(microsecond=0)

        log_dir = os.path.join(
            self._configs["cwd"],
            self._configs["log_dir"],
            "{}_{}_{}".format(
                self._configs["arch"],
                self._configs["model_name"],
                self._start_time.strftime("%Y%b%d_%H.%M"),
            ),
        )
        self._writer = SummaryWriter(log_dir)
        self._train_loss_list = []
        self._train_acc_list = []
        self._val_loss_list = []
        self._val_acc_list = []
        self._best_val_loss = 1e9
        self._best_val_acc = 0
        self._best_train_loss = 1e9
        self._best_train_acc = 0
        self._test_acc = 0.0
        self._plateau_count = 0
        self._current_epoch_num = 0

        # for checkpoints
        self._checkpoint_dir = os.path.join(self._configs["cwd"], "saved/checkpoints")
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._checkpoint_path = os.path.join(
            self._checkpoint_dir,
            "{}_{}_{}".format(
                self._configs["arch"],
                self._configs["model_name"],
                self._start_time.strftime("%Y%b%d_%H.%M"),
            ),
        )

    def _train(self):
        self._model.train()
        train_loss = 0.0
        train_acc = 0.0

        # for plot center lloss
        # all_features, all_labels = [], []

        for i, (images, targets) in tqdm(
            enumerate(self._train_loader), total=len(self._train_loader), leave=False
        ):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # compute output, measure accuracy and record loss
            features, outputs, A = self._model(images)
            loss = self._criterion(outputs, targets)
            loss_cent = self._criterion_cent(features, A, targets)
            # loss_cent *= self._configs["cweight"]
            loss = loss + 0.01 * loss_cent

            acc = accuracy(outputs, targets)[0]

            train_loss += loss.item()
            train_acc += acc.item()

            # compute gradient and do SGD step
            self._optimizer.zero_grad()
            self._optimizer_cent.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._optimizer_cent.step()

            # if self._current_epoch_num > 20:
                # by doing so, weight_cent would not impact on the learning of centers
                # for param in self._criterion_cent.parameters():
                #     param.grad.data *= 1.0 / self._configs["cweight"]
                # self._optimizer_cent.step()

            # all_features.append(features.data.cpu().numpy())
            # all_labels.append(targets.data.cpu().numpy())

        i += 1
        self._train_loss_list.append(train_loss / i)
        self._train_acc_list.append(train_acc / i)

        # plot center
        # all_features = np.concatenate(all_features, 0)
        # all_labels = np.concatenate(all_labels, 0)
        # self._plot_features(all_features, all_labels, prefix="train")

    def _val(self):
        self._model.eval()
        val_loss = 0.0
        val_acc = 0.0
        # for plot center lloss
        # all_features, all_labels = [], []

        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._val_loader), total=len(self._val_loader), leave=False
            ):
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                # compute output, measure accuracy and record loss
                # outputs, features = self._model(images)
                features, outputs, A = self._model(images)

                loss = self._criterion(outputs, targets)
                loss_cent = self._criterion_cent(features, A, targets)
                loss = loss + 0.01 * loss_cent
                acc = accuracy(outputs, targets)[0]

                val_loss += loss.item()
                val_acc += acc.item()

                # all_features.append(features.data.cpu().numpy())
                # all_labels.append(targets.data.cpu().numpy())

            i += 1
            self._val_loss_list.append(val_loss / i)
            self._val_acc_list.append(val_acc / i)

        # plot center
        # all_features = np.concatenate(all_features, 0)
        # all_labels = np.concatenate(all_labels, 0)
        # self._plot_features(all_features, all_labels, prefix="test")

    def _calc_acc_on_private_test(self):
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test..")
        f = open("private_test_log.txt", "w")
        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._test_loader), total=len(self._test_loader), leave=False
            ):

                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                # outputs = self._model(images)
                features, outputs, A = self._model(images)
                print(outputs.shape, outputs)
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()
                f.writelines("{}_{}\n".format(i, acc.item()))

            test_acc = test_acc / (i + 1)
        print("Accuracy on private test: {:.3f}".format(test_acc))
        f.close()
        return test_acc

    def _calc_acc_on_private_test_with_tta(self):
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test..")
        f = open(
            "private_test_log_{}_{}.txt".format(
                self._configs["arch"], self._configs["model_name"]
            ),
            "w",
        )

        with torch.no_grad():
            for idx in tqdm(
                range(len(self._test_set)), total=len(self._test_set), leave=False
            ):
                images, targets = self._test_set[idx]
                targets = torch.LongTensor([targets])

                images = make_batch(images)
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                # outputs, _ = self._model(images)
                features, outputs, A = self._model(images)
                outputs = F.softmax(outputs, 1)

                # outputs.shape [tta_size, 7]
                outputs = torch.sum(outputs, 0)

                outputs = torch.unsqueeze(outputs, 0)
                # print(outputs.shape)
                # TODO: try with softmax first and see the change
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()
                f.writelines("{}_{}\n".format(idx, acc.item()))

            test_acc = test_acc / (idx + 1)
        print("Accuracy on private test: {:.3f}".format(test_acc))
        f.close()
        return test_acc

    def train(self):
        """make a training job"""
        # print(self._model)

        try:
            while not self._is_stop():
                self._increase_epoch_num()
                self._train()
                self._val()

                self._update_training_state()
                self._logging()
        except KeyboardInterrupt:
            traceback.print_exc()
            pass

        # training stop
        try:
            # state = torch.load('saved/checkpoints/resatt18_rot30_2019Nov06_18.56')
            state = torch.load(self._checkpoint_path)
            if self._distributed:
                self._model.module.load_state_dict(state["net"])
            else:
                self._model.load_state_dict(state["net"])

            if not self._test_set.is_tta():
                self._test_acc = self._calc_acc_on_private_test()
            else:
                self._test_acc = self._calc_acc_on_private_test_with_tta()
            print(self._test_acc)
            self._save_weights()
        except Exception as e:
            traceback.print_exc()
            pass

        consume_time = str(datetime.datetime.now() - self._start_time)
        self._writer.add_text(
            "Summary",
            "Converged after {} epochs, consume {}".format(
                self._current_epoch_num, consume_time[:-7]
            ),
        )
        self._writer.add_text(
            "Results", "Best validation accuracy: {:.3f}".format(self._best_val_acc)
        )
        self._writer.add_text(
            "Results", "Best training accuracy: {:.3f}".format(self._best_train_acc)
        )
        self._writer.add_text(
            "Results", "Private test accuracy: {:.3f}".format(self._test_acc)
        )
        self._writer.close()

    def _update_training_state(self):
        if self._val_acc_list[-1] > self._best_val_acc:
            self._save_weights()
            self._plateau_count = 0
            self._best_val_acc = self._val_acc_list[-1]
            self._best_val_loss = self._val_loss_list[-1]
            self._best_train_acc = self._train_acc_list[-1]
            self._best_train_loss = self._train_loss_list[-1]
        else:
            self._plateau_count += 1

        # self._scheduler.step(self._train_loss_list[-1])
        self._scheduler.step(100 - self._val_acc_list[-1])
        # self._center_scheduler.step()

    def _logging(self):
        consume_time = str(datetime.datetime.now() - self._start_time)

        message = "\nE{:03d}  {:.3f}/{:.3f}/{:.3f} {:.3f}/{:.3f}/{:.3f} | p{:02d}  Time {}\n".format(
            self._current_epoch_num,
            self._train_loss_list[-1],
            self._val_loss_list[-1],
            self._best_val_loss,
            self._train_acc_list[-1],
            self._val_acc_list[-1],
            self._best_val_acc,
            self._plateau_count,
            consume_time[:-7],
        )

        self._writer.add_scalar(
            "Accuracy/Train", self._train_acc_list[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Accuracy/Val", self._val_acc_list[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Loss/Train", self._train_loss_list[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Loss/Val", self._val_loss_list[-1], self._current_epoch_num
        )

        print(message)

    def _is_stop(self):
        """check stop condition"""
        return (
            # self._plateau_count > self._max_plateau_count or
             self._current_epoch_num > self._max_epoch_num
        )

    def _increase_epoch_num(self):
        self._current_epoch_num += 1

    def _save_weights(self, test_acc=0.0):
        if self._distributed == 0:
            state_dict = self._model.state_dict()
        else:
            state_dict = self._model.module.state_dict()

        state = {
            **self._configs,
            "net": state_dict,
            "best_val_loss": self._best_val_loss,
            "best_val_acc": self._best_val_acc,
            "best_train_loss": self._best_train_loss,
            "best_train_acc": self._best_train_acc,
            "train_losses": self._train_loss_list,
            "val_loss_list": self._val_loss_list,
            "train_acc_list": self._train_acc_list,
            "val_acc_list": self._val_acc_list,
            "test_acc": self._test_acc,
        }

        torch.save(state, self._checkpoint_path)

    # def _plot_features(self, features, labels, prefix):
    #     """Plot features on 2D plane.
    #     Args:
    #         features: (num_instances, num_features).
    #         labels: (num_instances).
    #     """
    #     colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    #     for label_idx in range(7):
    #         plt.scatter(
    #             features[labels == label_idx, 0],
    #             features[labels == label_idx, 1],
    #             c=colors[label_idx],
    #             s=1,
    #         )
    #     plt.legend(["0", "1", "2", "3", "4", "5", "6"], loc="upper right")
    #     plt_dirname = os.path.join(
    #         "saved/plot/{}".format(os.path.basename(self._checkpoint_path))
    #     )
    #     if not os.path.exists(plt_dirname):
    #         os.makedirs(plt_dirname, exist_ok=True)
    #
    #     save_name = os.path.join(
    #         plt_dirname, "epoch_{}_{}.png".format(self._current_epoch_num, prefix)
    #     )
    #     plt.savefig(save_name, bbox_inches="tight")
    #     plt.close()
