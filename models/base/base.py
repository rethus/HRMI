import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from tqdm import tqdm

from root import absolute
from utils.evaluation import mae, mse, rmse
from utils.model_util import load_checkpoint, save_checkpoint
from utils.mylogger import myLog

from .utils import train_single_epoch_with_dataloader, train_mult_epochs_with_dataloader


class ModelBase(object):
    def __init__(self, loss_fn, use_gpu=True) -> None:
        super().__init__()
        self.loss_fn = loss_fn  # 损失函数
        self.optimizer = None
        self.device = ("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.name = self.__class__.__name__
        self.logger = myLog(self.name)  # 日志
        self.logger.initial_logger()

        # 获取当前时间
        self.date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        # 保存训练过程中的模型参数，用于预测使用
        self.saved_model_ckpt = []

        # Tensorboard
        # 自动打开tensorboard，只要浏览器中打开localhost:6006即可看到训练过程
        save_dir = absolute(f"output/{self.name}/{self.date}/TensorBoard")
        os.makedirs(save_dir)
        self.writer = SummaryWriter(log_dir=save_dir)
        # tensorboard = program.TensorBoard()
        # tensorboard.configure(argv=[None, '--logdir', save_dir])
        # tensorboard.launch()

    def fit(self, train_loader, epochs, optimizer, eval_=True, eval_loader=None, save_model=True, save_filename=""):
        """Eval为True: 自动保存最优模型（推荐）, save_model为True: 间隔epoch后自动保存模型

        Args:
            train_loader : 训练集
            epochs : 迭代次数
            optimizer : 优化器
            eval_ : 训练过程中是否需要验证 Defaults to True.
            eval_loader : 验证集数据 Defaults to None. 
            save_model :  是否保存模型 Defaults to True.
            save_filename :  保存的模型的名字 Defaults to "".
        """
        self.model.train()
        self.model.to(self.device)
        train_loss_list = []
        eval_loss_list = []
        best_loss = None
        self.optimizer = optimizer

        # 训练
        for epoch in tqdm(range(epochs), position=0, leave=True):
            train_batch_loss = 0
            eval_total_loss = 0

            train_batch_mae = 0
            train_batch_rmse = 0

            for batch_id, batch in enumerate(train_loader):

                users, items, ratings = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                y_real = ratings.reshape(-1, 1)
                self.optimizer.zero_grad()
                # y_pred = self.model(users, items)
                y_pred, u_attr, s_attr, u_vec, s_vec = self.model(users, items)
                # must be (1. nn output, 2. target)
                loss = self.loss_fn(y_pred, y_real, u_attr, s_attr, u_vec, s_vec)
                loss.backward()
                self.optimizer.step()

                train_batch_loss += loss.item()

            loss_per_epoch = train_batch_loss / len(train_loader)
            train_loss_list.append(loss_per_epoch)

            self.logger.info(f"Training Epoch:[{epoch + 1}/{epochs}] Loss:{loss_per_epoch:.8f}")
            self.writer.add_scalar("Training Loss", loss_per_epoch, epoch + 1)

            # 验证
            if (epoch + 1) % 10 == 0:
                if eval_ == True:
                    assert eval_loader is not None, "Please offer eval dataloader"
                    self.model.eval()
                    with torch.no_grad():
                        for batch_id, batch in tqdm(enumerate(eval_loader), position=0, leave=True):
                            user, item, rating = batch[0].to(self.device), \
                                                 batch[1].to(self.device), \
                                                 batch[2].to(self.device)
                            # y_pred = self.model(user, item)
                            y_pred, u_attr, s_attr, u_vec, s_vec = self.model(users, items)
                            # loss = self.loss_fn(y_pred, y_real)
                            loss = self.loss_fn(y_pred, y_real, u_attr, s_attr, u_vec, s_vec)
                            # loss = torch.mean(torch.abs(y_real - y_pred))

                            y_real = rating.reshape(-1, 1)
                            eval_total_loss += loss.item()
                        loss_per_epoch = eval_total_loss / len(eval_loader)

                        if best_loss is None:
                            best_loss = loss_per_epoch
                            is_best = True
                        elif loss_per_epoch < best_loss:
                            best_loss = loss_per_epoch
                            is_best = True
                        else:
                            is_best = False
                        eval_loss_list.append(loss_per_epoch)
                        self.logger.info(f"Test loss: {loss_per_epoch}")
                        self.writer.add_scalar("Eval loss", loss_per_epoch, epoch)
                        # 保存最优的loss
                        if is_best:
                            ckpt = {
                                "model": self.model.state_dict(),
                                "epoch": epoch + 1,
                                "optim": optimizer.state_dict(),
                                "best_loss": best_loss
                            }
                        # else:
                        #     ckpt = {}
                        save_checkpoint(ckpt, is_best, f"output/{self.name}/{self.date}/saved_model",
                                        f"{save_filename}_loss_{best_loss:.4f}.ckpt")
                        self.saved_model_ckpt.append(ckpt)

                elif save_model:
                    ckpt = {
                        "model": self.model.state_dict(),
                        "epoch": epoch + 1,
                        "optim": optimizer.state_dict(),
                        "best_loss": loss_per_epoch
                    }
                    save_checkpoint(ckpt, save_model, f"output/{self.name}/{self.date}/saved_model",
                                    f"{save_filename}_loss_{loss_per_epoch:.4f}.ckpt")
                    self.saved_model_ckpt.append(ckpt)

    def predict(self, test_loader, resume=False, path=None):
        """模型预测

        Args:
            test_loader : 测试数据
            resume: 是否加载预训练模型. Defaults to False.
            path: 预训练模型地址. Defaults to None.

        Returns:
            [type]: [description]
        """
        y_pred_list = []
        y_list = []
        # 加载预训练模型
        if resume:
            if path:
                ckpt = load_checkpoint(path)
            else:
                models = sorted(self.saved_model_ckpt, key=lambda x: x['best_loss'])
                ckpt = models[0]
            self.model.load_state_dict(ckpt['model'])
            self.logger.info(f"last checkpoint restored! ckpt: loss {ckpt['best_loss']:.4f} Epoch {ckpt['epoch']}")

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in tqdm(enumerate(test_loader), position=0, leave=True):
                user, item, rating = batch[0].to(self.device), batch[1].to(
                    self.device), batch[2].to(self.device)
                y_pred, a, b, c, d = self.model(user, item)
                y_pred = y_pred.squeeze()
                y_real = rating.reshape(-1, 1)
                if len(y_pred.shape) == 0:  # 防止因batch大小而变成标量,故增加一个维度
                    y_pred = y_pred.unsqueeze(dim=0)
                if len(y_real.shape) == 0:  # 防止因batch大小而变成标量,故增加一个维度
                    y_real = y_real.unsqueeze(dim=0)
                y_pred_list.append(y_pred)
                y_list.append(y_real)

        return torch.cat(y_list).cpu().numpy(), torch.cat(y_pred_list).cpu().numpy()


class MemoryBase(object):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
