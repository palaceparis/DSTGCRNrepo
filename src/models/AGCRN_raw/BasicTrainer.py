import torch
import math
import os
import time
import copy
import numpy as np

from lib.logger import get_logger
from lib.metrics import All_Metrics
from utils import evaluate_metrics


class Trainer(object):
    def __init__(
        self,
        model,
        loss,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        scaler,
        args,
        lr_scheduler=None,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, "best_model.pth")
        self.loss_figure_path = os.path.join(self.args.log_dir, "loss.png")
        # log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info("Experiment log path in: {}".format(args.log_dir))
        # if not args.debug:
        # self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., : self.args.input_dim]
                label = target[..., : self.args.output_dim]
                output = self.model(data, target, teacher_forcing_ratio=0.0)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)
                loss = self.loss(output.cpu(), label.cpu())
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                # collect all labels and predictions for metric evaluation
                all_labels.append(label.cpu().numpy())
                all_predictions.append(output.cpu().numpy())

        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info(
            "**********Val Epoch {}: average Loss: {:.6f}".format(epoch, val_loss)
        )

        # concatenate all labels and predictions
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        all_labels = all_labels.reshape(all_labels.shape[0], -1)
        all_predictions = all_predictions.reshape(all_predictions.shape[0], -1)

        # evaluate metrics
        metrics = evaluate_metrics(all_labels, all_predictions)
        for metric, value in metrics.items():
            self.logger.info(f"Validation {metric}: {value}")

        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., : self.args.input_dim]
            label = target[..., : self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            # teacher_forcing for RNN encoder-decoder model
            # if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(
                    global_step, self.args.tf_decay_steps
                )
            else:
                teacher_forcing_ratio = 1.0
            # data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(
                data, target, teacher_forcing_ratio=teacher_forcing_ratio
            )
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
            loss = self.loss(output.cpu(), label.cpu())
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
            self.optimizer.step()
            total_loss += loss.item()

            # log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info(
                    "Train Epoch {}: {}/{} Loss: {:.6f}".format(
                        epoch, batch_idx, self.train_per_epoch, loss.item()
                    )
                )
        train_epoch_loss = total_loss / self.train_per_epoch
        train_epoch_rmse = math.sqrt(train_epoch_loss)
        self.logger.info(
            "**********Train Epoch {}: averaged Loss: {:.6f}, Traing Loss: {:.4f}, RMSE: {:.6f}, tf_ratio: {:.6f}".format(
                epoch,
                train_epoch_loss,
                total_loss,
                train_epoch_rmse,
                teacher_forcing_ratio,
            )
        )

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss, train_epoch_rmse

    def train(self):
        best_model = None
        best_train_rmse = float("inf")
        best_loss = float("inf")
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            # epoch_time = time.time()
            train_epoch_loss, train_epoch_rmse = self.train_epoch(epoch)
            # print(time.time()-epoch_time)
            # exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            # print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break
            # if self.val_loader == None:
            # val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_train_rmse = train_epoch_rmse  # Add this line
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.args.early_stop_patience)
                    )
                    break
            # save the best state
            if best_state == True:
                self.logger.info(
                    "*********************************Current best model saved!"
                )
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info(
            "Total training time: {:.4f}min, best training RMSE: {:.6f}, best loss: {:.6f}".format(
                (training_time / 60), best_train_rmse, best_loss
            )
        )

        # save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.args,
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point["state_dict"]
            args = check_point["config"]
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., : args.input_dim]
                label = target[..., : args.output_dim]
                output = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label.cpu().numpy())  # added cpu().numpy()
                y_pred.append(output.cpu().numpy())  # added cpu().numpy()
        y_true = scaler.inverse_transform(np.concatenate(y_true, axis=0))
        if args.real_value:
            y_pred = np.concatenate(y_pred, axis=0)
        else:
            # y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_pred = scaler.inverse_transform(np.concatenate(y_pred, axis=0))

        # np.save("./{}_true.npy".format(args.dataset), y_true.cpu().numpy())
        # np.save("./{}_pred.npy".format(args.dataset), y_pred.cpu().numpy())

        # Evaluate metrics - previous version
        squared_errors = (y_pred - y_true) ** 2
        mean_squared_errors = squared_errors.mean()
        rmse_man = np.sqrt(mean_squared_errors)
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(
                y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh
            )
            logger.info(
                "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    t + 1, mae, rmse, mape * 100
                )
            )
        mae, rmse, mape, _, _ = All_Metrics(
            y_pred, y_true, args.mae_thresh, args.mape_thresh
        )
        logger.info(
            "Average Horizon, MAE: {:.2f}, RMSE: {:.6f}, RMSE-MAN: {:.6f}, MAPE: {:.4f}%".format(
                mae, rmse, rmse_man, mape * 100
            )
        )
        # y_true = y_true.reshape(y_true.shape[0], -1)
        # y_pred = y_pred.reshape(y_pred.shape[0], -1)
        # # Evaluate metrics
        # metrics = evaluate_metrics(y_true, y_pred)
        # for metric, value in metrics.items():
        #     logger.info(f"Test_New {metric}: {value}")

        # Evaluate metrics
        avg_metrics = {
            "MAE": 0,
            "MAPE": 0,
            "RMSE": 0,
            "RMSPE": 0,
            "R-squared": 0,
            "Adjusted R-squared": 0,
        }

        for t in range(y_pred.shape[1]):
            metrics = evaluate_metrics(
                y_true[:, t, ...].reshape(y_pred.shape[0], -1),
                y_pred[:, t, ...].reshape(y_pred.shape[0], -1),
            )
            for metric, value in metrics.items():
                avg_metrics[
                    metric
                ] += value  # Add the value to the accumulated metric values
                logger.info(f"Test_New {metric}: {value}")

        # Divide the accumulated metric values by the number of days to get the average
        for metric, value in avg_metrics.items():
            avg_metrics[metric] /= y_pred.shape[1]

        for metric, value in avg_metrics.items():
            logger.info(f"Test_New_Average {metric}: {value}")

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
