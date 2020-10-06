import argparse
import logging
import os
import sys
import torch
from glob import glob
from pathlib import Path
from typing import Any, Dict
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import ModelCheckpoin
from pytorch_lightning.utilities import rank_zero_info
from transformers import BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torchnlp import get_accuracy
from model import VIT

class VIT_Model(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        **config_kwargs
    ):
        super().__init__()
        self.hparams = hparams
        # save all variables in __init__ signature to self.hparams
        self.tfmr_ckpts = {}
        self.config = BertConfig.from_json_file(self.hparams.config)
        self.model = VIT(self.config)
        self.output_dir = Path(self.hparams.output_dir)
        self.current_loss = 0.0
        self.best_loss = sys.maxsize
        self.transform = transforms.Compose([transforms.CenterCrop(self.confing.image_size),
                                transforms.ToTensor()])
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        dataset = torchvision.datasets.ImageFolder(self.hpamras.data_path, transform=self.transforms)
        train_cnt = int(len(dataset) * self.hparams.train_ratio)
        val_cnt = len(dataset) - train_cnt
        self.train_set, self.val_set = torch.utils.data.random_split(dataset, [train_cnt, val_cnt])

    def forward(
        self,
        input_ids=None,
        output_attention=False
    ):
        out = self.model(input_ids=input_ids, output_attentions=output_attention)
        logits = out[1]
        return logits
    
    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        # model = self
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps
        )
        self.lr_scheduler = scheduler
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        train_loader = DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, drop_last=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, drop_last=True)
        return val_loader

    def training_step(self, batch, batch_idx):
        input_ids, label_cls = batch

        logits = self(input_ids=input_ids)
        loss_fct = nn.CrossEntropyLoss()

        total_loss = loss_fct(
            logits.view(-1, self.config.num_classes), logits.view(-1))
        
        accuracy = get_accuracy(label_cls.reshape(-1), logits.argmax(-1).reshape(-1))[0]
        
        self.current_loss = total_loss

        tensorboard_logs = {'train_loss': total_loss, 
                            "train_accuracy": accuracy,
                            'learning_rate':  self.lr_scheduler.get_last_lr()[-1]}

        return {'loss': total_loss, 'progress_bar':{'acc': accuracy}, 'log': tensorboard_logs,}
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        input_ids, label_cls = batch

        logits = self(input_ids=input_ids)
        loss_fct = nn.CrossEntropyLoss()

        total_loss = loss_fct(
            logits.view(-1, self.config.num_classes), logits.view(-1))
        
        accuracy = get_accuracy(label_cls.reshape(-1), logits.argmax(-1).reshape(-1))[0]
        
        self.current_loss = total_loss

        return {'val_loss': total_loss, 
                "val_accuracy": accuracy,
                }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()

        tensorboard_logs = {
            "val/loss": avg_loss,
            "val/accuracy": avg_acc,
        }

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }


    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.current_loss <= self.best_loss:
            save_path = self.output_dir.joinpath("best_tfmr")
            print("save best model at step :{} current_loss:{} best_loss:{}".format(self.global_step, self.current_loss, self.best_loss))
            save_path.mkdir(exist_ok=True)
            self.model.config.save_step = self.global_step
            self.model.save_pretrained(save_path)
            self.tfmr_ckpts[self.global_step] = save_path
            self.best_loss = self.current_loss
    
    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--data_root",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )

        parser.add_argument(
            "--config",
            default=None,
            type=str,
            required=True,
            help="model config path",
        )

        parser.add_argument(
            "--output_dir",
            default=os.getcwd(),
            type=str,
            required=False,
            help="model config path",
        )

        parser.add_argument("--gradient_clip_val", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
        parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=10000, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--total_steps", default=10000000, type=int, help="Linear warmup over total steps.")
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=20, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        return parser



class LoggingCallback(pl.Callback):

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))
                    
class SaveIterCallback(pl.Callback):
    def __init__(self, iter_num=200):
        self.iter_num = iter_num
        self.previous_name = ''
    
    @pl.utilities.rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        
        if pl_module.global_step % self.iter_num == 0:
            file_path = os.path.join(pl_module.output_dir, 'checkpoint_iter_{}.ckpt'.format(pl_module.global_step))
            trainer.checkpoint_callback._del_model(self.previous_name)
            trainer.checkpoint_callback._save_model(file_path)
            self.previous_name = file_path

                    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')                    


def add_train_parser(parser):
    parser.add_argument("--early_stopping_callback", default=False, type=bool, help="overwrite model binary to latest model",)
    parser.add_argument("--logger", default=True, type=str2bool, help ="user logger")
    parser.add_argument("--save_mode", default='min', type=str, help="the decision value for saving model",)
    parser.add_argument("--save_last", default=False, type=str2bool, help="overwrite model binary to latest model")
    parser.add_argument("--reload_dataloaders_every_epoch", default=True, type=str2bool, help="reset train dataloader")
    return parser


def generic_train(
    model ,
    args: argparse.Namespace,
    extra_callbacks=[],
    **extra_train_kwargs
):
    
    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=model.hparams.output_dir, prefix="checkpoint", monitor="loss", mode=args.save_mode, save_top_k=1,
            save_last=args.save_last
        )

    logging_callback = LoggingCallback()
    lr_logger = LearningRateLogger()
    train_params = {}

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"


    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=[logging_callback, lr_logger] + extra_callbacks,
        logger=args.logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=args.early_stopping_callback,
        **train_params,
    )

    trainer.fit(model)

    return trainer