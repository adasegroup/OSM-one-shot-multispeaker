from ..models.wavernn import WaveRNN
from ..data.dataset import VocoderDataset, VocoderDataloader
from .distribution import discretized_mix_logistic_loss
from .gen_wavernn import gen_testset
from torch.utils.data import DataLoader
from pathlib import Path
from torch import optim
import torch.nn.functional as F
import numpy as np
import time
import torch
import os


class VocoderTrainer:
    def __init__(self, config, model, optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.device = config.DEVICE
        if self.config.MODEL.CHECKPOINT_NAME is not None:
            self.checkpoint_path = os.path.join(self.config.MODEL.CHECKPOINT_DIR_PATH,
                                                self.config.MODEL.CHECKPOINT_NAME)

        self.syn_dir = self.config.DATA.SYN_DIR
        self.voc_dir = self.config.DATA.VOC_DIR
        self.models_dir = self.config.MODEL.CHECKPOINT_DIR_PATH
        self.ground_truth = self.config.MODEL.GROUND_TRUTH
        self.save_every = self.config.MODEL.SAVE_EVERY

    def train(self, run_id: str, backup_every: int, force_restart: bool):
        for p in self.optimizer.param_groups:
            p["lr"] = self.config.MODEL.LR = 1e-4
        loss_func = F.cross_entropy if self.model.mode == "RAW" else discretized_mix_logistic_loss

        # Load the weights
        model_dir = self.models_dir.joinpath(run_id)
        model_dir.mkdir(exist_ok=True)
        weights_fpath = model_dir.joinpath(run_id + ".pt")
        if force_restart or not weights_fpath.exists():
            print("\nStarting the training of WaveRNN from scratch\n")
            self.model.save(weights_fpath, self.optimizer)
        else:
            print("\nLoading weights at %s" % weights_fpath)
            self.model.load(weights_fpath, self.optimizer)
            print("WaveRNN weights loaded from step %d" % self.model.step)

        # Initialize the dataset
        metadata_fpath = self.syn_dir.joinpath("train.txt") if self.ground_truth else \
            self.voc_dir.joinpath("synthesized.txt")
        mel_dir = self.syn_dir.joinpath("mels") if self.ground_truth else self.voc_dir.joinpath("mels_gta")
        wav_dir = self.syn_dir.joinpath("audio")
        dataset = VocoderDataset(self.config, metadata_fpath, mel_dir, wav_dir)
        test_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 pin_memory=True)

        # Begin the training
        for epoch in range(1, 350):
            data_loader = VocoderDataloader(self.config, dataset)
            start = time.time()
            running_loss = 0.

            for i, (x, y, m) in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    x, m, y = x.cuda(), m.cuda(), y.cuda()

                # Forward pass
                y_hat = self.model(x, m)
                if self.model.mode == 'RAW':
                    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                elif self.model.mode == 'MOL':
                    y = y.float()
                y = y.unsqueeze(-1)

                # Backward pass
                loss = loss_func(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                speed = i / (time.time() - start)
                avg_loss = running_loss / i

                step = self.model.get_step()
                k = step // 1000

                if backup_every != 0 and step % backup_every == 0:
                    self.model.checkpoint(model_dir, self.optimizer)

                if self.save_every != 0 and step % self.save_every == 0:
                    self.model.save(weights_fpath, self.optimizer)

                msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                      f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                      f"steps/s | Step: {k}k | "
                print(msg)

            gen_testset(self.model, test_loader, self.config.MODEL.GEN_AT_CHECKPOINT, self.config.MODEL.GEN_BATCHED,
                        self.config.MODEL.TARGET, self.config.MODEL.OVERLAP, model_dir)
            print("")