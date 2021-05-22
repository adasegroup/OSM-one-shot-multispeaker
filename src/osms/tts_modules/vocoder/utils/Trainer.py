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
    def __init__(self, config, model, train_dataloader, test_loader, optimizer):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = config.DEVICE
        if self.config.MODEL.CHECKPOINT_NAME is not None:
            self.checkpoint_path = os.path.join(self.config.MODEL.CHECKPOINT_DIR_PATH,
                                                self.config.MODEL.CHECKPOINT_NAME)



        self.syn_dir =
        self.voc_dir =
        self.model_dirs =
        self.ground_truth =
        self.save_every =

        self.step = 1
        self.save_n_steps = self.config.TRAIN.SAVE_N_STEPS
        self.out_dir = self.config.TRAIN.OUT_DIR

        self.out_checkpoints_dir = self.config.TRAIN.OUT_DIR
        self.run_id = self.config.TRAIN.RUN_ID





    def train(self, run_id: str, syn_dir: Path, voc_dir: Path, models_dir: Path, ground_truth: bool,
          save_every: int, backup_every: int, force_restart: bool):
        # Check to make sure the hop length is correctly factorised
        assert np.cumprod(self.config.MODEL.UPSAMPLE_FACTORS)[-1] == self.config.SP.HOP_SIZE

        # Instantiate the model
        print("Initializing the model...")
        model = WaveRNN(
            rnn_dims=self.config.MODEL.RNN_DIMS,
            fc_dims=self.config.MODEL.FC_DIMS,
            bits=self.config.SP.BITS,
            pad=self.config.MODEL.PAD,
            upsample_factors=self.config.MODEL.UPSAMPLE_FACTORS,
            feat_dims=self.config.SP.NUM_MELS,
            compute_dims=self.config.MODEL.COMPUTE_DIMS,
            res_out_dims=self.config.MODEL.RES_OUT_DIMS,
            res_blocks=self.config.MODEL.RES_BLOCKS,
            hop_length=self.config.SP.HOP_SIZE,
            sample_rate=self.config.SP.SAMPLE_RATE,
            mode=self.config.MODEL.MODE
        )

        if torch.cuda.is_available():
            model = model.cuda()
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

            # Initialize the optimizer
        optimizer = optim.Adam(model.parameters())
        for p in optimizer.param_groups:
            p["lr"] = self.config.MODEL.LR = 1e-4
        loss_func = F.cross_entropy if model.mode == "RAW" else discretized_mix_logistic_loss

        # Load the weights
        model_dir = models_dir.joinpath(run_id)
        model_dir.mkdir(exist_ok=True)
        weights_fpath = model_dir.joinpath(run_id + ".pt")
        if force_restart or not weights_fpath.exists():
            print("\nStarting the training of WaveRNN from scratch\n")
            model.save(weights_fpath, optimizer)
        else:
            print("\nLoading weights at %s" % weights_fpath)
            model.load(weights_fpath, optimizer)
            print("WaveRNN weights loaded from step %d" % model.step)

        # Initialize the dataset
        metadata_fpath = syn_dir.joinpath("train.txt") if ground_truth else \
            voc_dir.joinpath("synthesized.txt")
        mel_dir = syn_dir.joinpath("mels") if ground_truth else voc_dir.joinpath("mels_gta")
        wav_dir = syn_dir.joinpath("audio")
        dataset = VocoderDataset(metadata_fpath, mel_dir, wav_dir)
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
                y_hat = model(x, m)
                if model.mode == 'RAW':
                    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                elif model.mode == 'MOL':
                    y = y.float()
                y = y.unsqueeze(-1)

                # Backward pass
                loss = loss_func(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                speed = i / (time.time() - start)
                avg_loss = running_loss / i

                step = model.get_step()
                k = step // 1000

                if backup_every != 0 and step % backup_every == 0:
                    model.checkpoint(model_dir, optimizer)

                if save_every != 0 and step % save_every == 0:
                    model.save(weights_fpath, optimizer)

                msg = f"| Epoch: {epoch} ({i}/{len(data_loader)}) | " \
                      f"Loss: {avg_loss:.4f} | {speed:.1f} " \
                      f"steps/s | Step: {k}k | "
                print(msg)

            gen_testset(model, test_loader, self.config.MODEL.GEN_AT_CHECKPOINT, self.config.MODEL.GEN_BATCHED,
                        self.config.MODEL.TARGET, self.config.MODEL.OVERLAP, model_dir)
            print("")