import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import audio
from .data import SynthesizerDataset, collate_synthesizer
from .utils import ValueWindow, data_parallel_workaround
from .utils.symbols import symbols
from .utils.text import sequence_to_text
from .utils.plot import plot_spectrogram
from datetime import datetime
import numpy as np
from pathlib import Path
import time


class SynthesizerTrainer:
    """
    The class-manager used to train the Synthesizer model
    """

    def __init__(self, configs, model, optimizer):
        self.configs = configs
        self.model = model
        self.optimizer = optimizer
        self.syn_dir = self.configs.DATA.SYN_DIR
        self.models_dir = self.configs.MODEL.CHECKPOINT_DIR_PATH

    @staticmethod
    def np_now(x: torch.Tensor):
        return x.detach().cpu().numpy()

    @staticmethod
    def time_string():
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def train(self,
              run_id: str,
              save_every: int,
              backup_every: int
              ):
        """
        Main method to train the Synthesizer

        :param run_id: ID of the run
        :param save_every: Frequency of quick saving of checkpoint
        :param backup_every: Frequency of backuping the model and optimizer
        :return: None
        """
        self.model.train()
        self.syn_dir = Path(self.syn_dir)
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(exist_ok=True)

        model_dir = self.models_dir.joinpath(run_id)
        plot_dir = model_dir.joinpath("plots")
        wav_dir = model_dir.joinpath("wavs")
        mel_output_dir = model_dir.joinpath("mel-spectrograms")
        meta_folder = model_dir.joinpath("metas")
        model_dir.mkdir(exist_ok=True)
        plot_dir.mkdir(exist_ok=True)
        wav_dir.mkdir(exist_ok=True)
        mel_output_dir.mkdir(exist_ok=True)
        meta_folder.mkdir(exist_ok=True)

        weights_fpath = model_dir.joinpath(run_id).with_suffix(".pt")
        metadata_fpath = self.syn_dir.joinpath("train.txt")

        print("Checkpoint path: {}".format(weights_fpath))
        print("Loading training data from: {}".format(metadata_fpath))
        print("Using model: Tacotron")

        # Book keeping
        step = 0
        time_window = ValueWindow(100)
        loss_window = ValueWindow(100)

        # From WaveRNN/train_tacotron.py
        if torch.cuda.is_available():
            device = torch.device("cuda")

            for session in self.configs.MODEL.SCHEDULE:
                _, _, _, batch_size = session
                if batch_size % torch.cuda.device_count() != 0:
                    raise ValueError("`batch_size` must be evenly divisible by n_gpus!")
        else:
            device = torch.device("cpu")
        print("Using device:", device)

        self.model = self.model.to(device)

        # Initialize the dataset
        metadata_fpath = self.syn_dir.joinpath("train.txt")
        mel_dir = self.syn_dir.joinpath("mels")
        embed_dir = self.syn_dir.joinpath("embeds")
        dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, self.configs)
        test_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 pin_memory=True)

        for i, session in enumerate(self.configs.MODEL.SCHEDULE):
            current_step = self.model.get_step()

            r, lr, max_step, batch_size = session

            training_steps = max_step - current_step

            if current_step >= max_step:
                # Are there no further sessions than the current one?
                if i == len(self.configs.MODEL.SCHEDULE) - 1:
                    # We have completed training. Save the model and exit
                    self.model.save(weights_fpath, self.optimizer)
                    break
                else:
                    # There is a following session, go to it
                    continue

            self.model.r = r

            for p in self.optimizer.param_groups:
                p["lr"] = lr

            data_loader = DataLoader(dataset,
                                     collate_fn=lambda batch: collate_synthesizer(batch, r, self.configs),
                                     batch_size=batch_size,
                                     num_workers=2,
                                     shuffle=True,
                                     pin_memory=True)

            total_iters = len(dataset)
            steps_per_epoch = np.ceil(total_iters / batch_size).astype(np.int32)
            epochs = np.ceil(training_steps / steps_per_epoch).astype(np.int32)

            for epoch in range(1, epochs + 1):
                for i, (texts, mels, embeds, idx) in enumerate(data_loader, 1):
                    start_time = time.time()

                    # Generate stop tokens for training
                    stop = torch.ones(mels.shape[0], mels.shape[2])
                    for j, k in enumerate(idx):
                        stop[j, :int(dataset.metadata[k][4]) - 1] = 0

                    texts = texts.to(device)
                    mels = mels.to(device)
                    embeds = embeds.to(device)
                    stop = stop.to(device)

                    # Forward pass
                    # Parallelize model onto GPUS using workaround due to python bug
                    if device.type == "cuda" and torch.cuda.device_count() > 1:
                        m1_hat, m2_hat, attention, stop_pred = data_parallel_workaround(self.model,
                                                                                        texts,
                                                                                        mels,
                                                                                        embeds)
                    else:
                        m1_hat, m2_hat, attention, stop_pred = self.model(texts, mels, embeds)

                    # Backward pass
                    m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                    m2_loss = F.mse_loss(m2_hat, mels)
                    stop_loss = F.binary_cross_entropy(stop_pred, stop)

                    loss = m1_loss + m2_loss + stop_loss

                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.configs.MODEL.CLIP_GRAD_NORM is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                   self.configs.MODEL.CLIP_GRAD_NORM)
                        if np.isnan(grad_norm.cpu()):
                            print("grad_norm was NaN!")

                    self.optimizer.step()

                    time_window.append(time.time() - start_time)
                    loss_window.append(loss.item())

                    step = self.model.get_step()
                    k = step // 1000

                    # Backup or save model as appropriate
                    if backup_every != 0 and step % backup_every == 0:
                        backup_fpath = Path("{}/{}_{}k.pt".format(str(weights_fpath.parent), run_id, k))
                        self.model.save(backup_fpath, self.optimizer)

                    if save_every != 0 and step % save_every == 0:
                        # Must save latest optimizer state to ensure that resuming training
                        # doesn't produce artifacts
                        self.model.save(weights_fpath, self.optimizer)

                    # Evaluate model to generate samples
                    epoch_eval = self.configs.MODEL.EVAL_INTERVAL == -1 and i == steps_per_epoch  # If epoch is done
                    step_eval = self.configs.MODEL.EVAL_INTERVAL > 0 and step % self.configs.MODEL.EVAL_INTERVAL == 0
                    if epoch_eval or step_eval:
                        for sample_idx in range(self.configs.MODEL.EVAL_NUM_SAMPLES):
                            # At most, generate samples equal to number in the batch
                            if sample_idx + 1 <= len(texts):
                                # Remove padding from mels using frame length in metadata
                                mel_length = int(dataset.metadata[idx[sample_idx]][4])
                                mel_prediction = SynthesizerTrainer.np_now(m2_hat[sample_idx]).T[:mel_length]
                                target_spectrogram = SynthesizerTrainer.np_now(mels[sample_idx]).T[:mel_length]
                                attention_len = mel_length // self.model.r

                                self.eval_model(
                                    mel_prediction=mel_prediction,
                                    target_spectrogram=target_spectrogram,
                                    input_seq=SynthesizerTrainer.np_now(texts[sample_idx]),
                                    step=step,
                                    plot_dir=plot_dir,
                                    mel_output_dir=mel_output_dir,
                                    wav_dir=wav_dir,
                                    sample_num=sample_idx + 1,
                                    loss=loss
                                )

                    # Break out of loop to update training schedule
                    if step >= max_step:
                        break

                # Add line break after every epoch
                print("")
        return None

    def eval_model(self, mel_prediction, target_spectrogram, input_seq, step,
                   plot_dir, mel_output_dir, wav_dir, sample_num, loss):

        # save predicted mel spectrogram to disk (debug)
        mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
        np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

        # save griffin lim inverted wav for debug (mel -> wav)
        wav = audio.inv_mel_spectrogram(mel_prediction.T, self.configs)
        wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
        audio.save_wav(wav, str(wav_fpath), sr=self.configs.SP.SAMPLE_RATE)

        # save real and predicted mel-spectrogram plot to disk (control purposes)
        spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
        title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", SynthesizerTrainer.time_string(), step, loss)
        plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                         target_spectrogram=target_spectrogram,
                         max_len=target_spectrogram.size // self.configs.SP.NUM_MELS)
        print("Input at step {}: {}".format(step, sequence_to_text(input_seq)))
        return None
