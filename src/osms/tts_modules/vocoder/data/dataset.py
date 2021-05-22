from torch.utils.data import Dataset,DataLoader
from ..utils import audio
import numpy as np
import torch


class VocoderDataset(Dataset):
    def __init__(self, config):
        self.config = config
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (self.config.DATASET.METADATA_PATH,
                                                        self.config.DATASET.MEL_DIR, self.config.DATASET.WAV_DIR))

        with self.config.DATASET.METADATA_PATH.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]

        gta_fnames = [x[1] for x in metadata if int(x[4])]
        gta_fpaths = [self.config.DATASET.MEL_DIR.joinpath(fname) for fname in gta_fnames]
        wav_fnames = [x[0] for x in metadata if int(x[4])]
        wav_fpaths = [self.config.DATASET.WAV_DIR.joinpath(fname) for fname in wav_fnames]
        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths))

        print("Found %d samples" % len(self.samples_fpaths))

    def __getitem__(self, index):
        mel_path, wav_path = self.samples_fpaths[index]

        # Load the mel spectrogram and adjust its range to [-1, 1]
        mel = np.load(mel_path).T.astype(np.float32) / self.config.SP.MAX_ABS_VALUE

        # Load the wav
        wav = np.load(wav_path)
        if self.config.SP.PREEMPHASIZE:
            wav = audio.pre_emphasis(wav)
        wav = np.clip(wav, -1, 1)

        # TODO: settle on whether this is any useful
        # Fix for missing padding
        r_pad = (len(wav) // self.config.SP.HOP_SIZE + 1) * self.config.SP.HOP_SIZE - len(wav)
        wav = np.pad(wav, (0, r_pad), mode='constant')
        assert len(wav) >= mel.shape[1] * self.config.SP.HOP_SIZE
        wav = wav[:mel.shape[1] * self.config.SP.HOP_SIZE]
        assert len(wav) % self.config.SP.HOP_SIZE == 0

        # Quantize the wav
        if self.config.MODEL.MODE == 'RAW':
            if self.config.SP.MU_LAW:
                quant = audio.encode_mu_law(wav, mu=2 ** self.config.SP.BITS)
            else:
                quant = audio.float_2_label(wav, bits=self.config.SP.BITS)
        elif self.config.MODEL.MODE == 'MOL':
            quant = audio.float_2_label(wav, bits=16)

        return mel.astype(np.float32), quant.astype(np.int64)

    def __len__(self):
        return len(self.samples_fpaths)


class VocoderDataloader(DataLoader):
    def __init__(self, config, dataset, num_workers=1, pin_memory=False, shuffle=True):
        self.dataset = dataset
        self.config = config

        super().__init__(
            dataset=dataset,
            batch_size=self.config.MODEL.BATCH_SIZE,
            num_workers=num_workers,
            collate_fn=self.collate_vocoder,
            pin_memory=pin_memory,
            shuffle=shuffle
        )

    def collate_vocoder(self, batch):
        mel_win = self.config.MODEL.SEQ_LEN // self.config.SP.HOP_SIZE + 2 * self.config.MODEL.PAD
        max_offsets = [x[0].shape[-1] - 2 - (mel_win + 2 * self.config.MODEL.PAD) for x in batch]
        mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        sig_offsets = [(offset + self.config.MODEL.PAD) * self.config.SP.HOP_SIZE for offset in mel_offsets]

        mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

        labels = [x[1][sig_offsets[i]:sig_offsets[i] + self.config.MODEL.SEQ_LEN + 1] for i, x in enumerate(batch)]

        mels = np.stack(mels).astype(np.float32)
        labels = np.stack(labels).astype(np.int64)

        mels = torch.tensor(mels)
        labels = torch.tensor(labels).long()

        x = labels[:, :self.config.MODEL.SEQ_LEN]
        y = labels[:, 1:]

        bits = 16 if self.config.MODEL.MODE == 'MOL' else self.config.SP.BITS

        x = audio.label_2_float(x.float(), bits)

        if self.config.MODEL.MODE == 'MOL':
            y = audio.label_2_float(y.float(), bits)

        return x, y, mels
