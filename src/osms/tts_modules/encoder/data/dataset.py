from torch.utils.data import Dataset, DataLoader
from typing import List
import numpy as np
from .DataObjects import Speaker, RandomCycler, SpeakerBatch
from pathlib import Path


class PreprocessDataset:
    def __init__(self, config, wav_preprocessor, wav2mel_transformer):
        self.config = config
        self.dataset_root = Path(config.DATASET.ROOT)
        self.output_dir = Path(config.DATASET.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        self.speaker_dirs = list(self.dataset_root.glob("*"))
        self.extension = config.DATASET.EXTENSION
        self.wav_preprocessor = wav_preprocessor
        self.wav2mel_transformer = wav2mel_transformer

    def preprocess_speaker_one_utterance(self, in_fpath, sources_file, out_mel_file):
        raise NotImplementedError

    def process_paths_to_speaker_utterances(self, speaker_dir) -> List:
        raise NotImplementedError

    def preprocess_dataset(self, n_speakers=None):
        # print(self.speaker_dirs)
        for idx, speaker_dir in enumerate(self.speaker_dirs):
            speaker_name = "_".join(speaker_dir.relative_to(self.dataset_root).parts)
            speaker_out_dir = self.output_dir.joinpath(speaker_name)
            speaker_out_dir.mkdir(exist_ok=True)
            sources_fpath = speaker_out_dir.joinpath("_sources.txt")
            sources_file = sources_fpath.open("w")
            for in_fpath in self.process_paths_to_speaker_utterances(speaker_dir):

                out_mel_fpath = "_".join(in_fpath.relative_to(speaker_dir).parts)
                out_mel_fpath = out_mel_fpath.replace(".%s" % self.extension, ".npy")
                out_mel_fpath = speaker_out_dir.joinpath(out_mel_fpath)

                self.preprocess_speaker_one_utterance(in_fpath, sources_file, out_mel_fpath)
            sources_file.close()
            if n_speakers is not None and idx == n_speakers-1:
                print(f'{n_speakers} speakers were preprocessed.')
                return None


class PreprocessLibriSpeechDataset(PreprocessDataset):

    def __init__(self, config, wav_preprocessor, wav2mel_transformer):
        super(PreprocessLibriSpeechDataset, self).__init__(config, wav_preprocessor, wav2mel_transformer)

    def process_paths_to_speaker_utterances(self, speaker_dir):
        return speaker_dir.glob("**/*.%s" % self.extension)

    def preprocess_speaker_one_utterance(self, in_fpath, sources_file, out_mel_fpath):

        wav = self.wav_preprocessor.preprocess_wav(in_fpath)
        if len(wav) == 0:
            return None

        # Create the mel spectrogram, discard those that are too short
        frames = self.wav2mel_transformer.wav_to_mel(wav)
        if len(frames) < self.config.AUDIO.PARTIAL_N_FRAMES:
            return None
        np.save(out_mel_fpath, frames)
        sources_file.write("%s,%s\n" % (out_mel_fpath, in_fpath))


class SpeakerEncoderDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.processed_dataset_root = Path(config.DATASET.OUTPUT_DIR)
        speaker_dirs = [f for f in self.processed_dataset_root.glob("[!.]*") if f.is_dir()]
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)

    def __getitem__(self, index):
        return next(self.speaker_cycler)


class SpeakerEncoderDataLoader(DataLoader):
    def __init__(self, config, dataset, mode, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0,
                 worker_init_fn=None):
        self.dataset = dataset
        self.config = config
        if mode == "train":
            self.speakers_per_batch = config.TRAIN.SPEAKERS_PER_BATCH
            self.utterances_per_speaker = config.TRAIN.UTTERANCES_PER_SPEAKER
        if mode == "validate":
            self.speakers_per_batch = config.VALIDATE.SPEAKERS_PER_BATCH
            self.utterances_per_speaker = config.VALIDATE.UTTERANCES_PER_SPEAKER

        super().__init__(
            dataset=dataset,
            batch_size=self.speakers_per_batch,
            shuffle=False,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=self.collate,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, self.config.AUDIO.PARTIAL_N_FRAMES)
