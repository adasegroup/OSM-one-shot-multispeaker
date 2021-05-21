from .. import AbstractTTSModuleManager
from .models import DVecModel
from .data.wav_preprocessing import StandardAudioPreprocessor
from .data.wav2mel import StandardWav2MelTransform
from .configs import get_default_encoder_config
from osms.common.configs import update_config
from .utils.Trainer import SpeakerEncoderTrainer
from .data.dataset import PreprocessLibriSpeechDataset, SpeakerEncoderDataset, SpeakerEncoderDataLoader
import torch
import numpy as np
import os


class SpeakerEncoderManager(AbstractTTSModuleManager):
    """Attributes
        ----------
        main_configs: dict
            dictionary of configuration files
        preprocessor: AudioPreprocessor
            preprocess wav audio
        wav2mel: Wav2MelTransform
            transfrom preprocessed wav to mel spectogram
        current_embed: numpy.array
            the latest produced embedding
        audio_config: dict
            Audio Configuration file
        device: int
            Chosen device: GPU/CPU
        model : nn.Module
            Speaker Encoder NN model
        Methods
        -------
        __init_baseline_model()
            initialize baseline model

        load_model()
            load model weights and optimizer if available

        process_speaker(speaker_speech_path, save_embeddings_path=None,
                        save_embeddings_speaker_name="test_speaker")
            produce embeddings for one utterance

        save_embeddings(save_embeddings_path,save_embeddings_speaker_name)
            save embeddings for given path and name

        embed_utterance(wav, using_partials=True, return_partials=False)
            calculate embeddings for given wav

        compute_partial_slices(n_samples, min_pad_coverage=0.75, overlap=0.5)
            Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
            partial utterances of <partial_utterance_n_frames> each.

        """
    def __init__(self,
                 main_configs,
                 model=None,
                 preprocessor=None,
                 wav2mel=None,
                 test_dataloader=None,
                 train_dataloader=None,
                 optimizer=None
                 ):
        super(SpeakerEncoderManager, self).__init__(main_configs,
                                                    model,
                                                    test_dataloader,
                                                    train_dataloader,
                                                    optimizer
                                                    )
        if preprocessor is None:
            self.preprocessor = StandardAudioPreprocessor(self.module_configs)
        else:
            self.preprocessor = preprocessor

        if wav2mel is None:
            self.wav2mel = StandardWav2MelTransform(self.module_configs)
        else:
            self.wav2mel = wav2mel

        self.current_embed = None
        self.trainer = None
        self.dataset = None

    def __init_trainer(self):
        if self.train_dataloader is None:
            dataset_preprocessor = PreprocessLibriSpeechDataset(self.module_configs, self.preprocessor, self.wav2mel)
            dataset_preprocessor.preprocess_dataset()
            train_dataset = SpeakerEncoderDataset(self.module_configs)
            self.train_dataloader = SpeakerEncoderDataLoader(self.module_configs, train_dataset, 'train')
        # if self.test_dataloader is None:
        #     dataset_preprocessor = PreprocessLibriSpeechDataset(self.module_configs, self.preprocessor, self.wav2mel)
        #     dataset_preprocessor.preprocess_dataset()
        #     test_dataset = SpeakerEncoderDataset(self.module_configs)
        #     self.test_dataloader = SpeakerEncoderDataLoader(self.module_configs, train_dataset, 'test')
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.module_configs.TRAIN.LEARNING_RATE_INIT)

        self.trainer = SpeakerEncoderTrainer(self.module_configs, self.model, self.train_dataloader,
                                             self.test_dataloader, self.optimizer)

    def train_session(self, number_steps=None):
        self.__init_trainer()
        if number_steps is None:
            self.trainer.train(self.module_configs.TRAIN.NUMBER_STEPS)
        else:
            self.trainer.train(number_steps)




    def process_speaker(self, speaker_speech_path, save_embeddings_path=None,
                        save_embeddings_speaker_name="test_speaker"):
        """ produce embeddings for one utterance"""
        processed_wav = self.preprocessor.preprocess_wav(speaker_speech_path)
        embed = self.embed_utterance(processed_wav)
        self.current_embed = embed
        if save_embeddings_path is not None:
            self.save_embeddings(save_embeddings_path, save_embeddings_speaker_name)

        return embed



    # TODO: Correct config loading
    def _load_local_configs(self):
        self.module_configs = get_default_encoder_config()
        self.module_configs = update_config(self.module_configs,
                                            update_file=self.main_configs.SPEAKER_ENCODER_CONFIG_FILE
                                            )
        # if "AudioConfigPath" in self.main_configs.keys():
        #     audio_config_path = self.main_configs["AudioConfigPath"]
        # else:
        #     audio_config_path = "./"
        # with open(audio_config_path, "r") as ymlfile:
        #     self.audio_config = yaml.load(ymlfile)
        # with open(self.main_configs["SpeakerEncoderConfigPath"], "r") as ymlfile:
        #     self.module_configs = yaml.load(ymlfile)
        return None

    def _init_baseline_model(self):
        self.model_name = "DVecModel"
        self.model = DVecModel(self.module_configs)
        if self.module_configs.MODEL.PRETRAINED:
            self._load_baseline_model()
        return None

    def save_embeddings(self, save_embeddings_path, save_embeddings_speaker_name):
        np.save(os.path.join(save_embeddings_path, save_embeddings_speaker_name), self.current_embed)

    def embed_utterance(self, wav, using_partials=True, return_partials=False):
        # Process the entire utterance if not using partials
        if not using_partials:
            # processed_wav = self.preprocessor.preprocess_wav(wav)
            frames = self.wav2mel.wav_to_mel(wav)
            frames = torch.from_numpy(frames[None, ...]).to(self.device)
            embed = self.model.forward(frames).detach().cpu().numpy()
            self.current_embed = embed[0]
            if return_partials:
                return embed[0], None, None
            return embed[0]

        # Compute where to split the utterance into partials and pad if necessary
        wave_slices, mel_slices = self.compute_partial_slices(len(wav))
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials
        # processed_wav = self.preprocessor.preprocess_wav(wav)
        frames = self.wav2mel.wav_to_mel(wav)
        frames_batch = np.array([frames[s] for s in mel_slices])
        frames = torch.from_numpy(frames_batch).to(self.device)
        partial_embeds = self.model.forward(frames).detach().cpu().numpy()

        # Compute the utterance embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)
        self.current_embed = embed
        if return_partials:
            return embed, partial_embeds, wave_slices
        return embed

    def compute_partial_slices(self, n_samples, min_pad_coverage=0.75, overlap=0.5):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
        partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
        spectrogram slices are returned, so as to make each partial utterance waveform correspond to
        its spectrogram. This function assumes that the mel spectrogram parameters used are those
        defined in params_data.py.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
        utterance
        :param min_pad_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
        then the last partial utterance will be considered, as if we padded the audio. Otherwise,
        it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
        utterance, this parameter is ignored so that the function always returns at least 1 slice.
        :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
        utterances are entirely disjoint.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        sampling_rate = self.module_configs.AUDIO.SAMPLING_RATE
        mel_window_step = self.module_configs.AUDIO.MEL_WINDOW_STEP
        partial_utterance_n_frames = self.module_configs.AUDIO.PARTIAL_N_FRAMES

        assert 0 <= overlap < 1
        assert 0 < min_pad_coverage <= 1

        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partial_utterance_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_pad_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices