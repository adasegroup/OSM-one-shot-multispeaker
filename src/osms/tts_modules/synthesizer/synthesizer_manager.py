import torch
from .. import AbstractTTSModuleManager
# from .utils import audio
from .utils.symbols import symbols
from .utils.text import text_to_sequence
from .models import Tacotron
from .configs import get_default_synthesizer_config
from osms.common.configs import update_config
from typing import Union, List
import numpy as np


class SynthesizerManager(AbstractTTSModuleManager):
    def __init__(self,
                 main_configs,
                 model=None,
                 test_dataloader=None,
                 train_dataloader=None
                 ):
        super(SynthesizerManager, self).__init__(main_configs,
                                                 model,
                                                 test_dataloader,
                                                 train_dataloader
                                                 )

    def __call__(self, *args, **kwargs):
        return self.synthesize_spectrograms(*args, **kwargs)

    @staticmethod
    def pad1d(x, max_len, pad_value=0):
        return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False,
                                do_save_spectrograms=True):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.
        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256)
        :param return_alignments: if True, a matrix representing the alignments between the
        characters
        and each decoder output step will be returned for each spectrogram
        :param do_save_spectrograms: bool flag defines whether to save obtained spectrograms or not
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the
        sequence length of spectrogram i, and possibly the alignments.
        """
        # Preprocess text inputs
        inputs = [text_to_sequence(text.strip(), self.module_configs.MODEL.CLEANER_NAMES) for text in texts]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        # Batch inputs
        batched_inputs = [inputs[i:i+self.module_configs.DATA.SYNTHESIS_BATCH_SIZE]
                             for i in range(0, len(inputs), self.module_configs.DATA.SYNTHESIS_BATCH_SIZE)]
        batched_embeds = [embeddings[i:i+self.module_configs.DATA.SYNTHESIS_BATCH_SIZE]
                             for i in range(0, len(embeddings), self.module_configs.DATA.SYNTHESIS_BATCH_SIZE)]

        specs = []
        for i, batch in enumerate(batched_inputs, 1):

            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [SynthesizerManager.pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            speaker_embeds = np.stack(batched_embeds[i-1])

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

            # Inference
            _, mels, alignments = self.model.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            for m in mels:
                # Trim silence from end of each spectrogram
                while np.max(m[:, -1]) < self.module_configs.MODEL.STOP_THRESHOLD:
                    m = m[:, :-1]
                specs.append(m)

        if do_save_spectrograms:
            if return_alignments:
                self.save_spectrograms(specs, alignments)
            else:
                self.save_spectrograms(specs)

        return (specs, alignments) if return_alignments else specs

    def save_spectrograms(self, specs, alignments=None):
        pass

    def _load_local_configs(self):
        self.module_configs = get_default_synthesizer_config()
        self.module_configs = update_config(self.module_configs,
                                            update_file=self.main_configs.SYNTHESIZER_CONFIG_FILE
                                            )
        return None

    def _init_baseline_model(self):
        self.model_name = "Tacotron"
        self.model = Tacotron(embed_dims=self.module_configs.MODEL.EMBED_DIMS,
                              num_chars=len(symbols),
                              encoder_dims=self.module_configs.MODEL.ENCODER_DIMS,
                              decoder_dims=self.module_configs.MODEL.DECODER_DIMS,
                              n_mels=self.module_configs.SP.NUM_MELS,
                              fft_bins=self.module_configs.SP.NUM_MELS,
                              postnet_dims=self.module_configs.MODEL.POSTNET_DIMS,
                              encoder_K=self.module_configs.MODEL.ENCODER_K,
                              lstm_dims=self.module_configs.MODEL.LSTM_DIMS,
                              postnet_K=self.module_configs.MODEL.POSTNET_K,
                              num_highways=self.module_configs.MODEL.NUM_HIGHWAYS,
                              dropout=self.module_configs.MODEL.DROPOUT,
                              stop_threshold=self.module_configs.MODEL.STOP_THRESHOLD,
                              speaker_embedding_size=self.module_configs.SV2TTS.SPEAKER_EMBEDDING_SIZE,
                              verbose=self.module_configs.VERBOSE
                              ).to(self.device)
        if self.module_configs.MODEL.PRETRAINED:
            self._load_baseline_model()
        return None
