import torch
from tts_modules.ttt_module_manager import TTSModuleManager
from tts_modules.synthesizer.configs.hparams import hparams
# from tts_modules.synthesizer.utils import audio
from tts_modules.synthesizer.utils.symbols import symbols
from tts_modules.synthesizer.utils.text import text_to_sequence
from tts_modules.synthesizer.models.tacotron import Tacotron
from typing import Union, List
import numpy as np
import yaml


class SynthesizerManager(TTSModuleManager):
    def __init__(self,
                 configs,
                 model=None,
                 test_dataloader=None,
                 train_dataloader=None
                 ):
        super(SynthesizerManager, self).__init__(configs,
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
        inputs = [text_to_sequence(text.strip(), hparams.tts_cleaner_names) for text in texts]
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        # Batch inputs
        batched_inputs = [inputs[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(inputs), hparams.synthesis_batch_size)]
        batched_embeds = [embeddings[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(embeddings), hparams.synthesis_batch_size)]

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
                while np.max(m[:, -1]) < hparams.tts_stop_threshold:
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
        with open(self.configs["SynthesizerConfigPath"], "r") as ymlfile:
            self.model_config = yaml.load(ymlfile)

    def _init_baseline_model(self):
        self.model_name = "Tacotron"
        self.model = Tacotron(embed_dims=hparams.tts_embed_dims,
                              num_chars=len(symbols),
                              encoder_dims=hparams.tts_encoder_dims,
                              decoder_dims=hparams.tts_decoder_dims,
                              n_mels=hparams.num_mels,
                              fft_bins=hparams.num_mels,
                              postnet_dims=hparams.tts_postnet_dims,
                              encoder_K=hparams.tts_encoder_K,
                              lstm_dims=hparams.tts_lstm_dims,
                              postnet_K=hparams.tts_postnet_K,
                              num_highways=hparams.tts_num_highways,
                              dropout=hparams.tts_dropout,
                              stop_threshold=hparams.tts_stop_threshold,
                              speaker_embedding_size=hparams.speaker_embedding_size,
                              verbose=self.model_config["verbose"]
                              ).to(self.device)
        if self.model_config["pretrained"]:
            self._load_baseline_model()
        return None
