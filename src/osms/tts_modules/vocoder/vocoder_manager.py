from .. import AbstractTTSModuleManager
from .models import WaveRNN
from .configs import hparams as hp
from .utils.audio import save_wav
import torch
import os
import yaml


class VocoderManager(AbstractTTSModuleManager):
    def __init__(self,
                 main_configs,
                 model=None,
                 test_dataloader=None,
                 train_dataloader=None
                 ):
        super(VocoderManager, self).__init__(main_configs,
                                             model,
                                             test_dataloader,
                                             train_dataloader
                                             )

    def infer_waveform(self, mel, normalize=True, batched=True,
                       target=8000, overlap=800, do_save_wav=True):
        """
        Infers the waveform of a mel spectrogram output by the synthesizer (the format must match
        that of the synthesizer!)

        :param normalize:
        :param batched:
        :param target:
        :param overlap:
        :return:
        """
        if self.model is None:
            raise Exception("Load WaveRNN, please!")

        if normalize:
            mel = mel / hp.mel_max_abs_value
        mel = torch.from_numpy(mel[None, ...])
        wav = self.model.generate(mel, batched, target, overlap, hp.mu_law)
        if do_save_wav:
            save_wav(wav, os.path.join(self.main_configs["OUTPUT_AUDIO_DIR"], 'result.wav'))
        return wav

    def _load_local_configs(self):
        with open(self.main_configs["VocoderConfigPath"], "r") as ymlfile:
            self.model_config = yaml.load(ymlfile)

    def _init_baseline_model(self):
        self.model_name = "WaveRNN"
        self.model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                             fc_dims=hp.voc_fc_dims,
                             bits=hp.bits,
                             pad=hp.voc_pad,
                             upsample_factors=hp.voc_upsample_factors,
                             feat_dims=hp.num_mels,
                             compute_dims=hp.voc_compute_dims,
                             res_out_dims=hp.voc_res_out_dims,
                             res_blocks=hp.voc_res_blocks,
                             hop_length=hp.hop_length,
                             sample_rate=hp.sample_rate,
                             device=self.device,
                             mode=hp.voc_mode,
                             verbose=self.model_config["verbose"]
                             ).to(self.device)
        if self.model_config["pretrained"]:
            self._load_baseline_model()
        return None
