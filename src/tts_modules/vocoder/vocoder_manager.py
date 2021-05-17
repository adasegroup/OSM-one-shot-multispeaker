from tts_modules.ttt_module_manager import TTSModuleManager
from tts_modules.vocoder.models.wavernn import WaveRNN
from tts_modules.vocoder.configs import hparams as hp
from tts_modules.vocoder.utils.audio import save_wav
import torch
import os
import yaml


class VocoderManager(TTSModuleManager):
    def __init__(self,
                 configs,
                 model=None,
                 test_dataloader=None,
                 train_dataloader=None
                 ):
        super(VocoderManager, self).__init__(configs,
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
            save_wav(wav, os.path.join(self.configs["OUTPUT_AUDIO_DIR"], 'result.wav'))
        return wav

    def _load_local_configs(self):
        with open(self.configs["VocoderConfigPath"], "r") as ymlfile:
            self.model_config = yaml.load(ymlfile)

    def _init_baseline_model(self):
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
                             mode=hp.voc_mode
                             ).to(self.device)
        if self.model_config["pretrained"]:
            self.load_model(self.model.get_download_url(), verbose=True)
        return None
