from .. import AbstractTTSModuleManager
from .models import WaveRNN
from .configs import get_default_vocoder_config
from osms.common.configs import update_config
from .utils.audio import save_wav
import torch
import os


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

        :param mel:
        :param normalize:
        :param batched:
        :param target:
        :param overlap:
        :param do_save_wav:
        :return: wav file
        """
        if self.model is None:
            raise Exception("Load WaveRNN, please!")

        if normalize:
            mel = mel / self.module_configs.SP.MAX_ABS_VALUE
        mel = torch.from_numpy(mel[None, ...])
        wav = self.model.generate(mel, batched, target, overlap, self.module_configs.SP.MU_LAW)
        if do_save_wav:
            save_wav(wav, os.path.join(self.main_configs.OUTPUT_AUDIO_DIR, 'result.wav'))
        return wav

    def _load_local_configs(self):
        self.module_configs = get_default_vocoder_config()
        self.module_configs = update_config(self.module_configs,
                                            update_file=self.main_configs.VOCODER_CONFIG_FILE
                                            )
        return None

    def _init_baseline_model(self):
        self.model_name = "WaveRNN"
        self.model = WaveRNN(rnn_dims=self.module_configs.MODEL.RNN_DIMS,
                             fc_dims=self.module_configs.MODEL.FC_DIMS,
                             bits=self.module_configs.SP.BITS,
                             pad=self.module_configs.MODEL.PAD,
                             upsample_factors=self.module_configs.MODEL.UPSAMPLE_FACTORS,
                             feat_dims=self.module_configs.SP.NUM_MELS,
                             compute_dims=self.module_configs.MODEL.COMPUTE_DIMS,
                             res_out_dims=self.module_configs.MODEL.RES_OUT_DIMS,
                             res_blocks=self.module_configs.MODEL.RES_BLOCKS,
                             hop_length=self.module_configs.SP.HOP_SIZE,
                             sample_rate=self.module_configs.SP.SAMPLE_RATE,
                             device=self.device,
                             mode=self.module_configs.MODEL.MODE,
                             verbose=self.module_configs.VERBOSE
                             ).to(self.device)
        if self.module_configs.MODEL.PRETRAINED:
            self._load_baseline_model()
        return None
