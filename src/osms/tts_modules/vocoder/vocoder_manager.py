from .. import AbstractTTSModuleManager
from .models import WaveRNN
from .configs import get_default_vocoder_config
from osms.common.configs import update_config
from .utils.audio import save_wav
from .utils.Trainer import VocoderTrainer
import torch
import os


class VocoderManager(AbstractTTSModuleManager):
    """
    The main manager class which controls the vocoder model,
    the corresponding datasets and dataloaders and the training and inference procedures.

    Attributes
    ----------
    main_configs: yacs.config.CfgNode
        main configurations
    module_configs: yacs.config.CfgNode
        Vocoder configurations
    model: nn.Module
        Vocoder NN model
    model_name: str
        Name of the model
    optimizer: nn.optim.Optimizer
        Pytorch optimizer
    trainer: VocoderTrainer
        VocoderTrainer instance

    Methods
    -------
    infer_waveform(...)
        Infers the waveform of a mel spectrogram output by the synthesizer.

    train()
        Launchs the training session

    _load_local_configs()
        Loads yacs configs for Vocoder

    _init_baseline_model()
        Initializes baseline model

    """
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
        self.trainer = None

    def infer_waveform(self, mel, normalize=True, batched=True,
                       target=8000, overlap=800, do_save_wav=True):
        """
        Infers the waveform of a mel spectrogram output by the synthesizer (the format must match
        that of the synthesizer!)

        :param mel: mel-spectrograms used to create the rsulting wav file
        :param normalize: Optional. The flag defines whether to normalize the mel-spectrograms or not
        :param batched: Optional. Flag define whether to fold with overlap and to xfade and unfold or not
        :param target: Optional. Target timesteps for each index of batch
        :param overlap: Optional. Timesteps for both xfade and rnn warmup
        :param do_save_wav: Optional. Flag define whether to save the resulting wav to a file or not

        :return: The resulting wav
        """

        if self.model is None:
            raise Exception("Load WaveRNN, please!")

        if normalize:
            mel = mel / self.module_configs.SP.MAX_ABS_VALUE
        mel = torch.from_numpy(mel[None, ...])
        wav = self.model.generate(mel, batched, target, overlap, self.module_configs.SP.MU_LAW)
        if do_save_wav:
            save_wav(wav, os.path.join(self.main_configs.OUTPUT_AUDIO_DIR,
                                       self.main_configs.OUTPUT_AUDIO_FILE_NAME))
        return wav

    def _load_local_configs(self):
        self.module_configs = get_default_vocoder_config()
        self.module_configs = update_config(self.module_configs,
                                            update_file=self.main_configs.VOCODER_CONFIG_FILE
                                            )
        return None

    def __init_trainer(self):
        if self.model is None:
            self._init_baseline_model()
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters())
        self.trainer = VocoderTrainer(self.module_configs, self.model, self.optimizer)

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
                             apply_preemphasis=self.module_configs.SP.PREEMPHASIZE,
                             verbose=self.module_configs.VERBOSE
                             ).to(self.device)
        if self.module_configs.MODEL.PRETRAINED:
            self._load_baseline_model()
        return None

    def train(self, run_id, force_restart):
        self.__init_trainer()
        self.trainer.train(run_id, force_restart)

