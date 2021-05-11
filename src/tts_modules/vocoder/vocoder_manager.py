from tts_modules.vocoder.models.wavernn import WaveRNN
from tts_modules.vocoder.configs import hparams as hp
import torch
import os


class VocoderManager:

    def __init__(self, configs, test_dataloader=None, train_dataloader=None,
                 model=None, checkpoint_path=None):
        self.configs = configs
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.model = model
        self.checkpoint_path = checkpoint_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.model is None:
            self.__init_wavernn()

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
            save_wav(wav, os.path.join(self.configs.wav_save_path, 'result.wav'))
        return wav


    def __init_wavernn(self):
        self.model = WaveRNN(
            rnn_dims=hp.voc_rnn_dims,
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
            mode=hp.voc_mode
        ).to(self.device)
        if self.checkpoint_path is not None:
            self.__load_model()
        self.model.eval()

    def __load_model(self):
        if self.checkpoint_path is not None:
            self.model.load(self.checkpoint_path)
        else:
            print('Vocoder was not loaded!!!')

    def __save_checkpoint(self, path):
        pass

