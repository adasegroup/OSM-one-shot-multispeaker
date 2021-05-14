import yaml
import numpy as np
import librosa


class Wav2MelTransform(object):
    """Interface """

    def __init__(self, audio_config_yaml_path):
        with open(audio_config_yaml_path, "r") as ymlfile:
            self.audio_config_yaml = yaml.load(ymlfile)

    def __call__(self, *args, **kwargs):
        return self.Wav2Mel(*args, **kwargs)

    def Wav2Mel(self, *args, **kwargs):
        pass


class StandardWav2MelTransform(Wav2MelTransform):
    def __init__(self, audio_config_yaml_path):
        super(StandardWav2MelTransform, self).__init__(audio_config_yaml_path)

    def Wav2Mel(self, wav):
        """
        Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
        Note: this not a log-mel spectrogram.
        """
        sampling_rate = self.audio_config_yaml["SAMPLING_RATE"]
        mel_window_length = self.audio_config_yaml["MEL_WINDOW_LENGTH"]
        mel_window_step = self.audio_config_yaml["MEL_WINDOW_STEP"]
        mel_n_channels = self.audio_config_yaml["MEL_N_CHANNELS"]
        frames = librosa.feature.melspectrogram(
            wav,
            sampling_rate,
            n_fft=int(sampling_rate * mel_window_length / 1000),
            hop_length=int(sampling_rate * mel_window_step / 1000),
            n_mels=mel_n_channels
        )
        return frames.astype(np.float32).T
