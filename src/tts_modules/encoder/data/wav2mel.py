import numpy as np
import librosa


class Wav2MelTransform:
    """
    Interface for deriving a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    """
    def __init__(self, audio_config):
        """
        attributes:
            audio_config - Audio Configuration file
        """
        self.audio_config = audio_config

    def __call__(self, *args, **kwargs):
        return self.Wav2Mel(*args, **kwargs)

    def Wav2Mel(self, *args, **kwargs):
        """
        Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
        Note: this not a log-mel spectrogram.
        """
        pass


class StandardWav2MelTransform(Wav2MelTransform):
    """
    Standard wav to mel spectogram tranformer (baseline)
    """
    def __init__(self, audio_config):
        super(StandardWav2MelTransform, self).__init__(audio_config)

    def Wav2Mel(self, wav):
        """
        Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform
        using Librosa implementation.
        """
        sampling_rate = self.audio_config["SAMPLING_RATE"]
        mel_window_length = self.audio_config["MEL_WINDOW_LENGTH"]
        mel_window_step = self.audio_config["MEL_WINDOW_STEP"]
        mel_n_channels = self.audio_config["MEL_N_CHANNELS"]
        frames = librosa.feature.melspectrogram(
            wav,
            sampling_rate,
            n_fft=int(sampling_rate * mel_window_length / 1000),
            hop_length=int(sampling_rate * mel_window_step / 1000),
            n_mels=mel_n_channels
        )
        return frames.astype(np.float32).T
