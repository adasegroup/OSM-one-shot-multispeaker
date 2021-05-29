"""
This module contains many usefull functions for audio preprocessing, processing and postprocessing
"""

import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    sf.write(path, wav.astype(np.float32), sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


# From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def get_hop_size(configs):
    hop_size = configs.SP.HOP_SIZE
    if hop_size is None:
        assert configs.frame_shift_ms is not None
        hop_size = int(configs.frame_shift_ms / 1000 * configs.SP.SAMPLE_RATE)
    return hop_size


def linearspectrogram(wav, configs):
    D = _stft(preemphasis(wav, configs.SP.PREEMPHASIS, configs.SP.PREEMPHASIZE), configs)
    S = _amp_to_db(np.abs(D), configs) - configs.SP.REF_LEVEL_DB

    if configs.MEL.SIGNAL_NORMALIZATION:
        return _normalize(S, configs)
    return S


def melspectrogram(wav, configs):
    D = _stft(preemphasis(wav, configs.SP.PREEMPHASIS, configs.SP.PREEMPHASIZE), configs)
    S = _amp_to_db(_linear_to_mel(np.abs(D), configs), configs) - configs.SP.REF_LEVEL_DB

    if configs.MEL.SIGNAL_NORMALIZATION:
        return _normalize(S, configs)
    return S


def inv_linear_spectrogram(linear_spectrogram, configs):
    """Converts linear spectrogram to waveform using librosa"""
    if configs.MEL.SIGNAL_NORMALIZATION:
        D = _denormalize(linear_spectrogram, configs)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + configs.SP.REF_LEVEL_DB)  # Convert back to linear

    if configs.AUDIO.USE_LWS:
        processor = _lws_processor(configs)
        D = processor.run_lws(S.astype(np.float64).T ** configs.MEL.POWER)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, configs.SP.PREEMPHASIS, configs.SP.PREEMPHASIZE)
    else:
        return inv_preemphasis(_griffin_lim(S ** configs.MEL.POWER, configs),
                               configs.SP.PREEMPHASIS,
                               configs.SP.PREEMPHASIZE
                               )


def inv_mel_spectrogram(mel_spectrogram, configs):
    """Converts mel spectrogram to waveform using librosa"""
    if configs.MEL.SIGNAL_NORMALIZATION:
        D = _denormalize(mel_spectrogram, configs)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + configs.SP.REF_LEVEL_DB), configs)  # Convert back to linear

    if configs.AUDIO.USE_LWS:
        processor = _lws_processor(configs)
        D = processor.run_lws(S.astype(np.float64).T ** configs.MEL.POWER)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y,
                               configs.SP.PREEMPHASIS,
                               configs.SP.PREEMPHASIZE)
    else:
        return inv_preemphasis(_griffin_lim(S ** configs.MEL.POWER, configs),
                               configs.SP.PREEMPHASIS,
                               configs.SP.PREEMPHASIZE)


def _lws_processor(configs):
    import lws
    return lws.lws(configs.SP.N_FFT, get_hop_size(configs), fftsize=configs.SP.WIN_SIZE, mode="speech")


def _griffin_lim(S, configs):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, configs)
    for i in range(configs.MEL.GRIFFIN_LIM_ITERS):
        angles = np.exp(1j * np.angle(_stft(y, configs)))
        y = _istft(S_complex * angles, configs)
    return y


def _stft(y, configs):
    if configs.AUDIO.USE_LWS:
        return _lws_processor(configs).stft(y).T
    else:
        return librosa.stft(y=y,
                            n_fft=configs.SP.N_FFT,
                            hop_length=get_hop_size(configs),
                            win_length=configs.SP.WIN_SIZE
                            )


def _istft(y, configs):
    return librosa.istft(y, hop_length=get_hop_size(configs), win_length=configs.SP.WIN_SIZE)


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram, configs):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(configs)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, configs):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(configs))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(configs):
    assert configs.AUDIO.FMAX <= configs.SP.SAMPLE_RATE // 2
    return librosa.filters.mel(configs.SP.SAMPLE_RATE, configs.SP.N_FFT, n_mels=configs.SP.NUM_MELS,
                               fmin=configs.SP.FMIN, fmax=configs.AUDIO.FMAX)


def _amp_to_db(x, configs):
    min_level = np.exp(configs.SP.MIN_LEVEL_DB / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(S, configs):
    if configs.AUDIO.ALLOW_CLIPPING_IN_NORMALIZATION:
        if configs.AUDIO.SYMMETRIC_MELS:
            return np.clip((2 * configs.SP.MAX_ABS_VALUE) * (
                        (S - configs.SP.MIN_LEVEL_DB) / (-configs.SP.MIN_LEVEL_DB)) - configs.SP.MAX_ABS_VALUE,
                           -configs.SP.MAX_ABS_VALUE, configs.SP.MAX_ABS_VALUE)
        else:
            return np.clip(configs.SP.MAX_ABS_VALUE * ((S - configs.SP.MIN_LEVEL_DB) / (-configs.SP.MIN_LEVEL_DB)), 0,
                           configs.SP.MAX_ABS_VALUE)

    assert S.max() <= 0 and S.min() - configs.SP.MIN_LEVEL_DB >= 0
    if configs.AUDIO.SYMMETRIC_MELS:
        return (2 * configs.SP.MAX_ABS_VALUE) * (
                    (S - configs.SP.MIN_LEVEL_DB) / (-configs.SP.MIN_LEVEL_DB)) - configs.SP.MAX_ABS_VALUE
    else:
        return configs.SP.MAX_ABS_VALUE * ((S - configs.SP.MIN_LEVEL_DB) / (-configs.SP.MIN_LEVEL_DB))


def _denormalize(D, configs):
    if configs.AUDIO.ALLOW_CLIPPING_IN_NORMALIZATION:
        if configs.AUDIO.SYMMETRIC_MELS:
            return (((np.clip(D, -configs.SP.MAX_ABS_VALUE,
                              configs.SP.MAX_ABS_VALUE) + configs.SP.MAX_ABS_VALUE) * -configs.SP.MIN_LEVEL_DB / (
                                 2 * configs.SP.MAX_ABS_VALUE))
                    + configs.SP.MIN_LEVEL_DB)
        else:
            return ((np.clip(D, 0,
                             configs.SP.MAX_ABS_VALUE) * -configs.SP.MIN_LEVEL_DB / configs.SP.MAX_ABS_VALUE) + configs.SP.MIN_LEVEL_DB)

    if configs.AUDIO.SYMMETRIC_MELS:
        return (((D + configs.SP.MAX_ABS_VALUE) * -configs.SP.MIN_LEVEL_DB / (
                    2 * configs.SP.MAX_ABS_VALUE)) + configs.SP.MIN_LEVEL_DB)
    else:
        return ((D * -configs.SP.MIN_LEVEL_DB / configs.SP.MAX_ABS_VALUE) + configs.SP.MIN_LEVEL_DB)
