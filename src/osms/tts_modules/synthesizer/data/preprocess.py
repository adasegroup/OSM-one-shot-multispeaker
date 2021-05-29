from multiprocessing.pool import Pool
from .audio import melspectrogram
from functools import partial
from itertools import chain
from pathlib import Path
from ..utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa


class SynthesizerPreprocessor:
    """
    The class which controls the preprocessing procedure of datasets for Synthesizer

    Attributes
    -----------
    configs: Synthesizer yacs config
    datasets_root: The path to the root of the dataset
    out_dir: The path to the output results
    datasets_name: The name of the dataset
    subfolders: Subfolders in the dataset
    no_alignments: Flag determines whether to do alignments or not
    skip_existing: Flag determines whether to skip existing utterances or not
    n_processes: Number of processes
    encoder_manager: SpeakerEncoderManger object

    Methods
    -----------
    preprocess_dataset()
        The main method where all the preprocessing steps are done sequentially
    """
    def __init__(self, configs, encoder_manager):
        self.configs = configs
        self.datasets_root = Path(self.configs.DATA.DATASET_ROOT_PATH)
        self.out_dir = Path(self.configs.DATA.SYN_DIR)
        self.datasets_name = self.configs.DATA.DATASETS_NAME
        self.subfolders = self.configs.DATA.SUBFOLDERS
        self.no_alignments = self.configs.DATA.NO_ALIGNMENTS
        self.skip_existing = self.configs.DATA.SKIP_EXISTING
        self.n_processes = self.configs.DATA.N_PROCESSES
        self.encoder_manager = encoder_manager

    def preprocess_dataset(self):
        """
        The main method where all the preprocessing steps are done sequentially

        :return: None
        """
        # Gather the input directories
        dataset_root = self.datasets_root.joinpath(self.datasets_name)
        input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in self.subfolders.split(",")]
        print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
        assert all(input_dir.exists() for input_dir in input_dirs)

        # Create the output directories for each output file type
        self.out_dir.joinpath("mels").mkdir(exist_ok=True)
        self.out_dir.joinpath("audio").mkdir(exist_ok=True)

        # Create a metadata file
        metadata_fpath = self.out_dir.joinpath("train.txt")
        metadata_file = metadata_fpath.open("a" if self.skip_existing else "w", encoding="utf-8")

        # Preprocess the dataset
        speaker_dirs = list(chain.from_iterable(input_dir.glob("[!.]*") for input_dir in input_dirs))
        func = partial(preprocess_speaker,
                       out_dir=self.out_dir,
                       skip_existing=self.skip_existing,
                       configs=self.configs,
                       no_alignments=self.no_alignments,
                       encoder_manager=self.encoder_manager
                       )
        job = Pool(self.n_processes).imap(func, speaker_dirs)
        for speaker_metadata in tqdm(job, self.datasets_name, len(speaker_dirs), unit="speakers"):
            for metadatum in speaker_metadata:
                metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
        metadata_file.close()

        # Verify the contents of the metadata file
        with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        mel_frames = sum([int(m[4]) for m in metadata])
        timesteps = sum([int(m[3]) for m in metadata])
        hours = (timesteps / self.configs.SP.SAMPLE_RATE) / 3600
        print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
              (len(metadata), mel_frames, timesteps, hours))
        print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
        print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
        print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))
        return None


def preprocess_speaker(speaker_dir,
                       out_dir: Path,
                       skip_existing: bool,
                       configs,
                       no_alignments: bool,
                       encoder_manager
                       ):
    metadata = []
    for book_dir in speaker_dir.glob("[!.]*"):
        if no_alignments:
            # Gather the utterance audios and texts
            # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
            extensions = ["*.wav", "*.flac", "*.mp3"]
            for extension in extensions:
                wav_fpaths = book_dir.glob(extension)

                for wav_fpath in wav_fpaths:
                    # Load the audio waveform
                    wav, _ = librosa.load(str(wav_fpath), configs.SP.SAMPLE_RATE)
                    if configs.DATA.RESCALE:
                        wav = wav / np.abs(wav).max() * configs.DATA.RESCALING_MAX

                    # Get the corresponding text
                    # Check for .txt (for compatibility with other datasets)
                    text_fpath = wav_fpath.with_suffix(".txt")
                    if not text_fpath.exists():
                        # Check for .normalized.txt (LibriTTS)
                        text_fpath = wav_fpath.with_suffix(".normalized.txt")
                        assert text_fpath.exists()
                    with text_fpath.open("r") as text_file:
                        text = "".join([line for line in text_file])
                        text = text.replace("\"", "")
                        text = text.strip()

                    # Process the utterance
                    metadata.append(process_utterance(wav,
                                                      text,
                                                      out_dir,
                                                      str(wav_fpath.with_suffix("").name),
                                                      skip_existing,
                                                      configs,
                                                      encoder_manager
                                                      )
                                    )
        else:
            # Process alignment file (LibriSpeech support)
            # Gather the utterance audios and texts
            try:
                alignments_fpath = next(book_dir.glob("*.alignment.txt"))
                with alignments_fpath.open("r") as alignments_file:
                    alignments = [line.rstrip().split(" ") for line in alignments_file]
            except StopIteration:
                # A few alignment files will be missing
                continue

            # Iterate over each entry in the alignments file
            for wav_fname, words, end_times in alignments:
                wav_fpath = book_dir.joinpath(wav_fname + ".flac")
                assert wav_fpath.exists()
                words = words.replace("\"", "").split(",")
                end_times = list(map(float, end_times.replace("\"", "").split(",")))

                # Process each sub-utterance
                wavs, texts = split_on_silences(wav_fpath, words, end_times, configs)
                for i, (wav, text) in enumerate(zip(wavs, texts)):
                    sub_basename = "%s_%02d" % (wav_fname, i)
                    metadata.append(process_utterance(wav,
                                                      text,
                                                      out_dir,
                                                      sub_basename,
                                                      skip_existing,
                                                      configs,
                                                      encoder_manager
                                                      )
                                    )

    return [m for m in metadata if m is not None]


def split_on_silences(wav_fpath, words, end_times, configs):
    # Load the audio waveform
    wav, _ = librosa.load(str(wav_fpath), configs.SP.SAMPLE_RATE)
    if configs.DATA.RESCALE:
        wav = wav / np.abs(wav).max() * configs.DATA.RESCALING_MAX

    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == "" and words[-1] == ""

    # Find pauses that are too long
    mask = (words == "") & (end_times - start_times >= configs.SV2TTS.SILENCE_MIN_DURATION_SPLIT)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]

    # Profile the noise from the silences and perform noise reduction on the waveform
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * configs.SP.SAMPLE_RATE).astype(np.int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > configs.SP.SAMPLE_RATE * 0.02:
        profile = logmmse.profile_noise(noisy_wav, configs.SP.SAMPLE_RATE)
        wav = logmmse.denoise(wav, profile, eta=0)

    # Re-attach segments that are too short
    segments = list(zip(breaks[:-1], breaks[1:]))
    segment_durations = [start_times[end] - end_times[start] for start, end in segments]
    i = 0
    while i < len(segments) and len(segments) > 1:
        if segment_durations[i] < configs.SV2TTS.UTTERANCE_MIN_DURATION:
            # See if the segment can be re-attached with the right or the left segment
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            # Do not re-attach if it causes the joined utterance to be too long
            if joined_duration > configs.SP.HOP_SIZE * configs.DATA.MAX_MEL_FRAMES / configs.SP.SAMPLE_RATE:
                i += 1
                continue

            # Re-attach the segment with the neighbour of shortest duration
            j = i - 1 if left_duration <= right_duration else i
            segments[j] = (segments[j][0], segments[j + 1][1])
            segment_durations[j] = joined_duration
            del segments[j + 1], segment_durations[j + 1]
        else:
            i += 1

    # Split the utterance
    segment_times = [[end_times[start], start_times[end]] for start, end in segments]
    segment_times = (np.array(segment_times) * configs.SP.SAMPLE_RATE).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
    texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]

    return wavs, texts


def process_utterance(wav: np.ndarray,
                      text: str,
                      out_dir: Path,
                      basename: str,
                      skip_existing: bool,
                      configs,
                      encoder_manager
                      ):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.

    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
        return None

    # Trim silence
    if configs.AUDIO.TRIM_SILENCE:
        wav = encoder_manager.preprocessor.preprocess_wav(wav, normalize=False, trim_silence=True)

    # Skip utterances that are too short
    if len(wav) < configs.SV2TTS.UTTERANCE_MIN_DURATION * configs.SP.SAMPLE_RATE:
        return None

    # Compute the mel spectrogram
    mel_spectrogram = melspectrogram(wav, configs).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Skip utterances that are too long
    if mel_frames > configs.DATA.MAX_MEL_FRAMES and configs.AUDIO.CLIP_MELS_LENGTH:
        return None

    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text


def embed_utterance(fpaths, encoder_manager=None):
    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    wav = np.load(wav_fpath)
    wav = encoder_manager.preprocessor.preprocess_wav(wav)
    embed = encoder_manager.embed_utterance(wav)
    np.save(embed_fpath, embed, allow_pickle=False)


def create_embeddings(synthesizer_root: Path, n_processes: int, encoder_manager):
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]

    # Embed the utterances in separate threads
    func = partial(embed_utterance, encoder_manager=encoder_manager)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
