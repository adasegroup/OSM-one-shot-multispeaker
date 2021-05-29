import random
from pathlib import Path
from typing import List
import numpy as np


class RandomCycler:
    """
    Creates an internal copy of a sequence and allows access to its items in a constrained random
    order. For a source sequence of n items and one or several consecutive queries of a total
    of m items, the following guarantees hold (one implies the other):
        - Each item will be returned between m // n and ((m - 1) // n) + 1 times.
        - Between two appearances of the same item, there may be at most 2 * (n - 1) other items.
    """

    def __init__(self, source):
        if len(source) == 0:
            raise Exception("Can't create RandomCycler from an empty collection")
        self.all_items = list(source)
        self.next_items = []

    def sample(self, count: int):
        shuffle = lambda l: random.sample(l, len(l))

        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))
        return out

    def __next__(self):
        return self.sample(1)[0]


class Speaker:
    """
    The class which contains the set of utterances of a single speaker
    """

    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None

    def _load_utterances(self):
        """
        Loads utterances from sources and applies RandomCycler
        :return: None
        """

        with self.root.joinpath("_sources.txt").open("r") as sources_file:
            sources = [l.split(",") for l in sources_file]
        sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
        self.utterances = [Utterance(f, w) for f, w in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)
        return None

    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all
        utterances come up at least once every two cycles and in a random order every time.

        :param count: The number of partial utterances to sample from the set of utterances from
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance,
        frames are the frames of the partial utterances and range is the range of the partial
        utterance with regard to the complete utterance.
        """

        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a


class SpeakerBatch:
    """
    The class which contains speakers and their partials used for collate func in dataloader
    """

    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        self.speakers = speakers
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}

        # Array of shape (n_speakers * n_utterances, n_frames, mel_n), e.g. for 3 speakers with
        # 4 utterances each of 160 frames of 40 mel coefficients: (12, 160, 40)
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])


class Utterance:
    """
    The class which represents utterances and manipulations whith them
    """

    def __init__(self, frames_fpath, wave_fpath):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath

    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        """
        Crops the frames into a partial utterance of n_frames

        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the
        partial utterance in the complete utterance.
        """
        frames = self.get_frames()
        if frames.shape[0] == n_frames:
            start = 0
        else:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)
