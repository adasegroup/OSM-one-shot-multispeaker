
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# class SpeakerEncoderDataset(Dataset):
#     """Speaker Encoder Dataset"""
#
#     def __init__(self, datasets_root, dataset_name, wav_preprocess=None, mel2wav = None):
#         """
#         Args:
#             datasets_root (string): Path to the directory with datasets.
#             dataset_name (string): dataset_name.
#             transform (callable, optional): Preprocessing wav
#         """
#         self.dataset_name = dataset_name
#         self.datasets_root = datasets_root
#         self.wav_preprocess = wav_preprocess
#         self.mel2wav = mel2wav

class LibriSpeechDataset(Dataset):
    def __init__(self, datasets_root, dataset_name, wav_preprocess=None, mel2wav = None):
        """
        Args:
            datasets_root (string): Path to the directory with datasets.
            dataset_name (string): dataset_name.
            transform (callable, optional): Preprocessing wav
        """
        self.dataset_name = dataset_name
        self.datasets_root = datasets_root
        self.wav_preprocess = wav_preprocess
        self.mel2wav = mel2wav






    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample