import torch
from torch.utils.data import DataLoader
from .data import SynthesizerDataset, collate_synthesizer
from .models import Tacotron
from .utils.symbols import symbols
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import requests


def run_synthesis(in_dir, out_dir, config):
    # Generate ground truth-aligned mels for vocoder training
    synth_dir = Path(out_dir).joinpath("mels_gta")
    synth_dir.mkdir(exist_ok=True)

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Synthesizer using device:", device)

    # Instantiate Tacotron model
    model = Tacotron(embed_dims=config.MODEL.EMBED_DIMS,
                     num_chars=len(symbols),
                     encoder_dims=config.MODEL.ENCODER_DIMS,
                     decoder_dims=config.MODEL.DECODER_DIMS,
                     n_mels=config.SP.NUM_MELS,
                     fft_bins=config.SP.NUM_MELS,
                     postnet_dims=config.MODEL.POSTNET_DIMS,
                     encoder_K=config.MODEL.ENCODER_K,
                     lstm_dims=config.MODEL.LSTM_DIMS,
                     postnet_K=config.MODEL.POSTNET_K,
                     num_highways=config.MODEL.NUM_HIGHWAYS,
                     dropout=0,
                     stop_threshold=config.MODEL.STOP_THRESHOLD,
                     speaker_embedding_size=config.SV2TTS.SPEAKER_EMBEDDING_SIZE,
                     verbose=config.VERBOSE
                     ).to(device)

    # Load the weights
    load_baseline_model(model, config, device)

    # Synthesize using same reduction factor as the model is currently trained
    r = np.int32(model.r)

    # Set model to eval mode (disable gradient and zoneout)
    model.eval()

    # Initialize the dataset
    in_dir = Path(in_dir)
    metadata_fpath = in_dir.joinpath("train.txt")
    mel_dir = in_dir.joinpath("mels")
    embed_dir = in_dir.joinpath("embeds")

    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, config)
    data_loader = DataLoader(dataset,
                             collate_fn=lambda batch: collate_synthesizer(batch, r, config),
                             batch_size=config.DATA.SYNTHESIS_BATCH_SIZE,
                             num_workers=2,
                             shuffle=False,
                             pin_memory=True)

    # Generate GTA mels
    meta_out_fpath = Path(out_dir).joinpath("synthesized.txt")
    with open(meta_out_fpath, "w") as file:
        for i, (texts, mels, embeds, idx) in tqdm(enumerate(data_loader), total=len(data_loader)):
            texts = texts.to(device)
            mels = mels.to(device)
            embeds = embeds.to(device)

            _, mels_out, _ = model(texts, mels, embeds)

            for j, k in enumerate(idx):
                # Note: outputs mel-spectrogram files and target ones have same names, just different folders
                mel_filename = Path(synth_dir).joinpath(dataset.metadata[k][1])
                mel_out = mels_out[j].detach().cpu().numpy().T

                # Use the length of the ground truth mel to remove padding from the generated mels
                mel_out = mel_out[:int(dataset.metadata[k][4])]

                # Write the spectrogram to disk
                np.save(mel_filename, mel_out, allow_pickle=False)

                # Write metadata into the synthesized file
                file.write("|".join(dataset.metadata[k]))


def load_baseline_model(model, config, device):
    checkpoint = get_baseline_checkpoint(model, config, device)
    model = load_routine(model, checkpoint, device)
    return model


def get_baseline_checkpoint(model, config, device):
    if not os.path.exists(config.MODEL.CHECKPOINT_DIR_PATH):
        os.mkdir(config.MODEL.CHECKPOINT_DIR_PATH)
    file_full_path = os.path.join(
        config.MODEL.CHECKPOINT_DIR_PATH,
        model.get_download_url().split('/')[-1]
    )
    if not os.path.isfile(file_full_path):
        model = load_from_dropbox(model, config, file_full_path)
    else:
        if config.VERBOSE:
            print(f'Loading Tacatron checkpoint from {file_full_path}')
    checkpoint = torch.load(file_full_path, map_location=device)
    return checkpoint


def load_from_dropbox(model, config, file_full_path):
    if config.VERBOSE:
        print(f'Downloading Tacatron checkpoint from Dropbox...')
    try:
        req = requests.get(model.get_download_url())
        with open(file_full_path, 'wb') as f:
            f.write(req.content)
    except requests.exceptions.RequestException as e:
        print(f'Baseline Tacatron checkpoint was not loaded from Dropbox!')
        print(f'Stacktrace: {e}')
    return model


def load_routine(model, checkpoint, device):
    model_state_dict = checkpoint["model_state"] if "model_state" in checkpoint.keys() else checkpoint
    model.load_state_dict(model_state_dict)
    model.eval()
    model = model.to(device)
    return model
