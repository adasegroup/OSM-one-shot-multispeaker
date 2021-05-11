import torch
from src.tts_modules.encoder.SpeakerEncoderManager import SpeakerEncoderManager
from src.tts_modules.synthesizer.synthesizer_manager import SynthesizerManager
from src.tts_modules.vocoder.vocoder_manager import VocoderManager



class MultispeakerManager:
    def __init__(self, configs,
                 encoder=None, encoder_checkpoint_path=None,
                 encoder_test_dataloader=None, encoder_train_dataloader=None,
                 synthesizer=None, synthesizer_checkpoint_path=None,
                 synthesizer_test_dataloader=None, synthesizer_train_dataloader=None,
                 vocoder=None, vocoder_checkpoint_path=None,
                 vocoder_test_dataloader=None, vocoder_train_dataloader=None):
        self.configs = configs
        self.encoder_manager = SpeakerEncoderManager(configs,
                                              model=encoder,
                                              checkpoint_path=encoder_checkpoint_path)

        self.synthesizer_manager = SynthesizerManager(configs,
                                                      model=synthesizer,
                                                      checkpoint_path=synthesizer_checkpoint_path,
                                                      test_dataloader=synthesizer_test_dataloader,
                                                      train_dataloader=synthesizer_train_dataloader)
        self.vocoder_manager = VocoderManager(configs,
                                              model=vocoder,
                                              checkpoint_path=vocoder_checkpoint_path,
                                              test_dataloader=vocoder_test_dataloader,
                                              train_dataloader=vocoder_train_dataloader)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def inference(self):
        pass

    def process_speaker(self, speaker_speech_path, save_embeddings_path=None,
                        save_embeddings_speaker_name="test_speaker"):
        self.encoder_manager.process_speaker(speaker_speech_path,
                                             save_embeddings_path=save_embeddings_path,
                                             save_embeddings_speaker_name=save_embeddings_speaker_name)

    def synthesize_spectrograms(self, texts, embeddings, do_save_spectrograms=True):
        self.synthesizer_manager.synthesize_spectrograms(texts, embeddings,
                                                         do_save_spectrograms=do_save_spectrograms)

    def generate_waveform(self, mel, normalize=True, batched=True,
                       target=8000, overlap=800, do_save_wav=True):
        self.vocoder_manager.infer_waveform(mel, normalize=normalize,
                                            batched=batched,
                                            target=target,
                                            overlap=overlap,
                                            do_save_wav=do_save_wav)
