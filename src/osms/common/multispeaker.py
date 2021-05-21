import torch
from ..tts_modules.encoder import SpeakerEncoderManager
from ..tts_modules.synthesizer import SynthesizerManager
from ..tts_modules.vocoder import VocoderManager


class MultispeakerManager:
    def __init__(self,
                 main_configs,
                 encoder=None,
                 encoder_test_dataloader=None,
                 encoder_train_dataloader=None,
                 synthesizer=None,
                 synthesizer_test_dataloader=None,
                 synthesizer_train_dataloader=None,
                 vocoder=None,
                 vocoder_test_dataloader=None,
                 vocoder_train_dataloader=None
                 ):
        self.configs = main_configs
        self.encoder_manager = SpeakerEncoderManager(main_configs,
                                                     model=encoder,
                                                     test_dataloader=encoder_test_dataloader,
                                                     train_dataloader=encoder_train_dataloader
                                                     )

        self.synthesizer_manager = SynthesizerManager(main_configs,
                                                      model=synthesizer,
                                                      test_dataloader=synthesizer_test_dataloader,
                                                      train_dataloader=synthesizer_train_dataloader
                                                      )
        self.vocoder_manager = VocoderManager(main_configs,
                                              model=vocoder,
                                              test_dataloader=vocoder_test_dataloader,
                                              train_dataloader=vocoder_train_dataloader
                                              )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def inference(self):
        embeddings = self.process_speaker(speaker_speech_path=self.configs["SPEAKER_SPEECH_PATH"])
        with open(self.configs["INPUT_TEXTS_PATH"], "r") as file:
            texts = file.readlines()
        specs = self.synthesize_spectrograms(texts=texts, embeddings=embeddings)
        specs = specs[0]
        wav = self.generate_waveform(specs)
        return wav

    def process_speaker(self, speaker_speech_path, save_embeddings_path=None,
                        save_embeddings_speaker_name="test_speaker"):
        embeddings = self.encoder_manager.process_speaker(speaker_speech_path,
                                                          save_embeddings_path=save_embeddings_path,
                                                          save_embeddings_speaker_name=save_embeddings_speaker_name
                                                          )
        return embeddings

    def synthesize_spectrograms(self, texts, embeddings, do_save_spectrograms=True):
        specs = self.synthesizer_manager.synthesize_spectrograms(texts,
                                                                 embeddings,
                                                                 do_save_spectrograms=do_save_spectrograms
                                                                 )
        return specs

    def generate_waveform(self, mel, normalize=True, batched=True,
                          target=8000, overlap=800, do_save_wav=True):
        wav = self.vocoder_manager.infer_waveform(mel,
                                                  normalize=normalize,
                                                  batched=batched,
                                                  target=target,
                                                  overlap=overlap,
                                                  do_save_wav=do_save_wav
                                                  )
        return wav
