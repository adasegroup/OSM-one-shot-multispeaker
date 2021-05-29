import torch
from ..tts_modules.encoder import SpeakerEncoderManager
from ..tts_modules.synthesizer import SynthesizerManager
from ..tts_modules.vocoder import VocoderManager
from .configs import get_default_main_configs


class MultispeakerManager:
    """
    A class used to control the three module managers and interactions between them

    Attributes
    -----------
    :param main_configs: main configuration (CfgNode object)
    :param encoder: Optional. The custom speaker encoder model. If None, the default DVecModel is used
    :param encoder_test_dataloader: Optional. Test dataset for speaker encoder model
    :param encoder_train_dataloader: Optional. Train dataset for speaker encoder model
    :param synthesizer: Optional. The custom synthesizer model. If None, the default Tacotron model is used
    :param synthesizer_test_dataloader: Optional. Test dataset for synthesizer model
    :param synthesizer_train_dataloader: Optional. Train dataset for synthesizer model
    :param vocoder: Optional. The custom vocoder model. If None, the default WaveRNN model is used
    :param vocoder_test_dataloader: Optional. Test dataset for vocoder model
    :param vocoder_train_dataloader: Optional. Train dataset for vocoder model

    Methods
    -----------
    inference()
        Runs the whole sound generation pipeline according to the given configuration.
        In particular, reads the given *.wav file with recorded sample voice and
        creates corresponding embeddings. After that reads the *.txt file with given text
        and produces the output result.wav file,
        where the text is read using the obtained voice embeddings.

    process_speaker(...)
        Processes the given *.wav file and produces the corresponding embeddings with the help of SpeakerEncoderManager

    synthesize_spectrograms(...)
        Creates spectrograms using voice embeddings and given text in *.txt file with the help of SynthesizerManager

    generate_waveform(..)
        Generates a result.wav file using the spectrograms with the help of VocoderManager
    """

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
        self.main_configs = main_configs
        if self.main_configs is None:
            self.main_configs = get_default_main_configs()
        self.encoder_manager = SpeakerEncoderManager(self.main_configs,
                                                     model=encoder,
                                                     test_dataloader=encoder_test_dataloader,
                                                     train_dataloader=encoder_train_dataloader
                                                     )

        self.synthesizer_manager = SynthesizerManager(self.main_configs,
                                                      model=synthesizer,
                                                      test_dataloader=synthesizer_test_dataloader,
                                                      train_dataloader=synthesizer_train_dataloader
                                                      )
        self.vocoder_manager = VocoderManager(self.main_configs,
                                              model=vocoder,
                                              test_dataloader=vocoder_test_dataloader,
                                              train_dataloader=vocoder_train_dataloader
                                              )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def inference(self):
        """
        Runs the whole sound generation pipeline according to the given configuration.
        In particular, reads the given *.wav file with recorded sample voice and
        creates corresponding embeddings. After that reads the *.txt file with given text
        and produces the output result.wav file,
        where the text is read using the obtained voice embeddings.
        The result.wav can be save according to the configs

        :return: generated wav file
        """
        embeddings = self.process_speaker(speaker_speech_path=self.main_configs.SPEAKER_SPEECH_PATH)
        with open(self.main_configs.INPUT_TEXTS_PATH, "r") as file:
            texts = file.readlines()
        specs = self.synthesize_spectrograms(texts=texts, embeddings=embeddings)
        specs = specs[0]
        wav = self.generate_waveform(specs)
        return wav

    def process_speaker(self, speaker_speech_path, save_embeddings_path=None,
                        save_embeddings_speaker_name="test_speaker"):
        """
        Processes the given *.wav file and produces the corresponding embeddings with the help of SpeakerEncoderManager

        :param speaker_speech_path: The path to the *.wav file with sample recordings of new speaker
        :param save_embeddings_path: Optional. The path for saving the obtained embeddings
        :param save_embeddings_speaker_name: Optional. The name of the file for saving the obtained embeddings

        :return: embeddings
        """
        embeddings = self.encoder_manager.process_speaker(speaker_speech_path,
                                                          save_embeddings_path=save_embeddings_path,
                                                          save_embeddings_speaker_name=save_embeddings_speaker_name
                                                          )
        return embeddings

    def synthesize_spectrograms(self, texts, embeddings, do_save_spectrograms=True):
        """
        Creates spectrograms using voice embeddings and given text in *.txt file with the help of SynthesizerManager

        :param texts: The text which is used to create spectrograms
        :param embeddings: The embeddings of a particular speaker used to create spectrograms
        :param do_save_spectrograms: Optional. Flag defines whether to save spectrograms or not

        :return: spectrograms
        """
        specs = self.synthesizer_manager.synthesize_spectrograms(texts,
                                                                 embeddings,
                                                                 do_save_spectrograms=do_save_spectrograms
                                                                 )
        return specs

    def generate_waveform(self, mel, normalize=True, batched=True,
                          target=8000, overlap=800, do_save_wav=True):
        """
        Generates a result.wav file using the spectrograms with the help of VocoderManager

        :param mel: mel-spectrograms used to create the rsulting wav file
        :param normalize: Optional. The flag defines whether to normalize the mel-spectrograms or not
        :param batched: Optional. Flag define whether to fold with overlap and to xfade and unfold or not
        :param target: Optional. Target timesteps for each index of batch
        :param overlap: Optional. Timesteps for both xfade and rnn warmup
        :param do_save_wav: Optional. Flag define whether to save the resulting wav to a file or not

        :return: The resulting wav
        """
        wav = self.vocoder_manager.infer_waveform(mel,
                                                  normalize=normalize,
                                                  batched=batched,
                                                  target=target,
                                                  overlap=overlap,
                                                  do_save_wav=do_save_wav
                                                  )
        return wav
