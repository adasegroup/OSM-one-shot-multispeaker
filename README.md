# OSM: One-Shot Multi-speaker

## Problem Statement
One-Shot Multi-Speaker Text-to-Speech (OS MS TTS) systems are aimed to transform text into speech with voice determined by small single sample.
The main problem here is to reproduce the new unseen voice without retraining the network. 
There is an approach with three main stages which is used to solve this problem.
The unique for each voice speaker embeddings, which reveal the voice characteristics, are generated at the first stage (_Speaker Encoder_).
At the second stage (_Synthesizer_) the text is transformed to mel-spectrogram using previously obtained embeddings. 
Finally, the speech is reproduced from the mel-spectrogram with the _Vocoder_. 
But there is lack of implementations with these three parts properly combined. So the goal of our project is to create a flexible framework to combine these parts and provide replaceable modules and methods in each part.

## Main Challenges
By now we see the following main challenges:
- The solution to our problem consists of three subtasks, which already have a great solutions. 
  Therefore, the existing solutions for OS MS TTS are essentially a compilation of solutions for these individual problems, for which there are many ready-made and well-implemented solutions. 
  The main challenge is to make the framework flexible and ensure the compatibility of individual parts.
- The methods used in each subtask differ in the set of parameters and the nature of the algorithm. Therefore, it will be quite difficult to provide a single API.

## Baseline Solution
We choose solution proposed by the instructors as a baseline, which can be found [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning "here"). 
It is the implementation of [1] made in Google in 2018. 
Here authors use the speaker encoder, presented in [2], which generates a fixed-dimensional embedding vector known as d-vector. 
As for Synthesizer they use model based on Tacotron 2 [3] while an auto-regressive WaveNet-based is used as the Vocoder [4].
The following image taken from [1] represents the model overview:
<img src=doc/baseline_arch.png/>

## Pros And Cons
The [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning "Real-Time-Voice-Cloning") contains the realizations of encoder, Tacotron 2 and WaveRNN. 
The whole pipeline described in [1], including preprocessing steps, is also implemented in this repository. 
However, the project is not flexible enough. 
More specifically, in the current state it cannot be used as the framework for One-Shot Multi-Speaker Text-to-Speech system as there are no convenient mechanisms for manipulating with the three main modules. 
For example, the proposed multi-speaker TTS system in [5] cannot be easily implemented with the help of [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning "Real-Time-Voice-Cloning") as there are no extensibility points which allow to adjust the pipeline for the new method. 

## Our Improvement 
Our plan is to use the [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning "Real-Time-Voice-Cloning") as starting point with implemented baseline. 
We will introduce the flexible modular design of the framework. 
Such approach will help us to create the convenient API for external users who will be able to use our framework for incorporating the Multi-Speaker TTS system in their products. 
The API will also let the users customize modules and pipeline steps without changing the source code of the framework if needed. 
We will implement several Speaker Encoders (LDE, TDNN) and add them to our framework as well.

## Project Structure Overview
From a high point, our project consists of 3 main elements: Speaker Encoder, Synthesizer, Vocoder. For each of them, a manager is implemented that allows one to access the parameters and perform standard actions such as inference and training. Above them, we implemented OS MS TTS manager, which brings together all three parts  and allows one to make all pipeline and produce speech with needed voice. Each of these parts is also consist from elementary sub-parts typical for the corresponding elements. They can be described as follows:
 - _Speaker Encoder_: Here the base class is SpeakerEncoderManager, which allows to train and inference model. Also, we have already  implemented the Wav Audio Preprocessing Interface. So, one can customize their own audio preprocessing functions, which can differ even for the same dataset. Also, the custom model can be used. We added standard preprocessing function and model presented in [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning "Real-Time-Voice-Cloning") 
 - _Synthesizer_: Here the base class is SynthesizerManager, which allows to train and inference model. Also, the same situation with preprocessing functions, иге with one difference. In addition to the audio, one also needs to process the text. For now, we implemented text and audio preprocessing function, as these operations are needed during inference and training. The baseline is from [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning "Real-Time-Voice-Cloning") 
 - _Vocoder_: Here the base class is VocoderManager, which allows to train, inference vocoder model and to set all the states it needs. The baseline is from [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning "Real-Time-Voice-Cloning") 
 
 ## Evaluation Results
 In our repository we added notebook, where one can upload the voice audio, .txt file and produce speech with cloned voice.
 Despite the weights of pretrained models are downloaded automatically at the first run, the user can still download archive [here](https://github.com/blue-fish/Real-Time-Voice-Cloning/releases/download/v1.0/pretrained.zip "load weights")
 Other instructions are in the notebook [here](https://github.com/adasegroup/OSM-one-shot-multispeaker/blob/main/notebooks/demonstration.ipynb " notebook")
 


## Roles of the Participants
Nikolay will design the modular architecture, API for external usage and training pipeline.
Gleb will implement the working stack of models, write documentations and usage examples.
## Project Structure 
```bash
.
└── osms
    ├── __init__.py
    ├── common
    │   ├── __init__.py
    │   ├── configs
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── main_config.yaml
    │   └── multispeaker.py
    ├── main.py
    ├── tts_modules
    │   ├── __init__.py
    │   ├── encoder
    │   │   ├── __init__.py
    │   │   ├── configs
    │   │   │   ├── AudioConfig.yaml
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   └── dVecModelConfig.yaml
    │   │   ├── data
    │   │   │   ├── DataObjects.py
    │   │   │   ├── __init__.py
    │   │   │   ├── dataset.py
    │   │   │   ├── wav2mel.py
    │   │   │   └── wav_preprocessing.py
    │   │   ├── models
    │   │   │   ├── __init__.py
    │   │   │   └── dVecModel.py
    │   │   ├── speaker_encoder_manager.py
    │   │   └── utils
    │   │       ├── Trainer.py
    │   │       └── __init__.py
    │   ├── synthesizer
    │   │   ├── LICENSE.md
    │   │   ├── __init__.py
    │   │   ├── configs
    │   │   │   ├── __init__.py
    │   │   │   ├── config.py
    │   │   │   ├── hparams.py
    │   │   │   └── tacotron_config.yaml
    │   │   ├── data
    │   │   │   ├── __init__.py
    │   │   │   ├── audio.py
    │   │   │   ├── dataset.py
    │   │   │   └── preprocess.py
    │   │   ├── models
    │   │   │   ├── __init__.py
    │   │   │   └── tacotron.py
    │   │   ├── synthesize.py
    │   │   ├── synthesizer_manager.py
    │   │   ├── trainer.py
    │   │   └── utils
    │   │       ├── __init__.py
    │   │       ├── cleaners.py
    │   │       ├── logmmse.py
    │   │       ├── numbers.py
    │   │       ├── plot.py
    │   │       ├── symbols.py
    │   │       └── text.py
    │   ├── tts_module_manager.py
    │   └── vocoder
    │       ├── __init__.py
    │       ├── configs
    │       │   ├── __init__.py
    │       │   ├── config.py
    │       │   ├── hparams.py
    │       │   └── wavernn_config.yaml
    │       ├── data
    │       │   ├── __init__.py
    │       │   ├── dataset.py
    │       │   └── preprocess.py
    │       ├── models
    │       │   ├── __init__.py
    │       │   └── wavernn.py
    │       ├── utils
    │       │   ├── Trainer.py
    │       │   ├── __init__.py
    │       │   ├── audio.py
    │       │   ├── distribution.py
    │       │   └── gen_wavernn.py
    │       └── vocoder_manager.py
    └── utils
        └── __init__.py
```
## Installation 
Run `pip3 install .` from root directory.
## Datasets
We have implemented complete processing for LibraSpeech Dataset for Speaker Encoder, Synthesizer and Vocoder . One can download LibraSpeech dataset via this [link](hhttps://www.openslr.org/12 "link"). Also, for Speaker Encoder we implemented interface to use custom dataset. One needs to implement `PreprocessDataset` interface functions, `WavPreprocessor` interface functions, `WavPreprocessor` interface functions, or use implemented ones. 
## Configs
For baseline models the default configs will be loaded automatically. To change them one can use `update_config(...)` in `osms/common/configs/config.py`. To load default config one can use `get_default_<module_name>_config(...)`. Also, one can implement his own configs to use them for other models. 
## Managers 
To work with each three modules we implemented its own manager: `SpeakerEncoderManager`, `SynthesizerManager`, `VocoderManager`. As main manager we implemented `MustiSpreakerManager` which give access to all three managers. One can use them to inference the whole TTS model and train each modules separately or together. The example of usage can be found in notebook.
## Checkpoints
Baseline checkpoints are downloaded automatically in `checkpoints` directory with creation of 'MultiSpeaker' object. Also, one can use other checkpoints by simple updating of config (change ...CHECKPOINT_DIR_PATH, CHECKPOINT_NAME). 



 
## References
1. [Ye Jia, Y. Zhang, Ron J. Weiss, Q. Wang, Jonathan Shen, Fei Ren, Z. Chen,P. Nguyen, R. Pang, I. Lopez-Moreno, and Y. Wu.  Transfer learning from speaker verification to multispeaker text-to-speech synthesis,](https://arxiv.org/pdf/1806.04558.pdf) 
2. [Li  Wan,  Quan  Wang,  Alan  Papir,  and  Ignacio  Lopez  Moreno.   Generalized  end-to-end  loss  for  speaker  verification,](https://arxiv.org/pdf/1710.10467.pdf)
3. [Jonathan  Shen,  R.  Pang,  Ron  J.  Weiss,  M.  Schuster,  Navdeep  Jaitly,Z. Yang, Z. Chen, Yu Zhang, Yuxuan Wang, R. Skerry-Ryan, R. Saurous,Yannis Agiomyrgiannakis, and Y. Wu. Natural TTS synthesis by conditioning wavenet on mel spectrogram predictions,](https://arxiv.org/pdf/1712.05884.pdf)
4. [ Aaron  van  den  Oord,  S.  Dieleman,  H.  Zen,  K.  Simonyan,  Oriol  Vinyals,A. Graves, Nal Kalchbrenner, A. Senior, and K. Kavukcuoglu.  Wavenet:  Agenerative model for raw audio,](https://arxiv.org/pdf/1609.03499.pdf)
5. [Erica  Cooper,  Cheng-I  Lai,  Yusuke  Yasuda,  Fuming  Fang,  Xin  Wang,Nanxin  Chen,  and  Junichi  Yamagishi.    Zero-shot  multi-speaker  text-to-speech with state-of-the-art neural speaker embeddings.](https://arxiv.org/pdf/1910.10838.pdf)
