# OSM: One-Shot Multi-speaker

## Problem Statement
One-Shot Multi-Speaker Text-to-Speech (OS MS TTS) systems are aimed to transform text into speech with voice determined by small single sample. The main problem here is to reproduce the new unseen voice without retraining the network. There is an approach with three main stages which is used to solve this problem.
The unique for each voice speaker embeddings, which reveal the voice characteristics, are generated at the first stage (Speaker Encoder).
At the second stage (Synthesizer) the text is transformed to mel-spectrogram using previously obtained embeddings. 
Finally, the speech is reproduced from the mel-spectrogram with the Vocoder. But there is lack of implementations with these three parts together. So, the goal of our project is to create a flexible framework to combine this parts and produce a functionality to replace methods in each part.

## Main Challenges
By now we see the following main challenges:
- The solution to our problem consists of three subtasks, which already have a great solutions. Therefore, the existing solutions for OS MS TTS are essentially a compilation of solutions for these individual problems, for which there are many ready-made and well-implemented solutions. The main challenge is to made the framework flexible and ensure the compatibility of individual parts.
-The methods used in each subtask differ in the set of parameters and the nature of the algorithm. Therefore, it will be quite difficult to provide them with a single API.


## Overview


## Setup
