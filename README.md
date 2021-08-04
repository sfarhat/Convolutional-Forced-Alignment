# Convolutional-Forced-Alignment

## Background

Generating word-timestamps for an audio file (aka generating a **Forced Alignment** has had a long and rich history in the NLP and AI communities as a whole. In fact, one of motivating uses for Hidden Markov Models (HMM) was Automatic Speech Recognition (ASR). I, on the other hand, was motivated by a desire to create my own Karaoke system, which I did in [this project](https://github.com/sfarhat/Karaoke/tree/main).

Classic Forced Alignment methods were based on a Dynamic Programming-esque algorithm called Dynamic Time Warping (DTW) or Hidden Markov Models in tandem with Gaussian Mixture Models (HMM-GMM). The latter is still very successful (though the GMM may be substituted with a Deep Neural Network instead), though modern methods focus on *recurrent* models, i.e. Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), and most recently *attention*-based models such as Transformers. All of these, however, are computationally expensive due to the temporal nature of the models

## Method

In this project, I attempted to massage a Convolutional Neural Network to generate a forced alignment. While work has been done on ASR with CNNs, a forced alignment wasn't possible due to the chosen, yet ubiquitous, loss function: Connectionist Temporal Classification (CTC). I was successful in doing this, creating a purely **Convolutional Forced Aligner** with an average alignment error rate of `67 ms`. 

To see my thought process as well as the ups and downs I experienced making this, you can read the blog-style [story.md](https://github.com/sfarhat/Convolutional-Forced-Alignment/blob/main/story.md). 

If you want a detailed technical report with diagrams and equations, you can read [Convolutional_Forced_Alignment.pdf](https://github.com/sfarhat/Convolutional-Forced-Alignment/blob/main/Convolutional_Forced_Alignment.pdf).

## Requirements

All requirements can be found in [environment.yml](https://github.com/sfarhat/Convolutional-Forced-Alignment/blob/main/environment.yml) and can be loaded in via Anaconda. However, if you wish to use an alternate package management system/environment manager, here are the packages + versions I used:

    1. python==3.9.1
    2. pytorch==1.7.1 + CUDA 11.0
    3. torchaudio==0.7.2
    4. nltk

## Use

After cloning this repository, all available options are in [config.py](https://github.com/sfarhat/Convolutional-Forced-Alignment/blob/main/config.py):

1. `DATASET_DIR`: The path to the training/test dataset. As of now, it only supports 'TIMIT' or 'Librispeech'.
2. `CHECKPONIT_DIR`: The path to the directory where you would like the model checkpoints to be saved to/loaded from.
3. `ADAM_lr`: The learning rate for the ADAM optimizer.
4. `batch_size`: The batch size for training.
5. `SGD_lr`, `SGD_l2_penalty`: Learning rate and weight penalization parameters for Stochastic Gradient Descent, which is used on the validation set.
6. `weights_init_a`, `weights_init_b`: The range of values which the model weights should be initialized uniformly between.
7. `epochs`: Number of epochs to train for.
8. `activation`: For flexibility, an activation chosen bewteen `relu`, `prelu`, and `maxout` can be selected. This will apply to all layers in the network.
9. `start_epoch`: The epoch to start at for traning or inference. Must have the appropriate `.pt` model weights saved beforehand to do this. This number should be +1 whatever the filename states.
10. `mode`: Choose bewteen `train`, `test`, `cam`, `align`, `test-align`. The details of each are desribed in the next section.
11. `dataset`: The dataset the model is based off of. Choose between `TIMIT` or `Librispeech`.
12. `sample_path`: If you wish to generate a forced alignmentt (mode `align`) or class activation map (mode `cam`) for an individual file, provide the path to the respective audio file here. Should be used in conjunction with `sample_transcript`.
13. `sample_transcript`: If you wish to generate a forced alignmentt (mode `align`) or class activation map (mode `cam`) for an individual file, provide the path to the text transcript file here. Should be used in conjunction with `sample_path`.
14. `timit_sample_path`: Can be used in place of `sample_path` and `sample_transcript`. Since TIMIT provides ground truth word-timings, set this path to the desired TIMIT sample (e.g. timit/data/TRAIN/DR4/MESG0/SX72) in `cam` or `align` mode to also see the alignment error.
15. `model`: For now, only `zhang` is valid.
16. `cam_phonemes`: Used in `cam` mode. Choose which phonemes to find the activations for (e.g. [1, 2, 10]).
17. `cam_word`: Used in `align` mode. Choose which word to find the activations for (e.g. 1).

To run the script, simply run `python main.py`.

### Modes

Note: For almost all of the modes, you should be using `dataset: TIMIT`, but `Librispeech` is left as an option *only* for `train` or `test` if you want to train the CNN on the dataset and see the results. (It is nothing novel, just normal training with CTC loss and decoding).

- `train`: Train the `model` on the `dataset` for `(epochs - start_epochs)` epochs with the hyperparameters `ADAM_lr`, `batch_size`, `SGD_lr`, `SGD_l2_penalty`, `weights_init_a`, `weights_init_b`, and `activation`. Every epoch, it will save a model checkpoint in the `CHECKPOINT_DIR`. For `TIMIT`, it will automatically find the pre-partitioned TRAIN dataset.

- `test`: Loads in a pre-trained model from epoch `start_epoch` (assuming the checkpoint exists). Tests the accuracy of the `model` on the `dataset`. For `TIMIT`, it will automatically find the pre-partioned TEST dataset. After 15 epochs, my model achieved a Phoneme Error Rate of `22%` on the TIMIT test set.

- `cam`: Loads in a pre-trained model from epoch `start_epoch` (assuming the checkpoint exists). Given a sample in `sample_path` or `timit_sample_path`, this will generated a class activation map via the GRAD-CAM method. It will show the activations for all phonemes listed in `cam_phonemes`; any invalid ones will be ignored.

- `align`: Loads in a pre-trained model from epoch `start_epoch` (assuming the checkpoint exists). Given a sample/transcript combo either via `sample_path` and `sample_transcript` or `timit_sample_path`, this will generate and print the Forced Alignment (in seconds). If a `timit_sample_path` is provided, it will also provide the alignment error. If a `cam_word` is provided, it will also generate the class activations for all phonemes in the desired word.

- `test-align`: Loads in a pre-trained model from epoch `start_epoch` (assuming the checkpoint exists). This will compute the average Alignment Error (in seconds) for the entire TIMIT dataset given our method. After 15 epochs, my model achieved an average Alignment Error of `67 ms`.
