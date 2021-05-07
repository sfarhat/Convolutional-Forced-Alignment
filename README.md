# Forced Alignment via Convolutional Network Activation Maps

In Karaoke, lyrics are presented on screen in time with the appropriate portion of music. The better karaoke systems accomplish this in addition to providing a guide on inidividual word timings as well. The best ones take this even a step further by guiding the singer on a sound-by-sound basis.

## Background

Formally, this is the Forced Alignment problem: aligning an output transcipt with an input sequence song. It is actually a solved problem, but that's no fun. Because of ~trends~, I want to approach it from a deep learning perspective rather than the classic methods involing Hidden Markov Models (HMM) or Dynamic Time Warping (DTW). A list of current old-school methods can be found [here](https://github.com/pettarin/forced-alignment-tools) (though it hasn't been updated since 2018).

There are 2 ideas that I wish to focus on to get this done: activation maps and attention.

## Activation Maps

In convolutional networks, with image problems, the concept of activation maps has been shown to effectively show where in the input the network is looking to make its decisions. (cite MAP and GRAD-MAP here) Can this be extended to our use case?

Well, we don't actually use raw waveforms when working with audio usually; instead, we use features such as spectograms or MFCC (cite these wikis), which are images! So we can use convolutional networks on audio as well, which allows us to use these activation maps. 

Now, all we have to do is find an appropriate CNN that does End-to-End Automatic Speech Recognition (ASR), implement it, and see if the heat maps are doing their job to find at what times in the audio we should be looking at.

Here is the model I am using by [Zhang et. al (2017)](https://arxiv.org/pdf/1701.02720.pdf).

Boom, timestamps. And a new idea! (I think)

## Attention (prob not going to explore this)

Using CNNs for audio is a weird choice honestly though. The standard is to use some sort of Recurrent Model, the forefather being RNNs (LSTMs to be specific) (cite this), later improved to the Listen, Attend, Spell (LAS) (cite this) network and its built-upon variants. These have shown superior performance *however* at a much higher computational cost relative to CNNs.

At the heart of these recurrent models is the concept of **attention**, which literally tells the network at what point in the sequential input it should look to process the next step in the output. (tbh this seems like it should be related to activation maps being the analogue to attention in a CNN? look into this). LAS has a very straightforward attention mechanism built-in which gives us exactly what we want.

Is this a new idea? No. I am re-building the wheel here. Will it be fun? Maybe, but I don't have a server of GPUs to train it on, so maybe finding a pre-trained model would be nice. Will it be a good learning experience? Absolutely.

## Progress

**3/15**

The appropriate ASR CNN is implmented with CUDA functionality, but it only learns to output the blank label. This is probably due to not having been trained long enough. While I would, there are 2 outstanding issues:

* The loss clips to NaN. When using maxout, this occurs within the 1st epoch, whereas with PReLU, this occured during the 7th epoch. Training must be stopped then.
* My single RTX 2060 isn't capable of doing the full batchsize of 20 as cited in the paper, so instead I used 3.

Both of these can be remedied by more compute, so progress is stalled until then and a pretrained network is probably a good step from here.

**5/7**

The NaN problem has been solved! AFter moving the log Mel Spectrograms instead of MFCC features, it was observed that NaN loss was being caused by NaNs in the output, which in turn was being caused by NaN's in the input features. This was because of a simple overlooked problem: log(0) = NaN. So when any Mel Spectrogram had a 0 in it, the entire model would break!

The solution: adding a small constant offset to the waveform before taking the log.

Did this fix everything? No. While we were now able to train for 10 epochs and get a steadily decreasing loss, the model still overwhelmingly predicts the blank label. I think digging deeper into CTCLoss and how the weights of the model are being updated will shed some light as to why this is happening.
