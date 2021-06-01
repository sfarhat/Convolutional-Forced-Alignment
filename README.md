# Forced Alignment via Convolutional Network Activation Maps

In Karaoke, lyrics are presented on screen in time with the appropriate portion of music. The better karaoke systems accomplish this in addition to providing a guide on inidividual word timings as well. The best ones take this even a step further by guiding the singer on a sound-by-sound basis.

## Background

Formally, this is the Forced Alignment problem: aligning an output transcipt with an input sequence song. It is actually a solved problem, but that's no fun. Because of *trends*, I want to approach it from a deep learning perspective rather than the classic methods involing Hidden Markov Models (HMM) or Dynamic Time Warping (DTW). A list of current old-school methods can be found [here](https://github.com/pettarin/forced-alignment-tools) (though it hasn't been updated since 2018).

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

**5/11**

Looking for a solution to the blank label problem, [this article](http://www.tbluche.com/ctc_and_blank.html) inspired me to train the network for 50 epochs, as it is apparenlty not uncommon for CTC to overwhelmingly prefer the blank label in early training. However, when doing this, something interesting happened: the loss steadily decreased until epoch 10, when it randomly exploded, but stabilized back down (albeit to a relatively higher loss than what it was pre-explosion). Thought we were in the clear? Wrong. At epoch 17, the same explosion happened again, except this time it reached NaN and never recovered :(.

How to fix this? Even though I was using ADAM, I had a hunch that my learning rate was too high (10e-4, which was deceivingly higher than what I thought it was in my mind, it's only .001! Boo me, and kinda the author of the paper I was reading from, for using 10 instead of 1 as the base). Instead, I just switched to a learning rate of 10e-5, or .0001. This did cause the loss to move slower, but it worked without any funny business! Over 50 epochs, it steadily decreased all the way down until it reach a loss of <1.

So, I managed to train the network, but did it help? Kinda. Using just a Greedy Decoder, I get the following result:

    Guessed transcript:  the dus were sufeor to ecs ail and the son had dispurst the miss and wasshetting asstron and clear lit in the forest when the travelers res oumnd their jurny
    True transcript: the dews were suffered to exhale and the sun had dispersed the mists and was shedding a strong and clear light in the forest when the travelers resumed their journey
    
As you can see, it is getting a phonetically similar sentence, but the spellings are atrocious. 

What can we do from here? 

* In CTC literature, it is well-known that using a Greedy Decoder is not the move as it ignores any kind of temporal structure of conditional probability information between characters. So the money move from here is to find/implement a Beam Search with a Language Model.
* Train on the dev set as well for fine-tuning. I'm not sure this is totally necessary since the goal isn't to create a perfect network, but just a working one for something else.

(Fun information: it took around 30 hours to train the whole thing. My room got very hot and the sound of a GPU fan running isn't really the greatest white noise for falling asleep.)

**5/19**

Given a semi-working model, I began working on implementing GRAD-CAM on it, however things turned sour quickly. First, I had to acknowledge the main problem: for a CNN-based ASR model, the output is a matrix of shape `(# labels, time)`, so a single "class label" that CAM likes to use is not well-defined to take the gradient with respect to. Each column of this output matrix specifies a separate distribution over the labels, and the output transcript is not easily interpretabable from this: it relies on the sum of all the probabilities of all possible alignments that correspond to the same transcript defined by CTC's assumptions.

This realization shed light on the larger problem with CTC. It's many-to-one (surjective) "collapsing" function (defined as beta in the original paper) is the main cause of 2 of CTC's weaknesses: 

1. It's huge reliance on a good decoding algorithm and language model to get a reasonable error rate
2. Bad interpretability

For example, consider the output transcript "cat" given an input sequence of length 5. There are 28 valid alignments for this that CTC will treat equally. But are they equal? Do `cccat` and `cattt` encode the same information? What about `--cat` vs `cat--`? The bigger question is: **what makes a good alignment?**

So, I propose 4 improvements:

1. Remove the blank symbol. I understand that this is, arguably, the main novelty of CTC, however its removal decreases the space of valid alignments. However, we still need a heuristic for an ideal alignment given that there are still multiple valid ones even without the blank symbol. All of these can be categorized by having repeat labels, whether they are intentional (feel) or not (catt). This motivates Proposal 2.

2. Use phonemes instead of characters. It removes the complexity of spelling that led to reliance on a decoder/LM. In addition, according to [Zhang et. al (2017)](https://arxiv.org/pdf/1701.02720.pdf), CNN-based models for ASR work better on phonetic information. To do this, we can use the TIMIT dataset. We also use the assumption that no English word has 2 neighbor repeat phonemes, which forces a 1-to-1 conversion from an alignment to the transcript. Why? Given that we are using the CTC method of collapsing alignments by 1) collapsing repeats then 2) removing blanks, repeats no longer exist via this Proposal, and blanks no longer exist by Proposal 1. So no collapsing is even necessary! To separate words (remember the original idea here is to look at how the model operates on each word), we will add a `<SPACE>` label in preprocessing using the time-annotations TIMIT provides.

3. It's great that we've removed the need for collapsing altogether, but won't the model have to do that anyway given the output dimensions? Correct. So, let's reduce the dimensionality of the output sequence from `(num_labels, time)` to `(num_labels, transcript length)`.

4. The caveat to Proposal 3 is that across batches, the transcript length differs, so instead of doing this in the model itself, it will be done in the loss function. The new loss function will use an Adaptive Average (or Max) Pooling for each sample within the batch to reduce the dimensions, then compute the Negative Log Likelihood of the singular sequence that is "correct".


Basically, all we are doing is forcibly decreasing the space of valid alignments, by only allowing 1 to be valid. That is, the actual transcript itself. No decoding necessary (except for a phonemes-to-word model). For example instead of `cat`'s 28 valid alignments with CTC in a length 5 input sequence (e.g. `c-a-t, caatt, -c-at`, etc), there is only 1: `k ae t`.

**5/29**

I integrated TIMIT, preprocessed to create the input/outputs desired, and created the custom loss function that I've named "Collapsed CTC". 

Did it work? After 20 epochs, it did **not**. The loss jumped around within the same range, showing no signs of decreasing. Maybe it needs hyperparameter tuning? Maybe I'm asking for too much from the model. For the sake of clarity, I did not follow the standard practice on training on 61 training labels then moving to the 39 test labels. Instead, I just used the 39 labels for training as well.

Perhaps I should return to what makes an ideal alignment and create a better objective loss function...