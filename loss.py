import torch.nn as nn

class SequentialNLLLoss(object):

    def __init__(self):
        pass

    def __call__(self, output, targets, target_lengths):
        # Output is shape (batch x time x class)
        # target_lengths is shape (batch size)

        loss = 0
        N = 0
        for log_probs, target, target_len in zip(output, targets, target_lengths):
            # log_probs is padded within batch, so use target_len

            # Since log softmax in model, already doing Log-Likelihood, just subtract probs to get NLL
            for t in range(target_len):
                prob_t = log_probs[t]
                loss -= prob_t[int(target[t])]
                N += 1

        # Normalize loss by batch size
        return loss / N

def calculate_loss(loss_fn, output, targets, input_lengths, target_lengths):
    """Uses correct loss function

    Args:
        loss_fn (object): Loss function to be used, can choose between nn.CTCLoss and Modified NLLLoss
        output (Tensor): Output of network of shape (batch x time x class)
        targets (Tensor): Ground truth target transcript pre-padding 
        input_lengths (list): List of lengths of input spectrograms pre-padding
        target_lengths (list): List of lengths of target transcripts pre-implicit-padding that network does 

    Raises:
        Exception: If valid loss function not provided, an Exception will be thrown

    Returns:
        [float]: Loss value
    """

    if isinstance(loss_fn, nn.CTCLoss):
        # CTC expects shape (time x batch x class)
        output = output.transpose(0, 1)
        return loss_fn(output, targets, input_lengths, target_lengths)
    elif isinstance(loss_fn, SequentialNLLLoss):
        return loss_fn(output, targets, target_lengths) 
    else:
        raise Exception("Not a valid loss function, please choose between CTCLoss and Modified NLLLoss")