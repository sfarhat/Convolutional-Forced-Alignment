from torch import nn

class CollapsedCTCLoss(object):

    def __init__(self):
        pass

    def __call__(self, output, targets, input_lengths, target_lengths):
        # Output is shape (batch size, time, num_classes)
        # target_lengths is shape (batch size)
        # TODO: keeping input lengths for generalizability, change this when you have the chance

        loss = 0
        for log_probs, target, target_len in zip(output, targets, target_lengths):
            # pooling layer requires batch dimension, but we don't
            # TODO: try max pooling
            collapsed_probs = nn.AdaptiveAvgPool2d((target_len, None))(log_probs.unsqueeze(0)).squeeze()

            # Since log softmax in model, already doing Log-Likelihood, just subtract probs to get NLL
            for i in range(len(collapsed_probs)):
                prob_t = collapsed_probs[i]
                loss -= prob_t[int(target[i])]

        # Normalize loss by batch size
        return loss / output.shape[0]