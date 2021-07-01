class ModifiedNLLLoss(object):

    def __init__(self):
        pass

    def __call__(self, output, targets, input_lengths, target_lengths):
        # Output is shape (time, batch_size, num_classes), inherited from CTC
        output = output.transpose(0, 1)
        # target_lengths is shape (batch size)
        # TODO: keeping input lengths for generalizability, change this when you have the chance

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