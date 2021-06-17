import torch
import torch.nn as nn
from ctcdecode import CTCBeamDecoder


def test(model, test_loader, criterion, device, transformer):
    """
    Evaluation for model.

    Args:
        model (nn.Module): Network to train
        test_loader (torch.utils.data.dataloader): DataLoader for test dataset
        criterion (nn.modules.loss): Loss function
        device (torch.device): Device (cpu or cuda)
    """

    model.eval()
    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths in test_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            output = output.transpose(0, 1)
            loss = criterion(output, targets, input_lengths, target_lengths)
            
            # Transpose back so that we can iterate over batch dimension
            output = output.transpose(0, 1)
            for log_probs, true_target, target_len in zip(output, targets, target_lengths):
                # guessed_target = greedy_decode(log_probs, transformer)
                guessed_target = timit_decode(log_probs, target_len, transformer)
                # guessed_target = beam_search_decode(log_probs, transformer)
                print('Guessed transcript: ' + guessed_target)
                print('True transcript: ' + transformer.target_to_text(true_target[:target_len]))
                print('-------------------------------')

def beam_search_decode(log_probs, transformer):

    # Using this ctc decoder: https://github.com/parlance/ctcdecode
    # Labels come from order specified in utils.py, _ represents blank
    labels = list("_ abcdefghijklmnopqrstuvwxyz'")

    # TODO: path to KenLM, alpha and beta values
    decoder = CTCBeamDecoder(
        labels,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=16,
        blank_id=0,
        log_probs_input=True
    )

    # input to decoder needs to be of shape BATCHSIZE x N_TIMESTEPS x N_LABELS
    # Currently doing single samples, so unsqueeze to create batch of 1
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(log_probs.unsqueeze(dim=0))
    # beam_results is of shape (num_batches, num_beams, time), so to get top beam, index [0][0]
    # cut it off by the appropriate length out_lens with same index
    return transformer.target_to_text(beam_results[0][0][:out_lens[0][0]])


def greedy_decode(log_probs, transformer):

    char_indices = torch.argmax(log_probs, dim=1)
    transcript = []
    blank_label = 0
    prev = None

    for idx in range(len(char_indices)):
        char = char_indices[idx].item()
        if char != blank_label:
            if char != prev:
                transcript.append(char)
        prev = char

    return transformer.target_to_text(transcript)


def timit_decode(log_probs, target_len, transformer):

    phon_indices = torch.argmax(log_probs, dim=1)
    return transformer.int_to_phon(phon_indices)
