import torch
from utils import target_to_text

def test(model, test_loader, criterion, device):
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
            
            # TODO: Beam search decoding algo instead of greedy
            # Transpose back so that we can iterate over batch dimension
            output = output.transpose(0, 1)
            for log_probs, target_len in zip(output, target_lenghs):
                guessed_target = greedy_decode(torch.argmax(log_probs, dim=1), target_len)

def greedy_decode(char_indices, target_len):

    transcript = []
    blank_seen = False
    prev = None
    for idx in range(target_len):
        char_idx = char_indices[idx]
        if char_idx == prev and not blank_seen:
           continue
        elif char_idx == 0:
            blank_seen = True
        else:
            transcript.append(char_idx)
            blank_seen = False

    return target_to_text(transcript)