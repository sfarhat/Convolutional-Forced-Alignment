import torch
from utils import text_transformer

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
            for log_probs, true_target, target_len in zip(output, targets, target_lengths):
                guessed_target = greedy_decode(log_probs)
                print('Guessed transcript: ' + guessed_target)
                print('True transcript: ' + text_transformer.target_to_text(true_target[:target_len]))
                print('-------------------------------')

def greedy_decode(log_probs):

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

    return text_transformer.target_to_text(transcript)