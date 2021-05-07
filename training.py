import torch

def train(model, train_loader, criterion, optimizer, epoch, device):
    """
    Train the model for 1 epoch.

    Args:
        model (nn.Module): Network to train
        train_loader (torch.utils.data.dataloader): DataLoader for training dataset
        criterion (nn.modules.loss): Loss function
        optimizer (torch.optim): Optimizer
        epoch (int): Current epoch #
        device (torch.device): Device (cpu or cuda)
        writer (torch.utils.tensorboard.writer): Tensorboard debugging info
    """

    model.train()
    data_len = len(train_loader.dataset)
    for batch_num, data in enumerate(train_loader): 
        inputs, input_lengths, targets, target_lengths = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # this is shape (batch size, time, num_classes)
        output = model(inputs)
        # Debug note: breakpoint here for expression: torch.isnan(output).any()

        # output passed in should be of shape (time, batch size, num_classes)
        output = output.transpose(0, 1)
        loss = criterion(output, targets, input_lengths, target_lengths)
        loss.backward()
        # Debug note: breakpoint here for expression: torch.isnan(loss).any()

        optimizer.step() 

        if batch_num % 10 == 0 or batch_num == data_len:
            # writer.add_scalar("Loss/train", loss.item(), epoch * data_len + batch_num * len(inputs))
            # writer.flush()     

            # len(inputs) is batch size, data_len is total size of samples in dataset, len(train_loader) is number of batches

            print(f"Train Epoch: {epoch} [{(batch_num+1) * len(inputs)}/{data_len} ({100. * (batch_num+1) / len(train_loader):.2f}%)]\tLoss: {loss.item():.6f}")