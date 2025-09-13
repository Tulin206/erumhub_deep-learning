import time
import torch
from helper_evaluation import compute_accuracy


def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, optimizer,
                device, logging_interval=50,
                scheduler=None,
                scheduler_on='valid_acc'):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## forward pass + backward pass + update model parameters
            logits = model(features)   # forward pass

            """CrossEntropyLoss is for multi-class classification (with logits).
            It automatically applies softmax internally.
            Suppose the output logits for one MNIST digit are: logits = [2.0, 1.0, 0.1]
            CrossEntropyLoss will turn them into probabilities automatically: Softmax(2.0,1.0,0.1)=[0.65,0.24,0.11]"""
            loss = torch.nn.functional.cross_entropy(logits, targets)   # compute loss

            optimizer.zero_grad()    # zero the gradients (clear old grads) before running the backward pass
            loss.backward()   # backward pass (compute gradients: d(loss)/d(params))

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()  # update model weights using the gradients

            # ## LOGGING the minibatch loss
            """Save the current batch’s loss (as a plain Python number) into a list.
            Later you can plot the curve of training loss.
            Every logging_interval batches, print progress:
                - Which epoch and batch you’re on
                - Current loss value
            Example: Epoch: 001/100 | Batch 0100/0210 | Loss: 0.5573"""
            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        model.eval()   # set the model to evaluation mode--turn off training-specific behavior (Dropout, BatchNorm)
        with torch.no_grad():  # stop tracking gradients (save memory and speeds up during inference)
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '                # Shows how well the model is doing after each epoch
                  f'| Validation: {valid_acc :.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        elapsed = (time.time() - start_time)/60         # Compute minutes since training began
        print(f'Time elapsed: {elapsed:.2f} min')       # Print progress so you know how long training is taking
        
        if scheduler is not None:

            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')
        

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list
