import numpy as np
import time
import torch

def basic_training(n_iter, batch_size, target_stim_fn, model, loss_fn, optimizer,
                   full_output=False, full_time_series=False, target_outputs=None):

    # Placeholders
    track_loss = np.zeros(n_iter)
    track_alpha = np.zeros(n_iter)
    if full_output:
        track_targets = np.zeros(n_iter * batch_size)
        track_outputs = np.zeros(n_iter * batch_size)

    t0 = time.time()

    # Loop over iterations
    for i in range(n_iter):

        if (i + 1) % 20 == 0:  # print progress every 100 iterations
            print('%.2f%% iterations completed...loss = %.2f' % (100 * (i + 1) / n_iter, loss))

        target, stim = target_stim_fn(batch_size)

        # Run model
        # if full_output:
        #     outputs, hidden, track_outputs = model.forward_outputs(stim)
        # else:
        #     outputs, hidden = model.forward(stim)
        if full_time_series:
            outputs, hidden, track_outputs = model.forward_outputs(stim)
        else:
            outputs, hidden = model.forward(stim)

        # Compute loss
        loss = loss_fn(outputs, target)

        # Keep track of outputs and loss
        track_loss[i] = loss.item()
        # track_alpha[i] = model.alpha.detach().cpu().numpy()

        if full_output:
            track_targets[(i * batch_size) : ((i+1) * batch_size)] = target.detach().cpu().numpy()
            track_outputs[(i * batch_size) : ((i+1) * batch_size)] = outputs.detach().cpu().numpy()

        # Compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # clip alpha
        # model.alpha = torch.nn.Parameter(torch.clamp(model.alpha, -1, 1))

    if hasattr(model, 'alpha'):
        model.Wrecdale = torch.abs(model.Wrecdale)

    t1 = time.time()
    print('took', t1 - t0, 'seconds')

    if full_output:
        return model, track_loss, track_alpha, track_targets, track_outputs
    else:
        return model, track_loss, track_alpha


def explicit_training(n_iter, batch_size, stim, target_series, model, loss_fn, optimizer):

    # Placeholders
    track_loss = np.zeros(n_iter)
    track_alpha = np.zeros(n_iter)

    t0 = time.time()

    first_time = True

    # Loop over iterations
    for i in range(n_iter):

        if (i + 1) % 20 == 0:  # print progress every 100 iterations
            print('%.2f%% iterations completed...loss = %.2f' % (100 * (i + 1) / n_iter, loss))

        if len(stim.shape) == 1:
            batch_stim = stim.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        elif len(stim.shape) == 2:
            batch_stim = stim.unsqueeze(0).repeat(batch_size, 1, 1)
        batch_target_series = target_series.unsqueeze(0).repeat(batch_size, 1)

        outputs, hidden, track_outputs = model.forward_outputs(batch_stim)

        # Compute loss
        loss = loss_fn(track_outputs, batch_target_series)

        # Keep track of outputs and loss
        track_loss[i] = loss.item()
        if hasattr(model, 'alpha'):
            track_alpha[i] = model.alpha.detach().cpu().numpy()

        # Compute gradients
        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        # if first_time:
        #     loss.backward(retain_graph=True)
        #     first_time = False
        # else:
        #     loss.backward()

        # Update weights
        optimizer.step()

        # clip alpha
        # model.alpha = torch.nn.Parameter(torch.clamp(model.alpha, -1, 1))

    if hasattr(model, 'alpha'):
        model.Wrecdale = torch.nn.Parameter(torch.abs(model.Wrecdale))

    t1 = time.time()
    print('took', t1 - t0, 'seconds')

    return model, track_loss, track_alpha

