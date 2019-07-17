from functools import partial

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
from models import RNN_Dale_Sum, RNN_Dale, RNN, print_trainable_parameters
from tasks import (sample_stimuli_continuous_samesign, sample_stimuli_continuous,
                   sample_log_stimuli_continuous, sample_stimuli_brief, sample_sequential_bells)
from training import basic_training
from plot_tools import plot_weights_square_ax
from math_tools import print_daleness, relu

stimulus_mean = 20
stimulus_noise = 0.5
stimulus_duration = 100
on_duration = 30

n_units = 2
prop_exc = 0.5
# n_units = 5
# prop_exc = 0.8
# n_units = 10
# prop_exc = 0.8
alpha_dale = 0.5
train_alpha = True

n_iter = 800
batch_size = 20
learning_rate = 1e-3

target_stim_fn = sample_sequential_bells
n_inputs = 2

# target_stim_fn = partial(sample_stimuli_continuous_samesign,
#                          stimulus_mean=stimulus_mean,
#                          stimulus_noise=stimulus_noise,
#                          stimulus_duration=stimulus_duration)
# target_stim_fn = partial(sample_stimuli_continuous,
#                          stimulus_mean=stimulus_mean,
#                          stimulus_noise=stimulus_noise,
#                          stimulus_duration=stimulus_duration)
# target_stim_fn = partial(sample_log_stimuli_continuous,
#                          stimulus_noise=stimulus_noise,
#                          stimulus_duration=stimulus_duration)
# target_stim_fn = partial(sample_stimuli_brief,
#                          stimulus_noise=stimulus_noise,
#                          stimulus_duration=stimulus_duration,
#                          on_duration=on_duration)
# n_inputs = 1

model = RNN(n_units, n_inputs)
# model = RNN_Dale(n_units, prop_exc, n_inputs)

# model.forward = model.forward_outputs
print_trainable_parameters(model)

# f, ax = plt.subplots()
# targets, stim = target_stim_fn(5, stimulus_noise=stimulus_noise, stimulus_duration=stimulus_duration)
# # targets, stim = sample_stimuli_brief(5, stimulus_noise, stimulus_duration, on_duration)
# for t, s in zip(targets, stim):
#     ax.plot(s.numpy().T, label='class %i' % t)
# ax.legend()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.CrossEntropyLoss()

full_output = True
# model, track_loss, track_alpha = basic_training(n_iter=n_iter,
#                                                 batch_size=batch_size,
#                                                 target_stim_fn=target_stim_fn,
#                                                 model=model,
#                                                 loss_fn=loss_fn,
#                                                 optimizer=optimizer
#                                                 )
model, track_loss, track_alpha, track_targets, track_outputs_progress = \
    basic_training(n_iter=n_iter,
                   batch_size=batch_size,
                   target_stim_fn=target_stim_fn,
                   model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   full_output=full_output,
                   )

target, stim = target_stim_fn(80)
output, activity, track_outputs = model.forward_outputs(stim)
example_targets = target.detach().cpu().numpy()
example_outputs = track_outputs.detach().cpu().numpy() # batch_size x time
example_stim = stim.detach().cpu().numpy()

do_plots = True
if do_plots:
    if n_inputs > 1:
        target_types = np.unique(example_targets)
        AB_trials = example_targets == target_types[0]
        BA_trials = example_targets == target_types[1]
        f, ax = plt.subplots()
        ax.plot(example_outputs[AB_trials, :].mean(0))
        ax.plot(example_outputs[BA_trials, :].mean(0))

        f, ax = plt.subplots(1, 3, figsize=(17, 5))
        ax[0].plot(example_stim[:, 0, :])
        ax[1].imshow(example_stim[:, 1, :])
        ax[2].imshow(example_outputs)
    else:
        f, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].plot(example_stim.squeeze().T)
        ax[1].plot(example_outputs.T)

    if full_output:
        f, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].hist(track_targets[-2000:])
        ax[1].hist(track_outputs_progress[-2000:])

    if hasattr(model, 'alpha'):
        alpha_trained = model.alpha.detach().cpu().numpy()
    else:
        alpha_trained = 1
    # alpha_trained = model.alpha.detach().cpu().numpy()
    # print(alpha_trained)

    f, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].plot(track_loss)
    ax[1].plot(track_alpha)

    w_output = model.out.detach().cpu().numpy().T
    if hasattr(model, 'Wrecnon'):
        print('seems unconstrained')
        w_rec = model.Wrecnon.detach().cpu().numpy()
        w_input = model.inp.detach().cpu().numpy().T
    elif hasattr(model, 'Wrecdale'):
        print('seems like dale')
        w_recdale = relu(model.Wrecdale.detach().cpu().numpy())
        s_dale = model.Sdale.detach().cpu().numpy()
        w_rec = np.matmul(w_recdale, s_dale)
        w_input = relu(model.inp.detach().cpu().numpy().T)

    print(w_input)
    print(w_rec)
    print(w_output)

    print_daleness(w_input, w_rec)

    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    ax_rec = fig.add_subplot(grid[:-1, 1:], xticklabels=[], yticklabels=[])
    ax_inp = fig.add_subplot(grid[:-1, 0], xticklabels=[], yticklabels=[])
    ax_out = fig.add_subplot(grid[-1, 1:], xticklabels=[], yticklabels=[])

    plot_weights_square_ax(w_input, ax_inp, plot_colorbar=True)
    plot_weights_square_ax(w_rec, ax_rec, plot_colorbar=True)
    plot_weights_square_ax(w_output, ax_out, plot_colorbar=True)

    # eigen value spectra

    eig_rec, eigv_rec = np.linalg.eig(w_rec)

    f, ax = plt.subplots(1, 1, figsize=(11, 5))
    ax.scatter(eig_rec.real, eig_rec.imag)

    print(eig_rec.real.max())
