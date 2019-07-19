from functools import partial

import numpy as np
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torch import nn
import torch
from models import RNN_Dale_Sum, RNN_Dale, RNN, print_trainable_parameters, extract_weights
from tasks import (sample_stimuli_continuous_samesign, sample_stimuli_continuous,
                   sample_log_stimuli_continuous, sample_stimuli_brief, sample_sequential_bells,
                   sample_stimuli_brief_pulse)
from training import basic_training, explicit_training
from plot_tools import plot_weights_square_ax, plot_weights
from math_tools import print_daleness, relu, sigmoid, clip_0to1

stimulus_mean = 1
stimulus_noise = 0
stimulus_duration = 100
start_time = 40
end_time = 45

# n_units = 2
n_units = 4
prop_exc = 0.5
alpha_dale = 0.5
train_alpha = True

n_iter = 1000
batch_size = 20
learning_rate = 1e-3

# target_stim_fn = partial(sample_stimuli_brief_pulse,
#                          stimulus_mean=stimulus_mean,
#                          stimulus_noise=stimulus_noise,
#                          stimulus_duration=stimulus_duration,
#                          on_duration=on_duration)f
n_inputs = 1

k = 1.1
W_miller = (4 + (2 / 7)) * np.array([[1, -k], [1, -k]])
W_millermean = np.fabs(W_miller).mean()

j0 = 2.1
j = 0.4
w0 = 1.11
w = 0.9
W_li = np.array([[j0, j, -1, -1],
                 [j, j0, -1, -1],
                 [w0, w, 0, 0],
                 [w, w0, 0, 0]])

noisy_target = False
apply_noise = True
use_dale = True
if use_dale:
    model_gt = RNN_Dale(n_units, prop_exc, n_inputs, apply_noise=noisy_target)
    print_trainable_parameters(model_gt)
    # model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 1])).unsqueeze(0))
    #     model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 0])).unsqueeze(0))
    model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 0, 0, 0])).unsqueeze(0))
    #     model_gt.Wrecdale = nn.Parameter(torch.Tensor(np.abs(W_miller)))  # recurrent weights - Dale
    model_gt.Wrecdale = nn.Parameter(torch.Tensor(np.abs(W_li)))  # recurrent weights - Dale
    #     model_gt.out = nn.Parameter(torch.Tensor(np.array([1, 1])).unsqueeze(-1))
    model_gt.out = nn.Parameter(torch.Tensor(np.array([1, 0, 0, 0])).unsqueeze(-1))
    print_trainable_parameters(model_gt)
else:
    model_gt = RNN(n_units, n_inputs, apply_noise=noisy_target)
    print_trainable_parameters(model_gt)
    #     model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 1])).unsqueeze(0))
    model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 0])).unsqueeze(0))
    k = 1.1
    W = (4 + (2 / 7)) * np.array([[-k, 1], [1, -k]])
    # W = W_millermean * (np.random.rand(2, 2) - 0.5) * 2
    model_gt.Wrecnon = nn.Parameter(torch.Tensor(W))  # recurrent weights - Dale
    model_gt.out = nn.Parameter(torch.Tensor(np.array([1, 1])).unsqueeze(-1))
    print_trainable_parameters(model_gt)

w_output_gt, w_input_gt, w_rec_gt = extract_weights(model_gt)

print(w_input_gt)
print(w_rec_gt)
print(w_output_gt)

print_daleness(w_input_gt, w_rec_gt)

f, ax_inp, ax_rec, ax_out = plot_weights(w_output_gt, w_input_gt, w_rec_gt)

# f.savefig(join(my_drive, 'Li-Ground-Truth-Matrices.tif'))

# target_trash, stim_gt = sample_stimuli_brief_pulse(1, stimulus_mean, stimulus_noise, stimulus_duration,
#                                                    start_time=start_time, end_time=end_time)
# use_stim = stim_gt[0, 0, :]
target_trash, stim_gt = sample_stimuli_continuous_samesign(1, 1, 0, 100)
use_stim = stim_gt[0, 0, :]
output, activity, track_outputs_gt = model_gt.forward_outputs(stim_gt)
use_output = track_outputs_gt.squeeze()

f, ax = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
ax[0].plot(use_stim.detach().cpu().numpy())
ax[0].set_xlabel(r'Time  ($\tau/5$)')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('Input Time Series')

ax[1].plot(use_output.detach().cpu().numpy())
ax[1].set_xlabel(r'Time  ($\tau/5$)')
ax[1].set_ylabel('Amplitude')
ax[1].set_title('Output Time Series')

# f.savefig(join(my_drive, 'MM-Ground-Truth.tif'))
# f.savefig(join(my_drive, 'Li-Ground-Truth.tif'))
model = RNN_Dale_Sum(n_units, prop_exc, alpha_dale=alpha_dale, n_inputs=n_inputs, apply_noise=apply_noise)
# model = RNN(n_units, n_inputs=n_inputs, apply_noise=apply_noise)
print_trainable_parameters(model)

w_output_start, w_input_start, w_rec_start = extract_weights(model)

if hasattr(model, 'alpha'):
    alpha_start = clip_0to1(model.alpha.detach().numpy())
    print(alpha_start)

print(w_input_start)
print(w_rec_start)
print(w_output_start)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

model, track_loss, track_alpha = \
    explicit_training(n_iter=n_iter,
                      batch_size=batch_size,
                      stim=use_stim,
                      target_series=use_output,
                      model=model,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      )
w_output, w_input, w_rec = extract_weights(model)
print(w_input)
print(w_rec)
print(w_output)

print_daleness(w_input, w_rec)

f, ax_inp, ax_rec, ax_out = plot_weights(w_output, w_input, w_rec)

# f.savefig(join(my_drive, 'Li-Dale-Sum-Matrices.tif'))
f, ax = plt.subplots()
ax.plot(track_loss)
ax.set_xlabel('iterations')
ax.set_ylabel('loss')
# f.savefig(join(my_drive, 'MM-DaleSum-Track-Loss.tif'))
# f.savefig(join(my_drive, 'Li-DaleSum-Track-Loss.tif'))
if hasattr(model, 'alpha'):
    f, ax = plt.subplots()
    ax.plot(track_alpha)
    ax.plot(alpha_dale * np.ones_like(track_alpha), 'k--')
    final_alpha = clip_0to1(model.alpha.detach().numpy())
    print(final_alpha)
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$\alpha$')
    ax.set_ylim(-0.1, 1.1)

    #     f.savefig(join(my_drive, 'MM-DaleSum-Track-Alpha.tif'))
    # f.savefig(join(my_drive, 'Li-DaleSum-Track-Alpha.tif'))