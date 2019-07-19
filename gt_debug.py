from functools import partial

import numpy as np
import scipy
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

from torch import nn
import torch
from models import RNN_Dale_Sum, RNN_Dale, RNN, print_trainable_parameters, extract_weights
from tasks import (sample_stimuli_continuous_samesign, sample_stimuli_continuous,
                   sample_log_stimuli_continuous, sample_stimuli_brief, sample_sequential_bells,
                   sample_stimuli_brief_pulse)
from training import basic_training, explicit_training
from plot_tools import plot_weights_square_ax, plot_weights, plot_phase_plane
from math_tools import print_daleness, relu, sigmoid, clip_0to1, unit_vector, angle_between

k = 1.1
W_miller = (4 + (2 / 7)) * np.array([[1, -k], [1, -k]])
W_millermean = np.fabs(W_miller).mean()

j0 = 2.1;
j = 0.4;
w0 = 1.11;
w = 0.9;
W_li = np.array([[j0, j, -1, -1],
                 [j, j0, -1, -1],
                 [w0, w, 0, 0],
                 [w, w0, 0, 0]])

stimulus_mean = 1
stimulus_noise = 0
stimulus_duration = 100
start_time = 50
end_time = 55

prop_exc = 0.5
alpha_dale = 0.5
train_alpha = True

n_iter = 1000
batch_size = 20
# n_iter = 2000
# batch_size = 10
learning_rate = 1e-3
n_inputs = 2

noisy_target = False
apply_noise = True

n_units = 2
gt_is_dale = True
input_type = 'order'  # brief or continuous or order
trained_type = 'sum'  # non, dale, sum

if n_units is 2:
    if gt_is_dale:
        model_gt = RNN_Dale(n_units, prop_exc, n_inputs, apply_noise=noisy_target)
        print_trainable_parameters(model_gt)
        model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 0])).unsqueeze(0))
        model_gt.Wrecdale = nn.Parameter(torch.Tensor(np.abs(W_miller)) / 10)  # recurrent weights - Dale
        model_gt.out = nn.Parameter(torch.Tensor(np.array([1, 1])).unsqueeze(-1))
        print_trainable_parameters(model_gt)
    else:
        model_gt = RNN(n_units, n_inputs, apply_noise=noisy_target)
        print_trainable_parameters(model_gt)
#         model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 0])).unsqueeze(0))
#         model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 1])).unsqueeze(0))
#         W = np.array([[j0 - w0, j - w], [j - w, j0 - w0]])
#         W = (4 + (2 / 7)) * np.array([[-k, 1], [1, -k]])
#         W = scipy.stats.ortho_group.rvs(n_units)
#         model_gt.Wrecnon = nn.Parameter(torch.Tensor(W / n_units))  # recurrent weights - Dale
#         model_gt.Wrecnon = nn.Parameter(torch.Tensor(W))  # recurrent weights - Dale
#         W = np.array([[ 0.34169018,  0.93981266], [ 0.93981266, -0.34169018]])  # recurrent weights - Dale
#         model_gt.Wrecnon = nn.Parameter(torch.Tensor(W))  # recurrent weights - Dale
#         model_gt.out = nn.Parameter(torch.Tensor(np.array([1, 1])).unsqueeze(-1))
elif n_units is 4:
    if gt_is_dale:
        model_gt = RNN_Dale(n_units, prop_exc, n_inputs, apply_noise=noisy_target)
        print_trainable_parameters(model_gt)
        model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 0, 0, 0])).unsqueeze(0))
        model_gt.Wrecdale = nn.Parameter(torch.Tensor(np.abs(W_li)))  # recurrent weights - Dale
        model_gt.out = nn.Parameter(torch.Tensor(np.array([1, 0, 0, 0])).unsqueeze(-1))
    else:
        model_gt = RNN(n_units, n_inputs, apply_noise=noisy_target)
        print_trainable_parameters(model_gt)
        model_gt.inp = nn.Parameter(torch.Tensor(np.array([1, 0, 0, 0])).unsqueeze(0))
        W = scipy.stats.ortho_group.rvs(n_units)
        model_gt.Wrecnon = nn.Parameter(torch.Tensor(W / n_units))  # recurrent weights - Dale
        model_gt.out = nn.Parameter(torch.Tensor(np.array([1, 1, 1, 1])).unsqueeze(-1))
else:
    if gt_is_dale:
        model_gt = RNN_Dale(n_units, prop_exc, n_inputs, apply_noise=noisy_target)
        print_trainable_parameters(model_gt)
    else:
        model_gt = RNN(n_units, n_inputs, apply_noise=noisy_target)
        #         W = scipy.stats.ortho_group.rvs(n_units)
        #         model_gt.Wrecnon = nn.Parameter(torch.Tensor(W / n_units))  # recurrent weights - Dale
        W = np.diag((1 + 0.2 / n_units) * np.ones(8)) - (0.2 / n_units)
        model_gt.Wrecnon = nn.Parameter(torch.Tensor(W))  # recurrent weights - Dale
        #         W_inp = np.ones(8)
        # #         W_inp = np.zeros(8)
        #         W_inp[0] = 1
        #         model_gt.inp = nn.Parameter(torch.Tensor(W_inp).unsqueeze(0))
        #         W_out = np.ones(8)
        #         W_out[0] = 1
        #         model_gt.out = nn.Parameter(torch.Tensor(W_out).unsqueeze(-1))
        print_trainable_parameters(model_gt)

w_output_gt, w_input_gt, w_rec_gt = extract_weights(model_gt)

print(w_input_gt)
print(w_rec_gt)
print(w_output_gt)

print_daleness(w_input_gt, w_rec_gt)

save_figs = False
save_prefix = 'LiDale'
f, ax_inp, ax_rec, ax_out = plot_weights(w_output_gt, w_input_gt, w_rec_gt)

# f.savefig(join(my_drive, 'Li-Ground-Truth-Matrices.tif'))
if save_figs:
    f.savefig(join(my_drive, save_prefix + '-Ground-Truth-Matrices.tif'))

if input_type is 'continuous':
    target_trash, stim_gt = sample_stimuli_continuous_samesign(1, 1, 0, 100)
    use_stim = stim_gt[0, 0, :]
    output, activity, track_outputs_gt = model_gt.forward_outputs(stim_gt)
elif input_type is 'brief':
    target_trash, stim_gt = sample_stimuli_brief_pulse(1, stimulus_mean, stimulus_noise, stimulus_duration,
                                                       start_time=start_time, end_time=end_time)
    use_stim = stim_gt[0, 0, :]
    output, activity, track_outputs_gt = model_gt.forward_outputs(stim_gt)
elif input_type is 'order':
    target_trash, stim_gt1 = sample_stimuli_brief_pulse(1, stimulus_mean, stimulus_noise, stimulus_duration,
                                                        start_time=20, end_time=25)
    target_trash, stim_gt2 = sample_stimuli_brief_pulse(1, stimulus_mean, stimulus_noise, stimulus_duration,
                                                        start_time=50, end_time=55)
    use_stim = torch.stack([stim_gt1[0, 0, :], stim_gt2[0, 0, :]], 0)
    output, activity, track_outputs_gt = model_gt.forward_outputs(stim_gt2)

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
if save_figs:
    f.tight_layout()
    f.savefig(join(my_drive, save_prefix + '-Ground-Truth.tif'))
if trained_type is 'sum':
    model = RNN_Dale_Sum(n_units, prop_exc, alpha_dale=alpha_dale, n_inputs=n_inputs, apply_noise=apply_noise)
elif trained_type is 'non':
    model = RNN(n_units, n_inputs=n_inputs, apply_noise=apply_noise)
elif trained_type is 'dale':
    model = RNN_Dale(n_units, prop_exc, n_inputs=n_inputs, apply_noise=apply_noise)

print_trainable_parameters(model)

w_output_start, w_input_start, w_rec_start = extract_weights(model)

if hasattr(model, 'alpha'):
    alpha_start = clip_0to1(model.alpha.detach().numpy())
    print(alpha_start)

print(w_input_start)
print(w_rec_start)
print(w_output_start)
# THIS RE-TRAINS - BE CAREFUL!!!!

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
