from functools import partial

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch

from models import RNN_Dale_Sum, RNN_Dale, RNN, print_trainable_parameters
from tasks import sample_stimuli_continuous, sample_log_stimuli_continuous, sample_stimuli_brief
from training import basic_training
from plot_tools import plot_weights_square_ax


stimulus_mean = 20
stimulus_noise = 0.5
stimulus_duration = 100
on_duration = 30

n_units = 20
prop_exc = 0.8
alpha_dale = 0.5
train_alpha = True

n_iter = 1000
batch_size = 20
learning_rate = 1e-4

# model = RNN(n_units)
model = RNN_Dale(n_units, prop_exc)
print_trainable_parameters(model)

# target_stim_fn = partial(sample_stimuli_continuous,
#                          stimulus_mean=stimulus_mean,
#                          stimulus_noise=stimulus_noise,
#                          stimulus_duration=stimulus_duration)
target_stim_fn = partial(sample_log_stimuli_continuous,
                         stimulus_noise=stimulus_noise,
                         stimulus_duration=stimulus_duration)
# target_stim_fn = partial(sample_stimuli_brief,
#                          stimulus_noise=stimulus_noise,
#                          stimulus_duration=stimulus_duration,
#                          on_duration=on_duration)

f, ax = plt.subplots()
targets, stim = target_stim_fn(5, stimulus_noise=stimulus_noise, stimulus_duration=stimulus_duration)
# targets, stim = sample_stimuli_brief(5, stimulus_noise, stimulus_duration, on_duration)
for t, s in zip(targets, stim):
    ax.plot(s.numpy().T, label='class %i' % t)
ax.legend()

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.CrossEntropyLoss()

model, track_loss, track_alpha = basic_training(n_iter=n_iter,
                                                batch_size=batch_size,
                                                target_stim_fn=target_stim_fn,
                                                model=model,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer
                                                )

target, stim = target_stim_fn(2)
output, activity, example_outputs = model.forward_outputs(stim)

example_stim = stim.detach().cpu().numpy()
f, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].plot(example_stim.T)
ax[1].plot(example_outputs.T)

if hasattr(model, 'alpha'):
    alpha_trained = model.alpha.detach().cpu().numpy()
else:
    alpha_trained = 1
# alpha_trained = model.alpha.detach().cpu().numpy()
# print(alpha_trained)

f, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].plot(track_loss)
ax[1].plot(track_alpha)

Sdale = model.Sdale.detach().cpu().numpy()
Wdale_pos = model.Wrecdale.detach().cpu().numpy()
w_dale = np.matmul(Wdale_pos, Sdale)
if hasattr(model, 'Wrecnon'):
    w_non = model.Wrecnon.detach().cpu().numpy()
    w_sum = alpha_trained * w_dale + (1 - alpha_trained) * w_non

f, ax = plt.subplots(1, 3, figsize=(17, 5))
ax = ax.ravel()

plot_weights_square_ax(w_dale, ax[0])
if hasattr(model, 'Wrecnon'):
    plot_weights_square_ax(w_non, ax[1])
    plot_weights_square_ax(w_sum, ax[2])

# eigen value spectra

eig_dale, eigv_dale = np.linalg.eig(w_dale)
if hasattr(model, 'Wrecnon'):
    eig_non, eigv_non = np.linalg.eig(w_non)

f, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].scatter(eig_dale.real, eig_dale.imag)
if hasattr(model, 'Wrecnon'):
    ax[1].scatter(eig_non.real, eig_non.imag)

print(eig_dale.real.max())
if hasattr(model, 'Wrecnon'):
    print(eig_non.real.max())
