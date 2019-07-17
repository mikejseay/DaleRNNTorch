import torch
import numpy as np
from scipy.stats import norm, zscore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using', device)

def sample_stimuli_continuous_samesign(batch_size, stimulus_mean, stimulus_noise, stimulus_duration):
    # Function to generate stimulus
    target = torch.ones(batch_size, device=device)  # -1 or 1
    stim = target.unsqueeze(-1) * stimulus_mean +\
           stimulus_noise * torch.randn(batch_size, stimulus_duration, device=device).unsqueeze(1)
    return target, stim

def sample_stimuli_continuous(batch_size, stimulus_mean, stimulus_noise, stimulus_duration):
    # Function to generate stimulus
    target = torch.sign(torch.randn(batch_size, device=device))  # -1 or 1
    stim = torch.mul(target.unsqueeze(-1),
                     stimulus_mean + \
                     stimulus_noise * torch.randn(batch_size, stimulus_duration, device=device)).unsqueeze(1)
    return target, stim

def sample_log_stimuli_continuous(batch_size, stimulus_noise, stimulus_duration):
    # Function to generate stimulus
    target = torch.sign(torch.randn(batch_size, device=device))  # -1 or 1
    intensities = 2 ** np.random.randint(1, 5, size=(batch_size,))
    sampled_means = torch.Tensor(intensities).to(device)
    stim = torch.mul(target, sampled_means).unsqueeze(-1).unsqueeze(-1) + \
           stimulus_noise * torch.randn(batch_size, stimulus_duration, device=device).unsqueeze(1)
    return target, stim

def sample_stimuli_brief(batch_size, stimulus_noise, stimulus_duration, on_duration):
    # Function to generate stimulus
    target = torch.sign(torch.randn(batch_size, device=device))  # -1 or 1
    intensities = 2 ** np.random.randint(1, 5, size=(batch_size,))
    square_wave = np.zeros((stimulus_duration,))
    square_wave[:on_duration] = 1
    multiple_square_waves = intensities.reshape(-1, 1) * square_wave
    noise = stimulus_noise * np.random.randn(*multiple_square_waves.shape)
    stim = target.unsqueeze(-1) * torch.Tensor(multiple_square_waves + noise).unsqueeze(1)
    return target, stim

def sample_stimuli_brief_pulse(batch_size, stimulus_mean, stimulus_noise, stimulus_duration, start_time, end_time):
    # Function to generate stimulus
    target = torch.ones(batch_size, device=device)  # -1 or 1
    # intensities = 2 ** np.random.randint(1, 5, size=(batch_size,))
    intensities = stimulus_mean * np.ones(batch_size)
    square_wave = np.zeros((stimulus_duration,))
    square_wave[start_time:end_time + 1] = 1
    multiple_square_waves = intensities.reshape(-1, 1) * square_wave
    noise = stimulus_noise * np.random.randn(*multiple_square_waves.shape)
    stim = target.unsqueeze(-1).unsqueeze(-1) * torch.Tensor(multiple_square_waves + noise).unsqueeze(1)
    return target, stim


t = np.arange(0, 80)
center1 = 20
center2 = 60
width = 5
bell1 = norm.pdf(t, center1, width)
bell2 = norm.pdf(t, center2, width)
# bells_fwd = zscore(np.vstack([bell1, bell2]), 1)
# bells_bwd = zscore(np.vstack([bell2, bell1]), 1)
bells_fwd = np.vstack([bell1, bell2])
bells_bwd = np.vstack([bell2, bell1])

def sample_sequential_bells(batch_size):
    order_bool = np.random.randint(0, 2, (batch_size,)).astype(bool)
    target = 2 * (order_bool - 0.5)
    # target = order_bool.astype(float)

    bellpair_lst = []
    for ob in order_bool:
        if ob:
            bellpair_lst.append(bells_fwd)
        else:
            bellpair_lst.append(bells_bwd)
    stim = np.stack(bellpair_lst)
    return torch.Tensor(target), torch.Tensor(stim)
