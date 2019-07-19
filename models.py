from torch import nn
import torch
import numpy as np
from math_tools import relu, sigmoid, clip_0to1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using', device)

hard_tanh_0to1 = nn.Hardtanh(0, 1)
global_noise_level = 0.01

def print_trainable_parameters(model):
    for name, param in model.state_dict().items():
        print(name, param.size())

def extract_weights(model):

    w_output = model.out.detach().cpu().numpy().T
    if hasattr(model, 'alpha'):
        print('seems like competitive sum of unconstrained and Dale')
        alpha = clip_0to1(model.alpha.detach().numpy())
        w_recnon = model.Wrecnon.detach().cpu().numpy()
        w_recdale = np.abs(model.Wrecdale.detach().cpu().numpy())
        s_dale = model.Sdale.detach().cpu().numpy()
        w_rec = alpha * np.matmul(w_recdale, s_dale) + (1 - alpha) * w_recnon
        w_input = np.abs(model.inp.detach().cpu().numpy().T)
    elif hasattr(model, 'Wrecdale'):
        print('seems like dale')
        w_recdale = np.abs(model.Wrecdale.detach().cpu().numpy())
        s_dale = model.Sdale.detach().cpu().numpy()
        w_rec = np.matmul(w_recdale, s_dale)
        w_input = np.abs(model.inp.detach().cpu().numpy().T)
    elif hasattr(model, 'Wrecnon'):
        print('seems unconstrained')
        w_rec = model.Wrecnon.detach().cpu().numpy()
        w_input = model.inp.detach().cpu().numpy().T

    return w_output, w_input, w_rec


class RNN(nn.Module):
    def __init__(self, n_units, n_inputs=1, apply_noise=True):
        super(RNN, self).__init__()

        self.n_units = n_units  # number of neurons

        # self.inp = nn.Linear(1, n_units, bias=False).to(device)  # input weights
        self.inp = nn.Parameter(torch.rand(n_inputs, n_units, device=device))  # input weights

        self.Wrecnon = nn.Parameter((torch.rand(n_units, n_units, device=device) * 2 - 1) \
                                    / n_units)  # recurrent weights - non-Dale

        # TODO: try doing abs(randn)
        # self.out = nn.Linear(n_units, 1, bias=False).to(device)  # output weights
        self.out = nn.Parameter(torch.rand(n_units, 1, device=device))  # output weights

        self.tau = 5
        self.prop_old_act = 1 - 1 / self.tau
        self.prop_new_act = 1 / self.tau

        if apply_noise:
            self.noise_constant = np.sqrt(2 * global_noise_level / self.tau)
        else:
            self.noise_constant = 0


    def forward(self, inputs):

        # Initialize network state as zeros
        activity = torch.zeros(inputs.size(0), self.n_units, device=device)

        # pre-calculate noise for all time-points
        noise = self.noise_constant * \
                torch.randn(inputs.size(2), inputs.size(0), self.n_units, device=device)

        # Run the input through the network
        for i in range(inputs.size(2)):
            output, activity = self.step(inputs[:, :, i], activity, noise[i, :, :])
        return output.squeeze(), activity

    def step(self, input_ext, activity, noise):

        # non-linear
        activity = self.prop_old_act * activity + \
                   self.prop_new_act * (torch.matmul(self.Wrecnon, torch.relu(activity.unsqueeze(-1))).squeeze() +
                                        torch.matmul(input_ext, self.inp)) + noise

        # linear
        # activity = self.prop_old_act * activity + \
        #            self.prop_new_act * (torch.matmul(self.Wrecnon, activity.unsqueeze(-1)).squeeze() +
        #                                 torch.matmul(input_ext, self.inp)) + noise


        output = torch.matmul(activity, self.out)
        # output = torch.matmul(activity, torch.relu(self.out))
        return output, activity

    def forward_outputs(self, inputs):

        # collect outputs
        track_outputs = torch.zeros(inputs.size(0), inputs.size(2), device=device)

        # Initialize network state as zeros
        activity = torch.zeros(inputs.size(0), self.n_units, device=device)
        # activity = torch.Tensor(np.array([1, 0])).to(device).unsqueeze(0)

        # pre-calculate noise for all time-points
        noise = self.noise_constant * \
                torch.randn(inputs.size(2), inputs.size(0), self.n_units, device=device)

        # Run the input through the network
        for i in range(inputs.size(2)):
            output, activity = self.step(inputs[:, :, i], activity, noise[i, :, :])
            track_outputs[:, i] = output.squeeze()

        return output.squeeze(), activity, track_outputs


class RNN_Dale(nn.Module):
    def __init__(self, n_units, prop_exc, n_inputs=1, apply_noise=True):
        super(RNN_Dale, self).__init__()

        self.n_units = n_units  # number of neurons

        self.inp = nn.Parameter(torch.rand(n_inputs, n_units, device=device))  # input weights

        self.Wrecdale = nn.Parameter(torch.rand(n_units, n_units, device=device) \
                                     / n_units)  # recurrent weights - Dale
        # TODO: try doing abs(randn)
        self.out = nn.Parameter(torch.rand(n_units, 1, device=device))  # output weights

        # Dale sign matrix
        n_exc = int(prop_exc * n_units)
        exc_units = np.array([1 if (i <= n_exc - 1) else -1 for i in range(n_units)])
        self.Sdale = torch.Tensor(np.diag(exc_units)).to(device)
        self.tau = 5
        self.prop_old_act = 1 - 1 / self.tau
        self.prop_new_act = 1 / self.tau

        if apply_noise:
            self.noise_constant = np.sqrt(2 * global_noise_level / self.tau)
        else:
            self.noise_constant = 0

    def forward(self, inputs):

        # Initialize network state as zeros
        activity = torch.zeros(inputs.size(0), self.n_units, device=device)

        # pre-calculate noise for all time-points
        # noise = np.sqrt(2 * global_noise_level / self.tau) * \
        noise = self.noise_constant * \
                torch.randn(inputs.size(2), inputs.size(0), self.n_units, device=device)
        self.Wrec = torch.matmul(torch.abs(self.Wrecdale), self.Sdale)

        # Run the input through the network
        for i in range(inputs.size(2)):
            output, activity = self.step(inputs[:, :, i].unsqueeze(-1), activity, noise[i, :, :])
        return output.squeeze(), activity

    def step(self, input_ext, activity, noise):

        # non-linear
        activity = self.prop_old_act * activity + \
                   self.prop_new_act * (torch.matmul(self.Wrec, torch.relu(activity.unsqueeze(-1))).squeeze() +
                                        torch.matmul(input_ext, torch.abs(self.inp))) + noise

        # linear
        # activity = self.prop_old_act * activity + \
        #            self.prop_new_act * (torch.matmul(self.Wrec, activity.unsqueeze(-1)).squeeze() +
        #                                 torch.matmul(input_ext, torch.abs(self.inp))) + noise

        output = torch.matmul(activity, self.out)
        return output, activity

    def forward_outputs(self, inputs):

        # collect outputs
        track_outputs = torch.zeros(inputs.size(0), inputs.size(2), device=device)

        # Initialize network state as zeros
        activity = torch.zeros(inputs.size(0), self.n_units, device=device)
        # activity = torch.Tensor(np.array([1, 0])).to(device).unsqueeze(0)

        # pre-calculate noise for all time-points
        noise = self.noise_constant * \
                torch.randn(inputs.size(2), inputs.size(0), self.n_units, device=device)
        self.Wrec = torch.matmul(torch.abs(self.Wrecdale), self.Sdale)

        # Run the input through the network
        for i in range(inputs.size(2)):
            output, activity = self.step(inputs[:, :, i], activity, noise[i, :, :])
            track_outputs[:, i] = output.squeeze()

        return output.squeeze(), activity, track_outputs


class RNN_Dale_Sum(nn.Module):
    def __init__(self, n_units, prop_exc, alpha_dale=0.5, n_inputs=1, train_alpha=True,
                 apply_noise=True):
        super(RNN_Dale_Sum, self).__init__()

        self.n_units = n_units  # number of neurons
        self.inp = nn.Parameter(torch.rand(n_inputs, n_units, device=device))  # input weights
        self.Wrecnon = nn.Parameter((torch.rand(n_units, n_units, device=device) - 1) \
                                     / n_units)  # recurrent weights - non-Dale
        self.Wrecdale = nn.Parameter(torch.rand(n_units, n_units, device=device) \
                                     / n_units)  # recurrent weights - Dale
        # TODO: try doing abs(randn)
        self.out = nn.Parameter(torch.rand(n_units, 1, device=device))  # output weights

        # Dale sign matrix
        n_exc = int(prop_exc * n_units)
        exc_units = np.array([1 if (i <= n_exc - 1) else -1 for i in range(n_units)])
        self.Sdale = torch.Tensor(np.diag(exc_units)).to(device)
        self.tau = 5
        self.prop_old_act = 1 - 1 / self.tau
        self.prop_new_act = 1 / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.Tensor([alpha_dale]).to(device))
        else:
            self.alpha = torch.Tensor([alpha_dale]).to(device)

        if apply_noise:
            self.noise_constant = np.sqrt(2 * global_noise_level / self.tau)
        else:
            self.noise_constant = 0

    def forward(self, inputs):

        # Initialize network state as zeros
        activity = torch.zeros(inputs.size(0), self.n_units, device=device)
        # activity = torch.Tensor(np.array([1, 0])).to(device).unsqueeze(0)

        # pre-calculate noise for all time-points
        noise = self.noise_constant * \
                torch.randn(inputs.size(2), inputs.size(0), self.n_units, device=device)
        # use_alpha = torch.sigmoid(self.alpha)
        use_alpha = hard_tanh_0to1(self.alpha)
        # use_alpha = self.alpha

        self.Wsum = use_alpha * torch.matmul(torch.abs(self.Wrecdale), self.Sdale) + \
                     (1 - use_alpha) * self.Wrecnon

        # Run the input through the network
        for i in range(inputs.size(2)):
            output, activity = self.step(inputs[:, :, i], activity, noise[i, :, :])
        return output.squeeze(), activity

    def step(self, input_ext, activity, noise):

        # non-linear
        activity = self.prop_old_act * activity + \
                   self.prop_new_act * (torch.matmul(self.Wsum, torch.relu(activity.unsqueeze(-1))).squeeze() +
                                        torch.matmul(input_ext, torch.abs(self.inp))) + noise

        # linear
        # activity = self.prop_old_act * activity + \
        #            self.prop_new_act * (torch.matmul(self.Wsum, activity.unsqueeze(-1)).squeeze() +
        #                                 torch.matmul(input_ext, torch.abs(self.inp))) + noise

        output = torch.matmul(activity, self.out)
        return output, activity

    def forward_outputs(self, inputs):

        # collect outputs
        track_outputs = torch.zeros(inputs.size(0), inputs.size(2), device=device)

        # Initialize network state as zeros
        activity = torch.zeros(inputs.size(0), self.n_units, device=device)
        # activity = torch.Tensor(np.array([1, 0])).to(device).unsqueeze(0)

        # pre-calculate noise for all time-points
        noise = self.noise_constant * \
                torch.randn(inputs.size(2), inputs.size(0), self.n_units, device=device)
        # use_alpha = torch.sigmoid(self.alpha)
        use_alpha = hard_tanh_0to1(self.alpha)
        # use_alpha = self.alpha
        self.Wsum = use_alpha * torch.matmul(torch.abs(self.Wrecdale), self.Sdale) + \
                    use_alpha * self.Wrecnon

        # Run the input through the network
        for i in range(inputs.size(2)):
            output, activity = self.step(inputs[:, :, i], activity, noise[i, :, :])
            track_outputs[:, i] = output.squeeze()

        return output.squeeze(), activity, track_outputs
