import math

from scipy import integrate
import torch
from torch import nn
from torchdiffeq import odeint
import torchsde
from tqdm.auto import trange, tqdm

import k_diffusion as K
from k_diffusion.sampling import default_noise_sampler, get_ancestral_step, to_d

@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None): # x is noise
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        # import pdb; pdb.set_trace()
        denoised, attns = model(x, sigmas[i] * s_in, **extra_args)
        # import pdb; pdb.set_trace()
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    import pdb; pdb.set_trace()
    return x

@torch.no_grad()
def sample_euler_ancestral_multistep_output(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None): # x is noise
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    total_output = []
    total_attns = []
    for i in trange(len(sigmas) - 1, disable=disable):
        # import pdb; pdb.set_trace()
        denoised, attns = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
            
        total_output.append(x)
        total_attns.append(attns)
    return total_output, total_attns