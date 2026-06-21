from tqdm import tqdm
from lib.diffc.utils.alpha_beta import get_alpha_prod_and_beta_prod
import torch

@torch.no_grad()
def denoise(noisy_latent, latent_timestep, timestep_schedule, noise_prediction_model, eta=0, denoise_seed=10):
    """
    Perform probability-flow-based denoising upon the noisy latent.

    Args:
        noisy_latent: latent to be denoised.
        latent_SNR: signal to noise ratio of the latent to be denoised.
        SNR_schedule (List[float]): List of signal-to-noise ratios in decreasing order,
            matching the schedule used during encoding. Last element should be 0 for fully denoised image.
        predict_noise (callable): Function that predicts the noise component given a noisy
            latent and its SNR.

    """
    latent = noisy_latent
    current_timestep = latent_timestep
    current_snr = noise_prediction_model.get_timestep_snr(current_timestep)

    timestep_schedule = [t for t in timestep_schedule if t < latent_timestep]

    device = noise_prediction_model.device
    gen = torch.Generator(device=device)

    if denoise_seed is None:
        gen.seed()
    else:
        gen.manual_seed(denoise_seed)
       
    for prev_timestep in tqdm(
        timestep_schedule
    ):  # "previous" as in higher than the current snr
        noise_prediction = noise_prediction_model.predict_noise(
            latent.to(noise_prediction_model.dtype), current_timestep
        ).to(torch.float32)
        prev_snr = noise_prediction_model.get_timestep_snr(prev_timestep)

        alpha_prod_t, beta_prod_t = get_alpha_prod_and_beta_prod(current_snr)
        alpha_prod_t_prev, beta_prod_t_prev = get_alpha_prod_and_beta_prod(prev_snr)

        # if int(prev_timestep) == 0:
        #    from IPython.core.debugger import set_trace
        #    set_trace()

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        sample = latent
        model_output = noise_prediction
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    
        # stochastic DDIM (DDIM paper eq. 12/16). eta=0 -> deterministic (original).

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        # sigma_t^2 = DDPM posterior variance, scaled by eta^2 (== `variance` in P.py)
        sigma_t = eta * (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        ) ** (0.5)

        # 6. direction term now excludes the sigma_t^2 budget
        pred_sample_direction = (1 - alpha_prod_t_prev - sigma_t ** 2) ** (0.5) * pred_epsilon

        # 7. add the stochastic noise term with FRESH, NON-SHARED randomness
        noise = torch.randn(
            latent.shape, device=latent.device, dtype=latent.dtype, generator=gen
        )
        latent = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample
            + pred_sample_direction
            + sigma_t * noise
        )

        current_timestep = prev_timestep
        current_snr = prev_snr

    return latent.to(noisy_latent.dtype)
