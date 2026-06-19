import torch
from diffusers import DDPMScheduler, UNet2DModel
from lib.models.latent_noise_prediction_model import LatentNoisePredictionModel


class DDPM_CIFAR10(LatentNoisePredictionModel):
    """
    Pixel-space DDPM trained on CIFAR-10 (32x32).
    No VAE — image_to_latent and latent_to_image are identity (with normalization).
    """

    def __init__(
        self,
        model_id="google/ddpm-cifar10-32",
        device="cuda",
        dtype=torch.float32,  # DDPM CIFAR10 works best in float32
    ):
        self.device = device
        self.dtype = dtype

        self.unet = UNet2DModel.from_pretrained(model_id).to(device).to(dtype)
        self.unet.eval()

        self.scheduler = DDPMScheduler.from_pretrained(model_id)
        self.scheduler.set_timesteps(1000)

        # Pre-compute SNR values: snr = sqrt(alpha_bar / (1 - alpha_bar))
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        self.snr_values = torch.sqrt(alphas_cumprod / (1 - alphas_cumprod))
        self.snr_values_actual = alphas_cumprod / (1 - alphas_cumprod)

        # Dummy config — no guidance, no prompt
        self.image_width = 32
        self.image_height = 32

    def get_timestep_snr(self, timestep):
        if timestep == 0:
            return torch.inf
        return self.snr_values[timestep - 1]
    
    def get_timestep_snr_actual(self, timestep):
        if timestep == 0:
            return torch.inf
        return self.snr_values_actual[timestep - 1]

    def image_to_latent(self, img_pt):
        """
        Identity transform with normalization.
        Input: [0, 1] range, shape (1, 3, 32, 32)
        Output: [-1, 1] range (what DDPM expects)
        """
        if img_pt.dim() == 3:
            img_pt = img_pt.unsqueeze(0)
        return (img_pt.to(device=self.device, dtype=self.dtype) * 2 - 1)

    def latent_to_image(self, latent):
        """
        Identity transform back to [0, 1].
        """
        return ((latent + 1) / 2).clamp(0, 1).detach()

    def configure(self, prompt, prompt_guidance, image_width, image_height):
        """No-op for unconditional DDPM."""
        self.image_width = image_width
        self.image_height = image_height

    @torch.no_grad()
    def predict_noise(self, noisy_latent, timestep):
        """
        Predict noise (epsilon) from the noisy image at the given timestep.
        """
        t = torch.tensor([timestep], device=self.device, dtype=torch.long)
        noise_pred = self.unet(
            noisy_latent.to(self.dtype), t
        ).sample
        return noise_pred.to(noisy_latent.dtype)