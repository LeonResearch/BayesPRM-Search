from functools import partial
from pathlib import Path

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel, RBFKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from transformers import PreTrainedModel

from models.autoencoder import Autoencoder
from models.configs import AutoencoderConfig, DeepKernelGPConfig


class BayesRewardModel(PreTrainedModel):
    def __init__(
            self, 
            gp_model: DeepKernelExactGP,
            emb_model: PreTrainedModel,
            config,
        ):
        super().__init__(config)
        
        self.emb_model = emb_model
        self.gp_model = gp_model
    

    def generate_embeddings(self, inputs_batch):
        with torch.no_grad():
            outputs = self.emb_model(inputs_batch)
            embeddings = outputs.last_hidden_state[:, 0].detach().cpu()
        return embeddings

    def ucb_acquisition(
            self,
            candidates: torch.Tensor, 
            beta: float = 2.0
        ) -> torch.Tensor:
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            posterior = self.gp_model.likelihood(self.gp_model(candidates))
            mean = posterior.mean
            var = posterior.variance
            acq_values = mean + beta * var.sqrt()
            return acq_values


    def forward(self, inputs_batch):
        # Get the BERT embeddings
        embeddings = self.generate_embeddings(inputs_batch)
        posterior_mean = self.gp_model(embeddings)
        
        return posterior_mean
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(model_name_or_path, *args, **kwargs)
        return model

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)


class DeepKernelExactGP(ExactGP):
    def __init__(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
            covar_module: Kernel,
            encoder: nn.Module,
            likelihood: Likelihood
    ):
        """
        Implements a Deep Kernel GP where the input is first projected to a latent space through an encoder,
        and then the covariance module is used on that latent space.
        Note: the encoder should have a sigmoid last layer to keep the output in [0,1]
        """
        super(DeepKernelExactGP, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = covar_module
        self.encoder = encoder

    @property
    def num_outputs(self):
        return 1

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        z = self.encoder(x)

        mean = self.mean_module(z)
        covar = self.covar_module(z)
        return MultivariateNormal(mean, covar)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            if p.requires_grad:
                p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    @classmethod
    def factory(
            cls,
            gp_cfg: DeepKernelGPConfig,
            autoencoder_cfg: AutoencoderConfig = None,
            autoencoder: Autoencoder = None
    ):
        """
        Partial constructor for DeepKernelExactGP.
        Takes the autoencoder and gp configs and instantiates both from scratch.
        There is also an option to pass an instance of Autoencoder if needed, in particular for joint training.
        """
        if autoencoder_cfg is None and autoencoder is None:
            raise ValueError("Either autoencoder_cfg or autoencoder must be specified.")
        if autoencoder is None:
            autoencoder = Autoencoder(autoencoder_cfg)
        encoder = autoencoder.encoder
        if gp_cfg.ard and autoencoder_cfg.emb_dim != gp_cfg.ard_num_dims:
            raise ValueError(
                f"ard_num_dims must be equal to emb_dim: {gp_cfg.ard_num_dims} != {autoencoder_cfg.emb_dim}")
        covar_module = get_kernel(**gp_cfg.kernel)
        likelihood = get_likelihood(**gp_cfg.likelihood)
        return partial(cls, encoder=encoder, covar_module=covar_module, likelihood=likelihood)

    @classmethod
    def factory_from_logdir(
            cls,
            logdir: Path | str,
            load_finetuned_encoder: bool = False,
    ):
        """
        Assumes that logdir contains `config.yaml` and `best_ckpt.pt`
        Loads the encoder weights but not the GP parameters.

        If `load_finetuned_encoder` is `True`, the encoder weights are loaded from finetuned checkpoint `best_ckpt.pt`.
        Otherwise, the encoder weights are loaded from pretrained checkpoint in config.
        """
        config = OmegaConf.load(logdir / "config.yaml")
        autoencoder_cfg = AutoencoderConfig(**config.autoencoder)

        # Load Autoencoder from checkpoint
        autoencoder = Autoencoder(autoencoder_cfg)
        if load_finetuned_encoder:
            state_dict = torch.load(logdir / "best_ckpt.pt",weights_only=True, map_location='cpu')
            encoder_state_dict = {k: v for k, v in state_dict.items() if 'encoder' in k}
            autoencoder.load_state_dict(encoder_state_dict, strict=False)

        # then make GP factory
        gp_cfg = DeepKernelGPConfig(**config.gp)
        factory = cls.factory(gp_cfg=gp_cfg, autoencoder=autoencoder)

        return factory


def get_kernel(name: str, ard: bool = False, ard_num_dims: int = None) -> Kernel:
    if name == 'matern-52':
        kernel = ScaleKernel(base_kernel=MaternKernel(nu=5 / 2, ard_num_dims=ard_num_dims))
    elif name == 'rbf':
        kernel = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=ard_num_dims))
    else:
        raise ValueError(f'Unknown kernel name: {name}')

    return kernel


def get_likelihood(name: str) -> Likelihood:
    if name == 'gaussian':
        likelihood = GaussianLikelihood()
    else:
        raise ValueError(f'Unknown likelihood name: {name}')
    return likelihood