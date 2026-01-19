import torch
from typing import Sequence, Optional, Tuple, Union
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, LinearKernel, CosineKernel
from assignment_kernel import AssignmentDistanceKernel
from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model


class GPModel:
    def __init__(self, prior_choice='standard', kernel_choice='linear', ard=True, noise_type='high', fixed_noise_value=None, nBits=None, **tkwargs):
        self.nBits = nBits
        self.prior_choice = prior_choice
        self.kernel_choice = kernel_choice
        self.ard = ard
        self.noise_type = noise_type
        self.fixed_noise_value = fixed_noise_value
        self.tkwargs = tkwargs
        self.model = None
        self.likelihood = None

    @classmethod
    def from_params(cls, prior_choice='standard', kernel_choice='linear', ard=True, noise_type='high', fixed_noise_value=None, nBits=None, **tkwargs):
        return cls(prior_choice, kernel_choice, ard, noise_type, fixed_noise_value, nBits, **tkwargs)

    def fit(self, train_x, train_y, train_yvar=None, use_hvarfner_prior=False):
        input_dim = train_x.shape[-1]
        if self.noise_type == 'standard':
            self.likelihood = GaussianLikelihood(
                noise_prior=GammaPrior(torch.tensor(0.9, **self.tkwargs), torch.tensor(10.0, **self.tkwargs)),
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            )
        elif self.noise_type == 'high':
            self.likelihood = GaussianLikelihood(
                noise_prior=LogNormalPrior(loc=-2.0, scale=1.0),
                noise_constraint=LogTransformedInterval(1e-4, 1.0),
            )
        elif self.noise_type == 'fixed':
            if self.fixed_noise_value is None:
                raise ValueError("fixed_noise_value must be provided if noise_type is 'fixed'")
            self.likelihood = GaussianLikelihood(noise_constraint=GreaterThan(self.fixed_noise_value - 1e-5))
            self.likelihood.noise = torch.tensor([self.fixed_noise_value])
            self.likelihood.raw_noise.requires_grad = False


        kernel1 = AssignmentDistanceKernel(nBits=self.nBits, active_dims=list(range(input_dim - 2)))
        kernel2 = MaternKernel(ard_num_dims=2, active_dims=list(range(input_dim - 2, input_dim)))
        kernel = kernel1 * kernel2
        
        # if use_hvarfner_prior:
        #     covar_module = get_covar_module_with_hvarfner_prior(input_dim, use_matern_kernel=True, use_scale_kernel=True, use_outputscale_prior=True)
        # else:
        outputscale_prior = GammaPrior(torch.tensor(2.0, **self.tkwargs), torch.tensor(0.15, **self.tkwargs))
        outputscale_constraint = GreaterThan(1e-6)
        covar_module = ScaleKernel(
            base_kernel=kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint,
        )
        input_transform = None
        outcome_transform = None
        self.model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y.unsqueeze(-1),
            train_Yvar=train_yvar,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            likelihood=self.likelihood,
        )
        mll = ExactMarginalLogLikelihood(model=self.model, likelihood=self.model.likelihood)
        fit_gpytorch_model(mll)
        return self.model

    def load_model(self, path, train_x_shape=None, train_y_shape=None, train_yvar=None):
        """
        Loads model weights from the given path. You must provide train_x_shape (tuple) and train_y_shape (tuple)
        to reconstruct the model architecture, or call fit() first with dummy data to initialize.
        """
        if self.model is None:
            if train_x_shape is None or train_y_shape is None:
                raise ValueError("You must provide train_x_shape and train_y_shape if model is not initialized.")
            # Create dummy tensors to initialize model
            dummy_x = torch.zeros(train_x_shape, dtype=torch.double)
            dummy_y = torch.zeros(train_y_shape, dtype=torch.double)
            self.fit(dummy_x, dummy_y, train_yvar=train_yvar)
        self.model.load_state_dict(torch.load(path))
        return self.model

    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        torch.save(self.model.state_dict(), path)

    def predict(self, test_x):
        if self.model is None:
            raise ValueError("Model has not been fitted/loaded yet.")
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(test_x)
            mean = posterior.mean
        return mean.squeeze(-1).cpu().numpy()


def moment_matched_ensemble_posterior(
    models: Sequence[object],
    X: torch.Tensor,
    weights: Optional[Sequence[float]] = None,
    return_distribution: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.distributions.Normal],
]:
    """
    Compute a moment-matched Gaussian approximation to an ensemble of GP predictions.

    For two (or more) independent GP models with predictive distributions
    Normal(m_k, v_k) at inputs X, treat the ensemble as an equally-weighted
    mixture of Gaussians (or weighted if `weights` provided) and match the first
    two moments to a single Normal(mean, variance):

        mean = sum_k w_k * m_k
        var  = sum_k w_k * (v_k + m_k^2) - mean^2

    This captures both aleatoric uncertainty (average of variances) and model
    disagreement (variance of means).

    Parameters
    - models: sequence of BoTorch models supporting `.posterior(X)`
    - X: input tensor at which to predict
    - weights: optional non-negative weights summing to 1. If None, uses uniform
    - return_distribution: if True, also returns a torch.distributions.Normal

    Returns
    - (mean, variance) tensors with shapes matching the model's posterior mean/variance

    Notes
    - Supports single-output models. For multi-output, wrap each output separately.
    - All computations are done on X.device and X.dtype.
    """
    if len(models) == 0:
        raise ValueError("'models' must contain at least one model.")

    device = X.device
    dtype = X.dtype

    with torch.no_grad():
        means = []
        variances = []
        for model in models:
            posterior = model.posterior(X)
            mean_k = posterior.mean  # shape: [..., q, m]
            var_k = posterior.variance  # same shape as mean

            # Enforce single-output for simplicity
            if mean_k.shape[-1] != 1:
                raise ValueError(
                    "moment_matched_ensemble_posterior currently supports single-output models only."
                )

            means.append(mean_k.to(device=device, dtype=dtype))
            variances.append(var_k.to(device=device, dtype=dtype))

        num_models = len(means)

        if weights is None:
            weights_t = torch.full((num_models,), 1.0 / num_models, device=device, dtype=dtype)
        else:
            weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
            if weights_t.numel() != num_models:
                raise ValueError("Length of 'weights' must match number of models.")
            if (weights_t < 0).any():
                raise ValueError("All weights must be non-negative.")
            weight_sum = weights_t.sum()
            if not torch.isclose(weight_sum, torch.tensor(1.0, device=device, dtype=dtype)):
                weights_t = weights_t / (weight_sum + 1e-12)

        # Stack on a new model dimension and compute weighted moments
        mean_stack = torch.stack(means, dim=0)  # [K, ..., q, 1]
        var_stack = torch.stack(variances, dim=0)  # [K, ..., q, 1]

        # Reshape weights for broadcasting
        w = weights_t.view((-1,) + (1,) * (mean_stack.dim() - 1))  # [K, 1, ..., 1]

        weighted_mean = (w * mean_stack).sum(dim=0)
        second_moment = (w * (var_stack + mean_stack.pow(2))).sum(dim=0)
        ensemble_variance = (second_moment - weighted_mean.pow(2)).clamp_min(1e-12)

        if return_distribution:
            dist = torch.distributions.Normal(loc=weighted_mean, scale=ensemble_variance.sqrt())
            return weighted_mean, ensemble_variance, dist

        return weighted_mean, ensemble_variance