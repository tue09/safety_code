import torch
from typing import List, Dict, Union, Tuple


class FAMO:
    """
    Fast Adaptive Multitask Optimization (FAMO), general K-task version.

    Usage:
        famo = FAMO(num_tasks=K, beta=1e-3, gamma=1e-3, device="cuda")
        total_loss, info = famo.combine([loss1, loss2, ..., lossK])
        total_loss.backward()
        optimizer.step()

    Notes:
      - Paper assumes each loss is positive (ℓ_i > 0). We enforce this by shifting
        each loss with its running minimum + eps, so log(ℓ) and 1/ℓ are safe. :contentReference[oaicite:1]{index=1}
      - We detach weights before combining losses to match Algorithm 1 update form.
      - Logits update uses a 1-step delay (needs prev step losses).
    """

    def __init__(
        self,
        num_tasks: int,
        beta: float = 1e-3,      # logits learning rate
        gamma: float = 1e-3,     # decay on logits
        eps: float = 1e-8,       # numerical stability
        clamp_xi: float = 10.0,  # keep logits bounded
        device: Union[str, torch.device, None] = None,
    ):
        assert num_tasks >= 2, "num_tasks must be >= 2"
        self.k = num_tasks
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.clamp_xi = clamp_xi

        self.device = torch.device(device) if device is not None else None
        self.xi = torch.zeros(self.k, device=self.device)  # task logits ξ

        # running minima to shift losses to positive domain
        self.min_losses = torch.full((self.k,), float("inf"), device=self.device)

        # cache previous step values for delayed xi update
        self.prev_losses_pos = None
        self.prev_z = None

    def reset(self):
        """Reset logits and history."""
        self.xi.zero_()
        self.min_losses.fill_(float("inf"))
        self.prev_losses_pos = None
        self.prev_z = None

    def _softmax(self) -> torch.Tensor:
        return torch.softmax(self.xi, dim=0)

    @torch.no_grad()
    def _make_positive(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Make losses strictly positive:
            ℓ_pos = (ℓ - min_seen) + eps
        This keeps math stable for log(ℓ) and 1/ℓ.
        """
        self.min_losses = torch.minimum(self.min_losses, losses.detach())
        return (losses - self.min_losses) + self.eps

    @torch.no_grad()
    def _delta_softmax_times_vec(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute (Jacobian of softmax at z) @ v efficiently:
            J(z) = diag(z) - z z^T
            J(z) v = z * v - z * (z^T v)
        O(K) time and no KxK matrix.
        """
        dot = torch.dot(z, v)
        return z * (v - dot)

    @torch.no_grad()
    def _update_logits(self, curr_losses_pos: torch.Tensor):
        """
        Update ξ using previous z and loss improvement:
            diff = log(prev_losses) - log(curr_losses)
            delta = J(prev_z) @ diff
            ξ <- ξ - beta * (delta + gamma * ξ)
        """
        if self.prev_losses_pos is None:
            return

        diff = torch.log(self.prev_losses_pos + self.eps) - torch.log(curr_losses_pos + self.eps)
        delta = self._delta_softmax_times_vec(self.prev_z, diff)

        self.xi -= self.beta * (delta + self.gamma * self.xi)
        if self.clamp_xi is not None:
            self.xi.clamp_(-self.clamp_xi, self.clamp_xi)

    @torch.no_grad()
    def _get_weights(self, losses_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute:
            z = softmax(ξ)
            w_i = (z_i / ℓ_i) / sum_j (z_j / ℓ_j)
        """
        z = self._softmax()
        ratio = z / (losses_pos + self.eps)
        w = ratio / (ratio.sum() + self.eps)
        return w, z

    def combine(
        self,
        losses: Union[List[torch.Tensor], Dict[str, torch.Tensor]],
        return_named: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Combine K task losses into a single scalar loss for backward().

        Args:
            losses:
              - list of K tensors: [ℓ1, ℓ2, ..., ℓK]
              - or dict {name: tensor}, length K
            return_named:
              - if dict input, return info per key

        Returns:
            total_loss: scalar tensor
            info: dict for logging (weights, probs, logits)
        """
        if isinstance(losses, dict):
            names = list(losses.keys())
            loss_list = [losses[n] for n in names]
        else:
            names = [f"task_{i}" for i in range(len(losses))]
            loss_list = list(losses)

        assert len(loss_list) == self.k, f"Expected {self.k} losses, got {len(loss_list)}"

        # stack losses
        L = torch.stack(loss_list)
        if self.device is not None:
            L = L.to(self.device)

        with torch.no_grad():
            # 1) shift to positive domain for weighting math
            L_pos = self._make_positive(L)

            # 2) update logits from last step improvement
            self._update_logits(L_pos)

            # 3) compute weights for this step
            w, z = self._get_weights(L_pos)

            # 4) cache for next step
            self.prev_losses_pos = L_pos.detach()
            self.prev_z = z.detach()

        # detach weights so gradients only apply to model params
        print(f'---- weight = {w}')
        total_loss = (w.detach() * L).sum()

        info = {
            "weights": w.detach().cpu(),
            "probs": z.detach().cpu(),
            "xi": self.xi.detach().cpu(),
            "losses_raw": L.detach().cpu(),
            "losses_pos": L_pos.detach().cpu(),
        }

        if return_named:
            info_named = {}
            for i, n in enumerate(names):
                info_named[n] = {
                    "w": float(w[i].item()),
                    "z": float(z[i].item()),
                    "xi": float(self.xi[i].item()),
                    "loss": float(L[i].item()),
                }
            info["named"] = info_named

        return total_loss, info
