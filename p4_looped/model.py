"""Small looped language model with a constrained or free injection step.

Layout: prelude blocks -> looped core (shared weights, run n times) -> readout C -> coda blocks -> head.
Loop update: h <- A * h + Bbar, then h <- core(h).

  inject="constrained": A = exp(-dt * exp(log_a)) in (0, 1), Bbar = dt * B(norm(x))
  inject="free":        A = learned per-channel vector initialised at 1, Bbar = B(norm(x))

The constrained form follows a negative-diagonal discretisation. It was written from a reading of the
Parcae paper and has not been checked line by line against the authors' reference code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    vocab: int = 50257
    ctx: int = 256
    d: int = 384
    n_heads: int = 6
    n_prelude: int = 1
    n_core: int = 3
    n_coda: int = 1
    inject: str = "constrained"   # "free" or "constrained"


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.n_heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.o = nn.Linear(c.d, c.d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.up = nn.Linear(c.d, 4 * c.d, bias=False)
        self.down = nn.Linear(4 * c.d, c.d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d), RMSNorm(c.d)
        self.attn, self.mlp = Attn(c), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Injection(nn.Module):
    """Diagonal SSM-style discretisation: A_bar = exp(-dt * exp(log_a)), B_bar = dt * B."""

    def __init__(self, c):
        super().__init__()
        self.mode = c.inject
        if self.mode == "free":
            self.a = nn.Parameter(torch.ones(c.d))          # unconstrained ablation
        else:
            self.log_a = nn.Parameter(torch.zeros(c.d))     # A_cont = -exp(log_a) < 0
            self.log_dt = nn.Parameter(torch.zeros(c.d))    # dt = softplus(log_dt) > 0

    def forward(self):
        if self.mode == "free":
            return self.a, None                             # no discretisation, B_bar = B
        dt = F.softplus(self.log_dt)
        return torch.exp(-dt * torch.exp(self.log_a)), dt


def sample_loops(B, mean=6, lo=1, hi=12, device="cuda"):
    """One recurrence count PER SEQUENCE (Poisson, clamped)."""
    n = torch.poisson(torch.full((B,), float(mean), device=device))
    return n.clamp(lo, hi).long()


class LoopedLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.prelude = nn.ModuleList([Block(c) for _ in range(c.n_prelude)])
        self.core = nn.ModuleList([Block(c) for _ in range(c.n_core)])
        self.coda = nn.ModuleList([Block(c) for _ in range(c.n_coda)])
        self.inj = Injection(c)
        self.norm_e = RMSNorm(c.d)
        self.B = nn.Linear(c.d, c.d, bias=False)   # input projection
        self.C = nn.Linear(c.d, c.d, bias=False)   # readout projection
        self.norm_f = RMSNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight         # weight tying
        self.apply(self._init)
        nn.init.eye_(self.C.weight)                # start C as identity

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, n_loops=4, k_grad=2, targets=None, return_norms=False):
        """n_loops: int (same depth for all sequences) or LongTensor (B,) (one count per sequence).
        k_grad: gradients flow only through the last k_grad loops of each sequence."""
        B, T = idx.shape
        dev = idx.device
        x = self.tok(idx) + self.pos(torch.arange(T, device=dev))
        for blk in self.prelude:
            x = blk(x)

        A, dt = self.inj()
        Bbar = self.B(self.norm_e(x))
        if dt is not None:
            Bbar = Bbar * dt                       # Delta factor on the input injection

        if isinstance(n_loops, int):
            n_vec = torch.full((B,), n_loops, device=dev, dtype=torch.long)
        else:
            n_vec = n_loops.to(dev)
        n_list = n_vec.tolist()
        grad_on = torch.is_grad_enabled()

        h = torch.zeros_like(x)
        norms = []
        for t in range(int(max(n_list))):
            active = (t < n_vec).view(B, 1, 1)                 # sequence still looping
            keep_grad = (t > n_vec - k_grad).view(B, 1, 1)     # h carries gradient from earlier loop
            need_grad = grad_on and any(n - k_grad <= t < n for n in n_list)
            with torch.set_grad_enabled(need_grad):
                h_in = torch.where(keep_grad, h, h.detach())   # cut gradient before each seq's window
                h_new = A * h_in + Bbar
                for blk in self.core:
                    h_new = blk(h_new)
                h = torch.where(active, h_new, h)              # finished sequences stay frozen
            if return_norms:
                norms.append(h.detach().float().pow(2).mean().sqrt().item())

        h = self.C(h)                                          # readout before the coda
        for blk in self.coda:
            h = blk(h)
        logits = self.head(self.norm_f(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, norms
