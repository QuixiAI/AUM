# AUM-Ø diagnostics (§21): calibration corr(pi,b), the Delta-sigma quartet, and the
# sigma-decode probe (the primary instrument for interpreting a §23 gate failure).

import torch
import torch.nn.functional as F


def corr(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float((a * b).sum() / denom) if float(denom) > 0 else 0.0


def corr_pi_benefit(pi, b):
    """corr(pi_t, b_t) held-out (§21). pi (B,L) or (B,L-1); b (B,L-1)."""
    if pi.shape[-1] != b.shape[-1]:
        pi = pi[:, : b.shape[-1]]
    return corr(pi, b)


def delta_sigma_quartet(aux):
    """Hypothesis inertia (§16): Delta-sigma_silent = ||sigma^{j*} - sigma^0|| + co-firing signals."""
    d_silent = (aux.sigma_star - aux.sigma0).norm(dim=-1)
    return {
        "delta_sigma_silent": float(d_silent.mean()),
        "expected_J": float(aux.expected_J.mean()),
        "pi": float(aux.pi.mean()),
    }


def sigma_decode_probe(sigma, labels, n_classes, epochs=300, lr=0.05, train_frac=0.7):
    """Linear probe: decode the latent rule from sigma_t (§21/§23). Returns held-out accuracy.

    sigma: (N, d_sigma). labels: (N,) int in [0, n_classes). Chance is 1/n_classes.
    """
    N = sigma.shape[0]
    sigma = sigma.reshape(N, -1).float().detach()
    labels = labels.long()
    n_tr = max(1, int(train_frac * N))
    clf = torch.nn.Linear(sigma.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(clf(sigma[:n_tr]), labels[:n_tr]).backward()
        opt.step()
    if N - n_tr == 0:
        return float("nan")
    with torch.no_grad():
        acc = (clf(sigma[n_tr:]).argmax(-1) == labels[n_tr:]).float().mean()
    return float(acc)
