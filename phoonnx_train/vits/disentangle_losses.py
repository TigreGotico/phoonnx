import math
import torch
import torch.nn.functional as F


def kl_divergence(mean1, logvar1, mean2, logvar2):
    """KL divergence between two diagonal Gaussians."""
    return 0.5 * (
        logvar2 - logvar1
        + (torch.exp(logvar1) + (mean1 - mean2) ** 2) / torch.exp(logvar2)
        - 1.0
    )


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)


def info_nce_loss(embeddings, labels, temperature=0.07):
    """
    InfoNCE / contrastive loss for disentanglement.

    Args:
        embeddings: [B, D] — a batch of embeddings.
        labels: [B] — integer labels (e.g. speaker IDs for timbre encoder).
                      Positive pairs share the same label.
        temperature: softmax temperature.
    Returns:
        Scalar loss.
    """
    if labels is None or embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Normalize for cosine similarity
    embeddings = F.normalize(embeddings, dim=-1)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]

    # Create a mask for positive pairs (same label, excluding self)
    labels = labels.unsqueeze(0)  # [1, B]
    mask = labels == labels.T  # [B, B]
    mask.fill_diagonal_(False)

    # For each anchor, positives are those with same label
    loss = 0.0
    valid_count = 0
    for i in range(embeddings.size(0)):
        pos_mask = mask[i]
        if pos_mask.sum() == 0:
            continue
        # Numerator: sum of exp(similarities to positives)
        pos_sim = sim_matrix[i, pos_mask]
        # Denominator: sum over all except self
        neg_mask = ~mask[i]
        neg_mask[i] = False
        all_sim = sim_matrix[i, neg_mask]
        denom = torch.cat([pos_sim, all_sim]).exp().sum()
        loss += -(pos_sim.exp().sum().log() - denom.log())
        valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return loss / valid_count


def mutual_information_loss(g_timbre, g_artic, g_prosody, sid,
                              lambda_timbre=0.1, lambda_artic=0.1, lambda_prosody=0.1):
    """
    Contrastive MI loss: timbre embeddings should cluster by speaker,
    while artic/prosody embeddings should *not* cluster by speaker.

    Args:
        g_timbre, g_artic, g_prosody: [B, D, 1] conditioning vectors.
        sid: [B] speaker IDs (or None).
    Returns:
        dict of loss components.
    """
    if sid is None:
        return {}

    g_timbre = g_timbre.squeeze(-1)  # [B, D]
    g_artic = g_artic.squeeze(-1)
    g_prosody = g_prosody.squeeze(-1)

    loss_timbre = info_nce_loss(g_timbre, sid)

    # For articulation and prosody, we want the *opposite*: they should NOT
    # encode speaker identity. We use gradient reversal + a speaker classifier.
    loss_artic = info_nce_loss(g_artic, sid)
    loss_prosody = info_nce_loss(g_prosody, sid)

    return {
        "loss_mi_timbre": lambda_timbre * loss_timbre,
        "loss_mi_artic": -lambda_artic * loss_artic,
        "loss_mi_prosody": -lambda_prosody * loss_prosody,
    }


def cycle_reconstruction_loss(model_g, x, x_lengths, y, y_lengths,
                              sid_a, sid_b,
                              timbre_ref_a, timbre_ref_b,
                              artic_ref_a, artic_ref_b,
                              prosody_ref_a, prosody_ref_b,
                              lambda_cycle=1.0):
    """
    Cycle consistency loss: swap one factor between two speakers and
    ensure reconstruction quality is maintained.

    The core idea: if we take speaker A's timbre with speaker B's prosody
    on text from A, the output should still be a plausible utterance of
    text A (not a garbled mix).

    For simplicity, we compute a forward pass with swapped timbre and
    compare mel reconstruction loss against the baseline.
    """
    with torch.no_grad():
        baseline = model_g(x, x_lengths, y, y_lengths, sid=sid_a)[0]

    # Swap timbre only: speaker B's voice speaking A's content with A's rhythm
    swapped = model_g(x, x_lengths, y, y_lengths,
                      sid=sid_a,
                      timbre_ref_mel=timbre_ref_b,
                      artic_ref_mel=artic_ref_a,
                      prosody_ref_mel=prosody_ref_a)[0]

    loss_cycle = F.l1_loss(swapped, baseline.detach()) * lambda_cycle
    return {"loss_cycle": loss_cycle}


def kl_regularization_loss(z_timbre, z_artic, z_prosody,
                           lambda_kl=0.01):
    """
    KL regularization on each latent to encourage smooth, compact latent spaces.
    Treats each factor as an isotropic Gaussian and penalizes deviation from
    N(0, I).
    """
    def _kl(z):
        mean = z.mean(dim=-1, keepdim=True)
        logvar = torch.log(z.var(dim=-1, keepdim=True) + 1e-8)
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    loss = _kl(z_timbre) + _kl(z_artic) + _kl(z_prosody)
    return {"loss_kl_dis": lambda_kl * loss / 3.0}
