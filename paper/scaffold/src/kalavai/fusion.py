"""
Fusion methods for combining independently trained specialist modules.

Three backends:
  1. Weight Averaging: average unfrozen layer weights (simplest baseline)
  2. MoE Routing: learned router selects specialist per token (KALAVAI Backend A)
  3. BTX-style: average attention weights, MoE-route FFN layers only (Meta baseline)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================================
# Method 1: Weight Averaging (unfrozen layers only)
# ============================================================================

def fuse_by_averaging(modules, config):
    """
    Average the unfrozen layer weights across N modules.
    Frozen layers are already identical by construction.
    Works for both custom MiniGPT and HuggingFace models.
    """
    fused = copy.deepcopy(modules[0])
    n = len(modules)

    # Get all parameter names
    param_names = dict(fused.named_parameters())

    for name, param in fused.named_parameters():
        # Check if this parameter is in the frozen region
        # (frozen params are identical across modules, so averaging is a no-op)
        if not param.requires_grad:
            continue
        # Average across all modules
        avg = sum(dict(m.named_parameters())[name].data for m in modules) / n
        param.data.copy_(avg)

    return fused


# ============================================================================
# Method 2: MoE Routing (KALAVAI Backend A)
# ============================================================================

class MoEFusion(nn.Module):
    """
    N-expert MoE fusion with learned router.
    Router operates on shared frozen backbone output.
    Each expert is the unfrozen specialist layers from one module.
    """
    def __init__(self, n_experts, d_model, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, expert_outputs, shared_hidden):
        """
        Args:
            expert_outputs: list of N tensors, each (B, T, D) from specialist layers
            shared_hidden: (B, T, D) output of frozen backbone (for routing)
        Returns:
            fused: (B, T, D) weighted combination of expert outputs
        """
        # Router logits from shared backbone (detached to not backprop through backbone)
        logits = self.router(shared_hidden.detach())  # (B, T, N)

        if self.top_k < self.n_experts:
            # Top-k routing
            topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            weights = torch.zeros_like(logits).scatter(-1, topk_idx, F.softmax(topk_vals, dim=-1))
        else:
            # Dense routing (all experts)
            weights = F.softmax(logits, dim=-1)

        # Weighted combination
        stacked = torch.stack(expert_outputs, dim=-1)  # (B, T, D, N)
        fused = (stacked * weights.unsqueeze(2)).sum(dim=-1)  # (B, T, D)
        return fused, weights


class MoEFusedModel(nn.Module):
    """
    Full MoE-fused model for N specialist modules.
    Architecture: shared frozen backbone → N parallel specialist stacks → MoE router → output head.
    """
    def __init__(self, modules, config, top_k=2):
        super().__init__()
        self.config = config
        n_freeze = config.alignment.freeze_layers
        n_experts = len(modules)

        # Shared components (from first module — identical across all due to freezing)
        self.shared_backbone = modules[0]  # we'll use its forward partially
        self.n_freeze = n_freeze

        # Extract specialist (unfrozen) blocks from each module
        self.specialist_blocks = nn.ModuleList()
        for m in modules:
            if hasattr(m, 'blocks'):
                # Custom MiniGPT
                blocks = nn.ModuleList([m.blocks[i] for i in range(n_freeze, len(m.blocks))])
            elif hasattr(m, 'model'):
                # HuggingFace model
                layers = m.model.layers if hasattr(m.model, 'layers') else m.model.decoder.layers
                blocks = nn.ModuleList([layers[i] for i in range(n_freeze, len(layers))])
            self.specialist_blocks.append(blocks)

        # Router
        d_model = config.architecture.d_model
        self.moe_router = MoEFusion(n_experts, d_model, top_k=top_k)

        # Output head (shared — tied to embeddings in most models)
        if hasattr(modules[0], 'ln_f'):
            self.ln_f = modules[0].ln_f
            self.lm_head = modules[0].lm_head
        self._is_custom = hasattr(modules[0], 'blocks')

    def forward_shared(self, idx):
        """Run input through the shared frozen backbone."""
        model = self.shared_backbone
        if self._is_custom:
            B, T = idx.shape
            tok = model.tok_emb(idx)
            pos = model.pos_emb(torch.arange(T, device=idx.device))
            x = tok + pos
            for i in range(self.n_freeze):
                x = model.blocks[i](x)
            return x
        else:
            raise NotImplementedError("HF forward_shared — implement per model family")

    def forward(self, idx, targets=None):
        # Shared backbone
        shared_out = self.forward_shared(idx)

        # Run each specialist stack
        expert_outputs = []
        for specialist in self.specialist_blocks:
            x = shared_out.clone()
            for block in specialist:
                x = block(x)
            expert_outputs.append(x)

        # MoE routing
        fused, router_weights = self.moe_router(expert_outputs, shared_out)

        # Output projection
        if self._is_custom:
            fused = self.ln_f(fused)
            logits = self.lm_head(fused)
        else:
            raise NotImplementedError("HF output head — implement per model family")

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, router_weights


def train_moe_router(fused_model, train_dataset, config, device, router_steps=500):
    """Train only the MoE router on mixed-domain data. Everything else frozen."""
    fused_model = fused_model.to(device)

    # Freeze everything except router
    for name, param in fused_model.named_parameters():
        param.requires_grad = "moe_router" in name

    router_params = [p for p in fused_model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in router_params)
    print(f"  Training MoE router: {n_params:,} params for {router_steps} steps")

    optimizer = torch.optim.Adam(router_params, lr=config.fusion.router_lr)
    loader = DataLoader(train_dataset, batch_size=config.training.batch_size,
                        shuffle=True, drop_last=True, num_workers=0)
    data_iter = iter(loader)

    fused_model.train()
    losses = []
    for step in range(1, router_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        _, loss, weights = fused_model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            # Router utilization: what % of experts get >10% weight on average
            with torch.no_grad():
                expert_usage = (weights.mean(dim=(0, 1)) > 0.1).float().mean().item()
            print(f"    Router step {step}: loss={avg:.4f}, expert_usage={expert_usage:.1%}")

    return fused_model, losses


# ============================================================================
# Method 3: BTX-style (Meta baseline for comparison)
# ============================================================================

def fuse_btx_style(modules, config):
    """
    Branch-Train-MiX baseline: average attention weights, MoE-route FFN only.
    This is what Meta does in BTX (COLM 2024).
    No frozen layers — everything was trained independently.
    """
    # This is a structural comparison — BTX averages attn and MoE-routes FFN
    # For fair comparison, we implement this on MiniGPT
    fused = copy.deepcopy(modules[0])
    n = len(modules)

    if not hasattr(fused, 'blocks'):
        raise NotImplementedError("BTX comparison only for custom MiniGPT models")

    # Average ALL attention weights (BTX doesn't freeze anything)
    for layer_idx in range(len(fused.blocks)):
        # Average attention
        for name in ['qkv.weight', 'out_proj.weight']:
            parts = name.split('.')
            avg_weight = sum(
                _get_nested_attr(m.blocks[layer_idx].attn, parts)
                for m in modules
            ) / n
            _get_nested_attr(fused.blocks[layer_idx].attn, parts).copy_(avg_weight)

        # FFN layers become MoE experts — handled separately by MoE router
        # For simplicity in this comparison, we use the same MoE routing
        # but on ALL layers, not just unfrozen ones

    return fused


def _get_nested_attr(obj, parts):
    """Get nested attribute like 'qkv.weight' from an object."""
    for part in parts:
        obj = getattr(obj, part)
    return obj
