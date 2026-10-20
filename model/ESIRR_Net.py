"""
Event-Guided Single Image Reflection Removal
=============================================
Two-stage pipeline:

  Stage 1 — Reflection Estimation Network (REN):
      Inputs  : I_mixed  (B, 3, H, W)  — reflection-contaminated RGB, [0,1]
                E_voxel  (B, 5, H, W)  — temporal event voxel grid,   [-1,1]
      Outputs : R_estimate (B, 3, H, W) — estimated reflection layer

  Stage 2 — Background Restoration Network (BRN):
      Inputs  : I_mixed    (B, 3, H, W)
                R_estimate (B, 3, H, W)   from Stage 1
                → concatenated to (B, 6, H, W)
      Output  : T_estimate (B, 3, H, W) — clean background via residual pred.

This is the full model exactly as described in the paper: RGB/event dual
encoders, Event-Driven Feature Fusion (EFF) with a learned event confidence
map, and Cross-Stage Event Skip Connections from Stage 1 into Stage 2. The
model is trained fully end-to-end.

Design choices
--------------
- Building block : ResidualBlock (He et al., CVPR 2016)
- Normalisation  : GroupNorm (Wu & He, ECCV 2018)
- Stage 2 input  : pixel-level concatenation (I_mixed + R_est), 6 channels

Architecture credits
--------------------
Dual-branch independent encoders + EFF (adapted from the OFF module):
    Zou et al., "Seeing Behind Dynamic Occlusions with Event Cameras",
    arXiv:2307.15829v2, Sec. 3.1, Fig. 2(b), Table 6 & 8.

RFB head for Event Encoder:
    Weng et al., "Event-based Image Enhancement Under HDR Scenarios",
    ACCV 2024, Sec. 3.2, Fig. 3.
    Original RFB: Liu et al., ECCV 2018.

Two-stage pipeline with intermediate R supervision:
    Song et al., "Real-Scene Reflection Removal With RAW-RGB Image Pairs",
    IEEE TCSVT 2023, Sec. III-B & III-C, Table II.

U-Net encoder-decoder:
    Ronneberger et al., MICCAI 2015.

ResidualBlock:
    He et al., CVPR 2016.

GroupNorm:
    Wu & He, ECCV 2018.

Perceptual loss:
    Zhang et al., "Single Image Reflection Separation with Perceptual
    Losses", CVPR 2018.

Exclusion loss:
    Zhang et al., CVPR 2018, Eq. 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
#  Helpers
# ============================================================

def _norm(num_ch: int, kind: str = "gn") -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(num_ch)
    elif kind == "gn":
        groups = min(16, num_ch)
        while num_ch % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, num_ch)
    else:
        raise ValueError(f"Unknown norm: {kind}")


# ============================================================
#  Building Block
# ============================================================

class ResidualBlock(nn.Module):
    """
    Standard two-layer residual block with optional channel projection.
    Reference: He et al., CVPR 2016, Fig. 2.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 norm: str = "gn", dropout: float = 0.0):
        super().__init__()
        self.proj = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else None)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.n1    = _norm(out_ch, norm)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2    = _norm(out_ch, norm)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x) if self.proj is not None else x
        h = self.relu(self.n1(self.conv1(x)))
        h = self.drop(self.n2(self.conv2(h)))
        return self.relu(h + identity)


# ============================================================
#  RFB Head
# ============================================================

class RFBHead(nn.Module):
    """
    Receptive Field Block head for the Event Encoder.
    Reference: Weng et al., ACCV 2024; Liu et al., ECCV 2018.

    Three parallel branches:
        B1 — 1x1 conv               (point-wise)
        B2 — 1x1 -> 3x3 dil=2      (medium context)
        B3 — 1x1 -> 3x3 -> 3x3 d=3 (large context)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 3
        rem = out_ch - 2 * mid

        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, rem, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(rem, rem, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(rem, rem, 3, padding=3, dilation=3, bias=False),
            nn.ReLU(inplace=True))
        self.fuse = nn.Conv2d(mid + mid + rem, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(
            self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)),
            inplace=True)


# ============================================================
#  Event Confidence Map
# ============================================================

class EventConfidenceMap(nn.Module):
    """
    Learns a spatially-varying confidence score for the event voxel signal.

    Motivation
    ----------
    Unlike the OFF module (Zou et al.), which computes gating weights solely
    from encoded feature maps and implicitly assumes that event features are
    always reliable, our EFF module explicitly estimates an event confidence
    map directly from the raw voxel grid.

    This is critical in the reflection removal setting, where event signal
    quality varies dramatically across samples: reflections with low contrast
    or slow motion generate sparse voxels, and blindly trusting event features
    in these cases can degrade performance below the RGB-only baseline.

    Architecture
    ------------
    Input  : E_voxel (B, bins, H, W)  — raw voxel, values in [-1, 1]
    Step 1 : max-projection over temporal bins → (B, 1, H, W)
             Captures the peak event activity at each spatial position,
             ignoring the temporal ordering (which is already encoded in
             the EventEncoder features).
    Step 2 : Conv3x3 with spatial context → (B, 1, H, W)
             Smooths isolated noise spikes and captures local coherence.
    Step 3 : Sigmoid → C ∈ (0, 1) per pixel
             Converts raw activations to a probability-like confidence score.

    The confidence map is downsampled to match the spatial resolution of
    each encoder scale before being applied to the gate W.
    """
    def __init__(self):
        super().__init__()
        # Conv3x3 provides local spatial context — distinguishes coherent
        # event structures from isolated noise spikes.
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        self.sig  = nn.Sigmoid()

        # Initialise close to identity so early training is not disrupted:
        # output ≈ 0.5 everywhere initially → mild down-weighting of events.
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, e_voxel: torch.Tensor,
                target_hw: tuple = None) -> torch.Tensor:
        """
        Args:
            e_voxel   : (B, bins, H, W) raw voxel in [-1, 1]
            target_hw : (H', W') spatial size to downsample to.
                        If None, returns at full voxel resolution.
        Returns:
            C : (B, 1, H', W') confidence map in (0, 1)
        """
        # Step 1: max absolute activation over temporal bins → peak activity map
        max_proj = torch.max(torch.abs(e_voxel), dim=1, keepdim=True).values
        # Step 2 & 3: learnable spatial smoothing + sigmoid
        C = self.sig(self.conv(max_proj))
        # Step 4: downsample to encoder scale if needed
        if target_hw is not None and C.shape[2:] != target_hw:
            C = F.interpolate(C, size=target_hw,
                              mode='bilinear', align_corners=False)
        return C


# ============================================================
#  Encoders
# ============================================================

class FrameEncoder(nn.Module):
    """
    U-Net style encoder for I_mixed. 4 scales: H → H/2 → H/4 → H/8.
    References: Ronneberger et al., MICCAI 2015; He et al., CVPR 2016;
                Zou et al., arXiv 2307.15829.
    """
    def __init__(self, in_ch: int = 3, base: int = 48,
                 ch_cap: int = None, norm: str = "gn", dropout: float = 0.0):
        super().__init__()
        C  = base
        C4 = min(C*4, ch_cap) if ch_cap else C*4
        C8 = min(C*8, ch_cap) if ch_cap else C*8
        self.ch = (C, C*2, C4, C8)

        self.head  = nn.Sequential(nn.Conv2d(in_ch, C, 3, padding=1, bias=False),
                                   _norm(C, norm), nn.ReLU(inplace=True))
        self.s1    = ResidualBlock(C,   C,   norm, dropout)
        self.down2 = nn.Conv2d(C,   C*2, 3, stride=2, padding=1, bias=False)
        self.s2    = ResidualBlock(C*2, C*2, norm, dropout)
        self.down3 = nn.Conv2d(C*2, C4,  3, stride=2, padding=1, bias=False)
        self.s3    = ResidualBlock(C4,  C4,  norm, dropout)
        self.down4 = nn.Conv2d(C4,  C8,  3, stride=2, padding=1, bias=False)
        self.s4    = ResidualBlock(C8,  C8,  norm, dropout)

    def forward(self, x: torch.Tensor):
        f1 = self.s1(self.head(x))
        f2 = self.s2(self.down2(f1))
        f3 = self.s3(self.down3(f2))
        f4 = self.s4(self.down4(f3))
        return f1, f2, f3, f4


class EventEncoder(nn.Module):
    """
    Dedicated encoder for E_voxel (B, 5, H, W). Fully independent from
    FrameEncoder — sharing parameters degrades performance ~7 dB
    (Zou et al., Table 8 ablation).

    The raw voxel grid is first processed by an RFB head (Weng et al.,
    ACCV 2024) before entering the residual stack.
    """
    def __init__(self, in_bins: int = 5, base: int = 48,
                 ch_cap: int = None, norm: str = "gn",
                 dropout: float = 0.0):
        super().__init__()
        C  = base
        C4 = min(C*4, ch_cap) if ch_cap else C*4
        C8 = min(C*8, ch_cap) if ch_cap else C*8
        self.ch = (C, C*2, C4, C8)

        self.head  = RFBHead(in_bins, C)
        self.s1    = ResidualBlock(C,   C,   norm, dropout)
        self.down2 = nn.Conv2d(C,   C*2, 3, stride=2, padding=1, bias=False)
        self.s2    = ResidualBlock(C*2, C*2, norm, dropout)
        self.down3 = nn.Conv2d(C*2, C4,  3, stride=2, padding=1, bias=False)
        self.s3    = ResidualBlock(C4,  C4,  norm, dropout)
        self.down4 = nn.Conv2d(C4,  C8,  3, stride=2, padding=1, bias=False)
        self.s4    = ResidualBlock(C8,  C8,  norm, dropout)

    def forward(self, ev: torch.Tensor):
        e1 = self.s1(self.head(ev))
        e2 = self.s2(self.down2(e1))
        e3 = self.s3(self.down3(e2))
        e4 = self.s4(self.down4(e3))
        return e1, e2, e3, e4


# ============================================================
#  EFF — Event-Driven Feature Fusion
# ============================================================

class RFFModule(nn.Module):
    """
    Event-Driven Feature Fusion (EFF) at a single encoder scale.
    (Class kept as `RFFModule` for backward compatibility with existing
    checkpoints — the state_dict keys, e.g. `rff1.conv_tilde...`, must stay
    unchanged. Conceptually / in the paper this module is called EFF.)

    Base structure adapted from the RFF module in:
        Zou et al., "Seeing Behind Dynamic Occlusions with Event Cameras",
        arXiv:2307.15829v2, Sec. 3.1, Fig. 2(b), Eqs. 4-7.

    Key difference from Zou et al.'s RFF module
    ---------------------------------------------
    Their RFF computes gating weights W solely from encoded feature maps,
    implicitly assuming event features are always reliable.

    Our EFF additionally modulates W with an Event Confidence Map C computed
    directly from the raw voxel grid:

        W_final = W * C_downsampled

    This allows the network to gracefully degrade to RGB-only processing
    (W_final → 0) when the voxel is sparse, and to fully trust events
    (W_final ≈ W) when the voxel is dense and informative.

    See EventConfidenceMap for the full motivation.

    Steps:
        1. F̃ = G(F_event, F_frame)             — cross-modal local estimate
        2. W  = σ(G(F̃, F̄_prev))               — spatial gate ∈ [0,1]
        3. C  = EventConfidenceMap(E_voxel)      — event reliability ∈ (0,1)
        4. W_final = W * C                       — confidence-modulated gate
        5. F̂  = (1 - W_final)·F_frame
                + W_final·F_event               — soft gated mix
        6. F_fused = G(F̂, F̄_prev)              — final output
    """
    def __init__(self, ch: int, ch_prev: int = None, norm: str = "gn"):
        super().__init__()

        # Step 1 — cross-modal local estimate
        self.conv_tilde = nn.Sequential(
            nn.Conv2d(ch*2, ch, 3, padding=1, bias=False),
            _norm(ch, norm), nn.ReLU(inplace=True),
            nn.Conv2d(ch,   ch, 3, padding=1, bias=False),
            _norm(ch, norm), nn.ReLU(inplace=True),
        )

        # Step 2a — gate without cross-scale context (scale 1)
        self.gate_no_prev = nn.Sequential(
            nn.Conv2d(ch*2, ch, 3, padding=1, bias=False),
            _norm(ch, norm), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.Sigmoid(),
        )
        self.out_no_prev = nn.Sequential(
            nn.Conv2d(ch*2, ch, 3, padding=1, bias=False),
            _norm(ch, norm), nn.ReLU(inplace=True),
        )

        if ch_prev is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_prev, ch, 3, stride=2, padding=1, bias=False),
                _norm(ch, norm), nn.ReLU(inplace=True),
            )
            # Step 2b — gate with cross-scale context (scales 2-4)
            self.gate_with_prev = nn.Sequential(
                nn.Conv2d(ch*2, ch, 3, padding=1, bias=False),
                _norm(ch, norm), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.Sigmoid(),
            )
            self.out_with_prev = nn.Sequential(
                nn.Conv2d(ch*2, ch, 3, padding=1, bias=False),
                _norm(ch, norm), nn.ReLU(inplace=True),
            )
        else:
            self.downsample     = None
            self.gate_with_prev = None
            self.out_with_prev  = None

        # Step 3 — event confidence map (our contribution vs Zou et al.'s RFF)
        # One EventConfidenceMap per scale: each learns its own
        # spatial sensitivity appropriate to the scale's resolution.
        self.conf_map = EventConfidenceMap()

    def forward(self, f_frame: torch.Tensor,
                f_event: torch.Tensor,
                e_voxel: torch.Tensor,
                prev_fused: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            f_frame   : (B, ch,      H,   W)  frame encoder features at scale j
            f_event   : (B, ch,      H,   W)  event encoder features at scale j
            e_voxel   : (B, bins,    H0,  W0) raw voxel at FULL resolution,
                        used to compute the confidence map.
            prev_fused: (B, ch_prev, 2H, 2W)  fused features from scale j-1,
                        or None at scale 1
        Returns:
            fused: (B, ch, H, W)
        """
        if prev_fused is not None and self.downsample is None:
            raise ValueError("RFFModule: prev_fused passed but ch_prev not set.")

        # Step 1
        f_tilde = self.conv_tilde(torch.cat([f_event, f_frame], dim=1))

        # Step 2 — compute raw gate W
        if prev_fused is None:
            W = self.gate_no_prev(torch.cat([f_tilde, f_frame], dim=1))
        else:
            f_prev = self.downsample(prev_fused)
            if f_prev.shape[2:] != f_frame.shape[2:]:
                f_prev = F.interpolate(f_prev, size=f_frame.shape[2:],
                                       mode='bilinear', align_corners=False)
            W = self.gate_with_prev(torch.cat([f_tilde, f_prev], dim=1))

        # Steps 3-4 — modulate W with event confidence
        C = self.conf_map(e_voxel, target_hw=f_frame.shape[2:])
        W = W * C   # W_final = W * C, both in [0,1]

        # Step 5 — soft gated mix
        f_hat = (1.0 - W) * f_frame + W * f_event

        # Step 6 — final output
        if prev_fused is None:
            fused = self.out_no_prev(torch.cat([f_hat, f_frame], dim=1))
        else:
            fused = self.out_with_prev(torch.cat([f_hat, f_prev], dim=1))

        return fused


# ============================================================
#  Stage 1 Decoder
# ============================================================

class Stage1Decoder(nn.Module):
    """
    U-Net decoder for Stage 1. Produces R_estimate ∈ [0,1].
    Reference: Ronneberger et al., MICCAI 2015;
               Song et al., IEEE TCSVT 2023, Table II.
    """
    def __init__(self, base: int = 48, ch_cap: int = None,
                 norm: str = "gn", dropout: float = 0.0):
        super().__init__()
        C  = base
        C4 = min(C*4, ch_cap) if ch_cap else C*4
        C8 = min(C*8, ch_cap) if ch_cap else C*8

        self.up3    = nn.ConvTranspose2d(C8,  C4,  2, stride=2)
        self.d3     = ResidualBlock(C4*2, C4,  norm, dropout)
        self.up2    = nn.ConvTranspose2d(C4,  C*2, 2, stride=2)
        self.d2     = ResidualBlock(C4,   C*2, norm, dropout)
        self.up1    = nn.ConvTranspose2d(C*2, C,   2, stride=2)
        self.d1     = ResidualBlock(C*2,  C,   norm, dropout)
        self.head_R = nn.Sequential(
            nn.Conv2d(C, C//2, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(C//2, 3, 1), nn.Sigmoid())

    def forward(self, ff1, ff2, ff3, ff4):
        x = self.d3(torch.cat([self.up3(ff4), ff3], dim=1))
        x = self.d2(torch.cat([self.up2(x),   ff2], dim=1))
        x = self.d1(torch.cat([self.up1(x),   ff1], dim=1))
        return self.head_R(x)


# ============================================================
#  Stage 2 Encoder + Decoder
# ============================================================

class Stage2Encoder(nn.Module):
    """
    U-Net encoder for Stage 2. Input: cat(I_mixed, R_estimate) = 6ch.
    Reference: Song et al., IEEE TCSVT 2023, Sec. III-B2.
    """
    def __init__(self, in_ch: int = 6, base: int = 48,
                 ch_cap: int = None, norm: str = "gn", dropout: float = 0.0):
        super().__init__()
        C  = base
        C4 = min(C*4, ch_cap) if ch_cap else C*4
        C8 = min(C*8, ch_cap) if ch_cap else C*8

        self.head  = nn.Sequential(nn.Conv2d(in_ch, C, 3, padding=1, bias=False),
                                   _norm(C, norm), nn.ReLU(inplace=True))
        self.s1    = ResidualBlock(C,   C,   norm, dropout)
        self.down2 = nn.Conv2d(C,   C*2, 3, stride=2, padding=1, bias=False)
        self.s2    = ResidualBlock(C*2, C*2, norm, dropout)
        self.down3 = nn.Conv2d(C*2, C4,  3, stride=2, padding=1, bias=False)
        self.s3    = ResidualBlock(C4,  C4,  norm, dropout)
        self.down4 = nn.Conv2d(C4,  C8,  3, stride=2, padding=1, bias=False)
        self.s4    = ResidualBlock(C8,  C8,  norm, dropout)

    def forward(self, x: torch.Tensor):
        g1 = self.s1(self.head(x))
        g2 = self.s2(self.down2(g1))
        g3 = self.s3(self.down3(g2))
        g4 = self.s4(self.down4(g3))
        return g1, g2, g3, g4


class Stage2Decoder(nn.Module):
    """
    U-Net decoder for Stage 2. Predicts residual ΔT:
        T_estimate = clamp(I_mixed + ΔT, 0, 1)
    Reference: Zou et al., arXiv 2307.15829, Sec. 3.1.

    Cross-Stage Event Skip Connections
    -----------------------------------
    Stage 2 decoder receives additional skip connections from the Stage 1
    fused event-RGB features (ff1, ff2, ff3) produced by the EFF modules.
    These are injected as residuals into the corresponding decoder scales
    via lightweight 1x1 adapters.

    Motivation: Stage 2 only receives cat(I_mixed, R_estimate). When
    R_estimate is imperfect (which is always the case), Stage 2 has no way
    to recover event information that could guide the restoration of T.
    The cross-stage connections propagate the event-guided spatial
    structure directly into Stage 2 without re-running the encoders, at
    negligible parameter cost (3 × Conv1x1 adapters).

    The injection is additive (residual), so when ff* features carry no
    useful event information (sparse voxel case, confidence map ≈ 0), the
    adapters can learn to output near-zero.

    Adapter design: Conv1x1 with zero-init weights and zero-init bias.
    This ensures the cross-stage path starts as an identity bypass (zero
    contribution) and is activated gradually during training.
    """
    def __init__(self, base: int = 48, ch_cap: int = None,
                 norm: str = "gn", dropout: float = 0.0):
        super().__init__()
        C  = base
        C4 = min(C*4, ch_cap) if ch_cap else C*4
        C8 = min(C*8, ch_cap) if ch_cap else C*8

        self.up3    = nn.ConvTranspose2d(C8,  C4,  2, stride=2)
        self.d3     = ResidualBlock(C4*2, C4,  norm, dropout)
        self.up2    = nn.ConvTranspose2d(C4,  C*2, 2, stride=2)
        self.d2     = ResidualBlock(C4,   C*2, norm, dropout)
        self.up1    = nn.ConvTranspose2d(C*2, C,   2, stride=2)
        self.d1     = ResidualBlock(C*2,  C,   norm, dropout)
        self.head_T = nn.Sequential(
            nn.Conv2d(C, C//2, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(C//2, 3, 1), nn.Tanh())

        # Lightweight 1x1 adapters that project Stage 1 fused features into
        # the channel space of the corresponding Stage 2 decoder scale.
        # Scales injected: scale 1 (C), scale 2 (C*2), scale 3 (C4). Scale 4
        # (bottleneck C8) is intentionally skipped: at that resolution the
        # fused features are highly compressed and the Stage 2 bottleneck
        # already captures global context.
        #
        # Zero-init: adapters start as identity bypasses (zero output), so
        # the cross-stage path does not disturb the rest of the network at
        # the start of training and is activated gradually.
        self.cs_adapt1 = nn.Conv2d(C,   C,   1, bias=True)
        self.cs_adapt2 = nn.Conv2d(C*2, C*2, 1, bias=True)
        self.cs_adapt3 = nn.Conv2d(C4,  C4,  1, bias=True)

        nn.init.zeros_(self.cs_adapt1.weight)
        nn.init.zeros_(self.cs_adapt1.bias)
        nn.init.zeros_(self.cs_adapt2.weight)
        nn.init.zeros_(self.cs_adapt2.bias)
        nn.init.zeros_(self.cs_adapt3.weight)
        nn.init.zeros_(self.cs_adapt3.bias)

    def forward(self, g1: torch.Tensor, g2: torch.Tensor,
                g3: torch.Tensor, g4: torch.Tensor,
                i_mixed: torch.Tensor,
                ff1: torch.Tensor, ff2: torch.Tensor, ff3: torch.Tensor):
        """
        Args:
            g1, g2, g3, g4 : Stage 2 encoder features at scales 1-4.
            i_mixed        : (B, 3, H, W) input mixture image, used for
                             the residual prediction clamp.
            ff1, ff2, ff3  : Stage 1 fused event-RGB features at scales 1-3.
        Returns:
            T_estimate : (B, 3, H, W) clean background in [0, 1].
            delta_T    : (B, 3, H, W) predicted residual in [-1, 1].
        """
        # Inject Stage 1 fused event-RGB features as additive residuals.
        # ff* and g* are produced at the same encoder resolution (both
        # encoders use identical stride patterns), so no interpolation
        # is needed.
        g1 = g1 + self.cs_adapt1(ff1)
        g2 = g2 + self.cs_adapt2(ff2)
        g3 = g3 + self.cs_adapt3(ff3)

        x = self.d3(torch.cat([self.up3(g4), g3], dim=1))
        x = self.d2(torch.cat([self.up2(x),  g2], dim=1))
        x = self.d1(torch.cat([self.up1(x),  g1], dim=1))
        delta_T = self.head_T(x)                     # (B,3,H,W), range [-1,1]
        T_est = torch.clamp(i_mixed + delta_T, 0.0, 1.0)
        return T_est, delta_T


# ============================================================
#  Full Two-Stage Model
# ============================================================

class EventGuidedSIRR(nn.Module):
    """
    Event-Guided Single Image Reflection Removal — full model.

    This is the architecture reported in the paper: RFB event head +
    confidence-modulated EFF fusion + cross-stage skip connections,
    trained fully end-to-end (Stage 1 and Stage 2 share gradients).
    """

    def __init__(self, base: int = 48, voxel_bins: int = 5,
                 ch_cap: int = None, norm: str = "gn", dropout: float = 0.0):
        super().__init__()
        C  = base
        C4 = min(C*4, ch_cap) if ch_cap else C*4
        C8 = min(C*8, ch_cap) if ch_cap else C*8

        # ---- Stage 1 ----
        self.frame_enc = FrameEncoder(in_ch=3, base=C, ch_cap=ch_cap,
                                      norm=norm, dropout=dropout)
        self.event_enc = EventEncoder(in_bins=voxel_bins, base=C, ch_cap=ch_cap,
                                      norm=norm, dropout=dropout)

        # EFF modules — one per encoder scale.
        self.rff1 = RFFModule(C,   ch_prev=None, norm=norm)
        self.rff2 = RFFModule(C*2, ch_prev=C,    norm=norm)
        self.rff3 = RFFModule(C4,  ch_prev=C*2,  norm=norm)
        self.rff4 = RFFModule(C8,  ch_prev=C4,   norm=norm)

        self.s1_dec = Stage1Decoder(base=C, ch_cap=ch_cap,
                                    norm=norm, dropout=dropout)

        # ---- Stage 2 ----
        self.s2_enc = Stage2Encoder(in_ch=6, base=C, ch_cap=ch_cap,
                                    norm=norm, dropout=dropout)
        self.s2_dec = Stage2Decoder(base=C, ch_cap=ch_cap,
                                    norm=norm, dropout=dropout)

    def _stage1(self, i_mixed: torch.Tensor, e_voxel: torch.Tensor):
        """
        Run Stage 1 (Reflection Estimation Network).

        Returns:
            R_est        : (B, 3, H, W) estimated reflection layer.
            ff1, ff2, ff3: fused event-RGB features at scales 1, 2, 3,
                           passed on to Stage 2 for the cross-stage
                           skip connections.
        """
        f1, f2, f3, f4 = self.frame_enc(i_mixed)
        e1, e2, e3, e4 = self.event_enc(e_voxel)

        ff1 = self.rff1(f1, e1, e_voxel, prev_fused=None)
        ff2 = self.rff2(f2, e2, e_voxel, prev_fused=ff1)
        ff3 = self.rff3(f3, e3, e_voxel, prev_fused=ff2)
        ff4 = self.rff4(f4, e4, e_voxel, prev_fused=ff3)

        R_est = self.s1_dec(ff1, ff2, ff3, ff4)

        # ff4 (bottleneck) is intentionally excluded from cross-stage
        # injection — see Stage2Decoder docstring.
        return R_est, ff1, ff2, ff3

    def _stage2(self, i_mixed: torch.Tensor, R_est: torch.Tensor,
                ff1: torch.Tensor, ff2: torch.Tensor, ff3: torch.Tensor):
        """Run Stage 2 (Background Restoration Network)."""
        g1, g2, g3, g4 = self.s2_enc(torch.cat([i_mixed, R_est], dim=1))
        return self.s2_dec(g1, g2, g3, g4, i_mixed, ff1, ff2, ff3)

    def forward(self, i_mixed: torch.Tensor, e_voxel: torch.Tensor) -> dict:
        """
        Full end-to-end forward pass.

        Args:
            i_mixed : (B, 3, H, W) reflection-contaminated RGB input, [0,1]
            e_voxel : (B, voxel_bins, H, W) synchronized event voxel grid
        Returns:
            dict with keys "R_estimate", "T_estimate", "residual".
        """
        R_est, ff1, ff2, ff3 = self._stage1(i_mixed, e_voxel)
        T_est, delta_T = self._stage2(i_mixed, R_est, ff1, ff2, ff3)
        return {"R_estimate": R_est, "T_estimate": T_est, "residual": delta_T}


# ============================================================
#  Losses
# ============================================================

class PerceptualLoss(nn.Module):
    """
    VGG-19 perceptual loss on relu1_2 and relu2_2.
    Reference: Zhang et al., CVPR 2018; Song et al., IEEE TCSVT 2023.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.slice1 = nn.Sequential(*list(vgg.features)[:5]).eval()
        self.slice2 = nn.Sequential(*list(vgg.features)[5:10]).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred   = (pred   - self.mean) / self.std
        target = (target - self.mean) / self.std
        p1 = self.slice1(pred);   t1 = self.slice1(target)
        p2 = self.slice2(p1);     t2 = self.slice2(t1)
        return F.mse_loss(p1, t1) + F.mse_loss(p2, t2)


def exclusion_loss(T_est: torch.Tensor, R_est: torch.Tensor,
                   levels: int = 3) -> torch.Tensor:
    """
    Penalises spatial correlation between gradients of T and R.
    Reference: Zhang et al., CVPR 2018, Eq. 4.
    """
    loss = 0.0
    for _ in range(levels):
        gTx = torch.sigmoid(T_est[:, :, :, 1:] - T_est[:, :, :, :-1])
        gTy = torch.sigmoid(T_est[:, :, 1:, :] - T_est[:, :, :-1, :])
        gRx = torch.sigmoid(R_est[:, :, :, 1:] - R_est[:, :, :, :-1])
        gRy = torch.sigmoid(R_est[:, :, 1:, :] - R_est[:, :, :-1, :])
        loss += torch.mean(gTx * gRx) + torch.mean(gTy * gRy)
        T_est = F.avg_pool2d(T_est, 2, 2)
        R_est = F.avg_pool2d(R_est, 2, 2)
    return loss / levels


class SIRRLoss(nn.Module):
    """
    Combined loss for both training stages.

    Stage-1 (on R_estimate):  L1 + Perceptual
    Stage-2 (on T_estimate):  L1 + Perceptual
    Cross-stage:              Exclusion loss (T_est, R_est) — Zhang et al., CVPR 2018

    Total:
        L = L_s1 + L_s2 + w_excl * L_excl
    """
    def __init__(self,
                 w_l1_R:   float = 1.0,
                 w_perc_R: float = 0.1,
                 w_l1_T:   float = 1.0,
                 w_perc_T: float = 0.1,
                 w_excl:   float = 0.02):
        super().__init__()
        self.w_l1_R   = w_l1_R
        self.w_perc_R = w_perc_R
        self.w_l1_T   = w_l1_T
        self.w_perc_T = w_perc_T
        self.w_excl   = w_excl
        self.perc     = PerceptualLoss()

    def forward(self, preds: dict, targets: dict) -> dict:
        R_est = preds["R_estimate"]
        T_est = preds["T_estimate"]
        R_gt  = targets["R_warped_gt"]
        T_gt  = targets["T_gt"]

        # Stage-1 loss
        l1_R   = F.l1_loss(R_est, R_gt)
        perc_R = self.perc(R_est, R_gt)
        L_s1   = self.w_l1_R * l1_R + self.w_perc_R * perc_R

        # Stage-2 loss
        l1_T   = F.l1_loss(T_est, T_gt)
        perc_T = self.perc(T_est, T_gt)
        L_s2   = self.w_l1_T * l1_T + self.w_perc_T * perc_T

        # Cross-stage exclusion loss
        L_excl = exclusion_loss(T_est, R_est)

        total = L_s1 + L_s2 + self.w_excl * L_excl

        return {
            "total":  total,
            "L_s1":   L_s1,   "L_s2":   L_s2,   "L_excl": L_excl,
            "l1_R":   l1_R,   "perc_R": perc_R,
            "l1_T":   l1_T,   "perc_T": perc_T,
        }


# ============================================================
#  Sanity check
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    B, H, W = 2, 320, 320
    i_mixed = torch.rand(B, 3, H, W, device=device)
    e_voxel = torch.rand(B, 5, H, W, device=device) * 2 - 1
    targets = {
        "R_warped_gt": torch.rand(B, 3, H, W, device=device),
        "T_gt":        torch.rand(B, 3, H, W, device=device),
    }

    model = EventGuidedSIRR(base=48, voxel_bins=5, ch_cap=48*4, norm="gn").to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out    = model(i_mixed, e_voxel)
    losses = SIRRLoss().to(device)(out, targets)

    print(f"Params : {n/1e6:.2f} M")
    print(f"Loss   : {losses['total'].item():.4f}  "
          f"(L_s1={losses['L_s1'].item():.4f}  "
          f"L_s2={losses['L_s2'].item():.4f}  "
          f"L_excl={losses['L_excl'].item():.4f})")
    for k, v in out.items():
        print(f"  {k:12s}: {tuple(v.shape)}")