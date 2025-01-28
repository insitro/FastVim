# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from einops import rearrange, repeat
from torch import Tensor

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        init_layer_scale=None,
        token_size=None,
        use_norm_after_ssm=True,
        use_our_selective_scan=False,
        scan_order="Channel-First",
        collapse_method="mean",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  # 384
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.use_our_selective_scan = use_our_selective_scan
        self.num_of_rows = token_size[0]
        self.num_of_col = token_size[1]
        self.scan_order = scan_order
        self.collapse_method = collapse_method

        assert (
            self.num_of_rows % 2 == 0
        ), "num_of_rows needs to be even for this implementation since we do compress and expand"
        assert (
            self.num_of_col % 2 == 0
        ), "num_of_col needs to be even for this implementation since we do compress and expand"

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(
                init_layer_scale * torch.ones((d_model)), requires_grad=True
            )

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.use_norm_after_ssm = use_norm_after_ssm

        if self.use_norm_after_ssm is True:
            self.layernorm = nn.LayerNorm(self.d_inner)
        else:
            print("using No norm after SSM")

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(  # covers B, C, and delta linear projection
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, hidden_states, tokens_per_patch, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        A_b = -torch.exp(self.A_b_log.float())

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path and inference_params is None
        ):  # Doesn't support outputting the states
            print("write the code with autograd")

        else:
            if self.use_norm_after_ssm is True:
                x, z = xz.chunk(2, dim=1)
                x_flip = x.flip([-1])

                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
                x_flip = causal_conv1d_fn(
                    x=x_flip,
                    weight=rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    bias=self.conv1d_b.bias,
                    activation=self.activation,
                )

                if self.collapse_method == "mean":
                    if self.scan_order == "Spatial-First":
                        print("not implemented yet")

                    elif self.scan_order == "Channel-First":
                        if (self.layer_idx + 1) % 3 == 0:  # channelwise scan
                            pre_x_shape = (
                                -1,
                                self.d_inner,
                                self.num_of_rows * self.num_of_col,
                                tokens_per_patch,
                            )

                            x_compressed = x.reshape(pre_x_shape).mean(
                                dim=2
                            )  # Shape: (B, d, 14)
                            x_compressed_b = x_flip.reshape(pre_x_shape).mean(
                                dim=2
                            )  # Shape: (B, d, 14)

                        else:  # rowwise/colwise scan. Collapse along col in colwise still since we already rotate the input in model_mamba code so here row is actually col.
                            pre_x_shape = (
                                -1,
                                self.d_inner,
                                self.num_of_rows,
                                self.num_of_col * tokens_per_patch,
                            )

                            x_compressed = x.reshape(pre_x_shape).mean(
                                dim=3
                            )  # Shape: (B, d, 14)
                            x_compressed_b = x_flip.reshape(pre_x_shape).mean(
                                dim=3
                            )  # Shape: (B, d, 14)

                elif self.collapse_method == "max":
                    if self.scan_order == "Spatial-First":
                        print("not implemented yet")

                    elif self.scan_order == "Channel-First":
                        # x_compressed, x_compressed_b =  compress_noclstoken_channelfirst_maxpool(x, x_flip, self.layer_idx, self.num_of_rows, self.num_of_col, tokens_per_patch, self.d_inner)

                        if (self.layer_idx + 1) % 3 == 0:  # channelwise scan
                            pre_x_shape = (
                                -1,
                                self.d_inner,
                                self.num_of_rows * self.num_of_col,
                                tokens_per_patch,
                            )

                            x_compressed = (
                                x.reshape(pre_x_shape).max(dim=2).values
                            )  # Shape: (B, d, 14)
                            x_compressed_b = (
                                x_flip.reshape(pre_x_shape).max(dim=2).values
                            )  # Shape: (B, d, 14)

                        else:  # rowwise/colwise scan. Collapse along col in colwise still since we already rotate the input in model_mamba code so here row is actually col.
                            pre_x_shape = (
                                -1,
                                self.d_inner,
                                self.num_of_rows,
                                self.num_of_col * tokens_per_patch,
                            )

                            x_compressed = (
                                x.reshape(pre_x_shape).max(dim=3).values
                            )  # Shape: (B, d, 14)
                            x_compressed_b = (
                                x_flip.reshape(pre_x_shape).max(dim=3).values
                            )  # Shape: (B, d, 14)

                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

                x_dbl = self.x_proj(
                    rearrange(x_compressed, "b d l -> (b l) d")
                )  # (bl d)
                dt, B, C = torch.split(
                    x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
                )
                dt = self.dt_proj.weight @ dt.t()

                dt = rearrange(dt, "d (b l) -> b d l", b=batch)

                B = rearrange(B, "(b l) dstate -> b dstate l", b=batch).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", b=batch).contiguous()

                # with default selective scan
                if self.use_our_selective_scan is False:
                    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

                    out = selective_scan_fn(
                        x_compressed,
                        dt,
                        A,
                        B,
                        C,
                        D=None,  # D=None
                        z=None,
                        delta_bias=self.dt_proj.bias.float(),
                        delta_softplus=True,
                        return_last_state=False,
                    )

                    if self.scan_order == "Spatial-First":
                        print("not implemented yet")
                    else:
                        if (self.layer_idx + 1) % 3 == 0:  # channelwise scan
                            out = out.repeat(1, 1, self.num_of_rows * self.num_of_col)
                        else:
                            out = out.repeat_interleave(
                                self.num_of_col * tokens_per_patch, dim=2
                            )

                    out += self.D.float().unsqueeze(-1) * x
                # # with our selective scan
                # else:
                #     from faster_mamba_ssm.ops.selective_scan_interface import selective_scan_fn

                #     out = selective_scan_fn(
                #         x_compressed,
                #         dt_compressed,
                #         A,
                #         B_compressed,
                #         C_compressed,
                #         D=None,   # D=None
                #         z=None,
                #         delta_bias=None,
                #         delta_softplus=False,
                #         return_last_state=False,
                #         has_classtoken=self.if_cls_token,
                #         compression_factor=self.num_of_col,
                #     )  + self.D.float().unsqueeze(-1) * x

                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

                x_dbl_b = self.x_proj_b(
                    rearrange(x_compressed_b, "b d l -> (b l) d")
                )  # (bl d)
                dt_b, B_b, C_b = torch.split(
                    x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1
                )
                dt_b = self.dt_proj_b.weight @ dt_b.t()

                dt_b = rearrange(dt_b, "d (b l) -> b d l", b=batch)

                B_b = rearrange(B_b, "(b l) dstate -> b dstate l", b=batch).contiguous()
                C_b = rearrange(C_b, "(b l) dstate -> b dstate l", b=batch).contiguous()

                # with default selective scan
                if self.use_our_selective_scan is False:
                    out_b = selective_scan_fn(
                        x_compressed_b,
                        dt_b,
                        A_b,
                        B_b,
                        C_b,
                        D=None,
                        z=None,
                        delta_bias=self.dt_proj_b.bias.float(),
                        delta_softplus=True,
                        return_last_state=False,
                    )

                    if self.scan_order == "Spatial-First":
                        print("not implemented yet")
                        # out_b = expand_noclstoken_spatialfirst(out_b, self.layer_idx, self.num_of_rows, self.num_of_col, tokens_per_patch, self.d_inner)
                    else:
                        if (self.layer_idx + 1) % 3 == 0:  # channelwise scan
                            out_b = out_b.repeat(
                                1, 1, self.num_of_rows * self.num_of_col
                            )
                        else:
                            out_b = out_b.repeat_interleave(
                                self.num_of_col * tokens_per_patch, dim=2
                            )

                    out_b += self.D_b.float().unsqueeze(-1) * x_flip

                # # with our selective scan
                # else:
                #     out_b = selective_scan_fn(
                #         x_compressed_b,
                #         delta_compressed_b,
                #         A_b,
                #         B_compressed_b,
                #         C_compressed_b,
                #         D=None,
                #         z=None,
                #         delta_bias=None,
                #         delta_softplus=False,
                #         return_last_state=False,
                #         has_classtoken=self.if_cls_token,
                #         compression_factor=self.num_of_col,
                #     ) + self.D_b.float().unsqueeze(-1) * x_flip

                out = F.linear(
                    (
                        self.layernorm(
                            rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2
                        )
                        * F.silu(rearrange(z, "b d l -> b l d"))
                    ),
                    self.out_proj.weight,
                    self.out_proj.bias,
                )

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
