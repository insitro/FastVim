# Copyright (c) 2023, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


from mamba_ssm.ops.selective_scan_interface import (
    FastVim_mamba_inner_fn_no_out_proj_withoutZ,
)

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
        scanpath_type="rowwise",
        token_size=None,
        use_norm_after_ssm=True,
        use_our_selective_scan=False,
        collapse_method="mean",
        scaling_factor=1,
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
        self.scanpath_type = scanpath_type
        self.scaling_factor = scaling_factor

        self.collapse_method = collapse_method

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

        self.pre_x_shape = (-1, self.d_inner, self.num_of_rows, self.num_of_col)

    def forward(self, hidden_states, inference_params=None):
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
            self.use_fast_path
            and inference_params is None
            and self.collapse_method == "mean"
        ):  # doesn't support maxpool with fast path yet. use without fastpath code below.
            if xz.stride(-1) != 1:
                xz = xz.contiguous()

            x, z = xz.chunk(2, dim=1)

            if self.use_our_selective_scan is False:
                out = FastVim_mamba_inner_fn_no_out_proj_withoutZ(
                    x,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    num_of_col=self.num_of_col,
                    collapse_method=self.collapse_method,
                    scaling_factor=self.scaling_factor,
                    pre_x_shape=self.pre_x_shape,
                )

                out_b = FastVim_mamba_inner_fn_no_out_proj_withoutZ(
                    x.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                    num_of_col=self.num_of_col,
                    collapse_method=self.collapse_method,
                    scaling_factor=self.scaling_factor,
                    pre_x_shape=self.pre_x_shape,
                )

            else:
                print("To implement")

            if self.use_norm_after_ssm is True:
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
            else:
                out = F.linear(
                    rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2,
                    self.out_proj.weight,
                    self.out_proj.bias,
                )

        else:
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
                x_compressed = x.reshape(self.pre_x_shape).mean(
                    dim=3
                )  # Shape: (B, d, 14)
                x_compressed_b = x_flip.reshape(self.pre_x_shape).mean(
                    dim=3
                )  # Shape: (B, d, 14)

                if self.scaling_factor != 1:
                    x_compressed *= self.scaling_factor
                    x_compressed_b *= self.scaling_factor

            elif self.collapse_method == "max":
                x_compressed = (
                    x.reshape(self.pre_x_shape).max(dim=3).values
                )  # Shape: (B, d, 14)
                x_compressed_b = (
                    x_flip.reshape(self.pre_x_shape).max(dim=3).values
                )  # Shape: (B, d, 14)

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

            # new to incorporate autocast as Mamba paper
            x_proj_weight = self.x_proj.weight
            delta_proj_weight = self.dt_proj.weight
            if torch.is_autocast_enabled():
                x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
                delta_proj_weight = delta_proj_weight.to(
                    dtype=torch.get_autocast_gpu_dtype()
                )

            # x_dbl = self.x_proj(rearrange(x_compressed, "b d l -> (b l) d"))  # (bl d)
            x_dbl = F.linear(
                rearrange(x_compressed, "b d l -> (b l) d"), x_proj_weight
            )  # (bl d)
            dt, B, C = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )

            dt = delta_proj_weight @ dt.t()

            # x_dbl = self.x_proj(rearrange(x_compressed, "b d l -> (b l) d"))  # (bl d)
            # dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            # dt = self.dt_proj.weight @ dt.t()

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

                out = out.repeat_interleave(self.num_of_col, dim=2)

                out += self.D.float().unsqueeze(-1) * x

            # with our selective scan
            else:
                from faster_mamba_ssm.ops.selective_scan_interface import (
                    selective_scan_fn,
                )

                out = selective_scan_fn(
                    x,
                    x_compressed,
                    dt,
                    A,
                    B,
                    C,
                    D=self.D.float(),  # D=None
                    z=None,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

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

                out_b = out_b.repeat_interleave(
                    self.num_of_col, dim=2
                )  # Shape: (B, L, ED)

                out_b += self.D_b.float().unsqueeze(-1) * x_flip

            # with our selective scan
            else:
                out_b = selective_scan_fn(
                    x_flip,
                    x_compressed_b,
                    dt_b,
                    A_b,
                    B_b,
                    C_b,
                    D=self.D_b.float(),
                    z=None,
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                )

            if self.use_norm_after_ssm is True:
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
            else:
                out = F.linear(
                    (
                        (rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2)
                        * F.silu(rearrange(z, "b d l -> b l d"))
                    ),
                    self.out_proj.weight,
                    self.out_proj.bias,
                )

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out
