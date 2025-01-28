# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple_faster import Mamba
from timm.layers import DropPath, lecun_normal_, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg, _load_weights
from torch import Tensor

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import torch.nn.functional as F
from mmdet.registry import MODELS
from mmseg.models.builder import BACKBONES


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        strict_img_size=True,
        dynamic_img_pad=False,
        scanpath_type="rowwise",  # rowwise denotes Pool_col in FastVim paper
    ):
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)

        if scanpath_type == "colwise":  # colwise denotes Pool_row in FastVim paper
            self.grid_size = tuple(
                [
                    self.img_size[1] // self.patch_size[1],
                    self.img_size[0] // self.patch_size[0],
                ]
            )
        elif scanpath_type == "rowwise":  # rowwise denotes Pool_col in FastVim paper
            self.grid_size = tuple(
                [
                    self.img_size[0] // self.patch_size[0],
                    self.img_size[1] // self.patch_size[1],
                ]
            )

        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.scanpath_type = scanpath_type

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        if self.strict_img_size:
            assert (
                H == self.img_size[0]
            ), f"Input height ({H}) doesn't match model ({self.img_size[0]})."
            assert (
                W == self.img_size[1]
            ), f"Input width ({W}) doesn't match model ({self.img_size[1]})."
        elif not self.dynamic_img_pad:
            assert (
                H % self.patch_size[0] == 0
            ), f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
            assert (
                W % self.patch_size[1] == 0
            ), f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."

        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.proj(x)

        if self.scanpath_type == "colwise":  # colwise denotes Pool_row in FastVim paper
            x = x.transpose(2, 3)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        rotate_every_block=True,  # rotate denotes transpose operation to do alternate row-col pooling
        layer_idx=None,
        token_size=None,
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
        self.rotate_every_block = rotate_every_block
        self.layer_idx = layer_idx
        self.token_size = token_size
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
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
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        if (
            self.rotate_every_block is True and self.layer_idx % 2 != 0
        ):  # to alternate the pooling across col or across row and aligning with conv1d scanpath
            B, M, _ = hidden_states.shape

            hidden_states = hidden_states.reshape(
                (B, self.token_size[0], self.token_size[1], -1)
            )
            hidden_states = hidden_states.transpose(1, 2).reshape((B, M, -1))

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        if (
            self.rotate_every_block is True and self.layer_idx % 2 != 0
        ):  # to alternate the pooling across col or across row and aligning with conv1d scanpath
            hidden_states = hidden_states.reshape(
                (B, self.token_size[1], self.token_size[0], -1)
            )
            hidden_states = hidden_states.transpose(1, 2).reshape((B, M, -1))

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    init_layer_scale=None,
    scanpath_type="rowwise",
    use_norm_after_ssm=True,
    rotate_every_block=True,
    collapse_method="mean",  # collapse denotes pooling.
    token_size=None,
    use_our_selective_scan=False,
    scaling_factor=1,  # we use 0.25 when transfer from MAE with masking ration 75%. Based on masking ratio, change the scaling factor
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    if rotate_every_block is True and layer_idx % 2 != 0:
        mixer_cls = partial(
            Mamba,
            layer_idx=layer_idx,
            init_layer_scale=init_layer_scale,
            scanpath_type=scanpath_type,
            use_norm_after_ssm=use_norm_after_ssm,
            token_size=[
                token_size[1],
                token_size[0],
            ],  # pool across row - Pool_row. Col-wise interation
            collapse_method=collapse_method,
            use_our_selective_scan=use_our_selective_scan,
            scaling_factor=scaling_factor,
            **ssm_cfg,
            **factory_kwargs,
        )
    else:
        mixer_cls = partial(
            Mamba,
            layer_idx=layer_idx,
            init_layer_scale=init_layer_scale,
            scanpath_type=scanpath_type,
            use_norm_after_ssm=use_norm_after_ssm,
            token_size=token_size,  # pool across col - Pool_col. Row-wise interation
            collapse_method=collapse_method,
            use_our_selective_scan=use_our_selective_scan,
            scaling_factor=scaling_factor,
            **ssm_cfg,
            **factory_kwargs,
        )

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        rotate_every_block=rotate_every_block,
        layer_idx=layer_idx,
        token_size=token_size,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=192,
        channels=3,
        num_classes=1000,
        ssm_cfg=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        final_pool_type="none",
        if_abs_pos_embed=True,
        init_layer_scale=None,
        embed_layer=PatchEmbed,
        scanpath_type="rowwise",  # rowwise denotes Pool_col in FastVim paper
        use_norm_after_ssm=True,
        rotate_every_block=True,  # for acorss col and row alternate pooling
        collapse_method="mean",
        use_our_selective_scan=False,  # to use our kernel
        scaling_factor=1,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.rotate_every_block = rotate_every_block

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=embed_dim,
            strict_img_size=False,
            dynamic_img_pad=True,
            scanpath_type=scanpath_type,
        )

        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.token_size = self.patch_embed.grid_size

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=inter_dpr[i],
                    init_layer_scale=init_layer_scale,
                    scanpath_type=scanpath_type,
                    use_norm_after_ssm=use_norm_after_ssm,
                    rotate_every_block=rotate_every_block,
                    collapse_method=collapse_method,
                    token_size=self.token_size,
                    use_our_selective_scan=use_our_selective_scan,
                    scaling_factor=scaling_factor,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None, out_indices=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        _, M, _ = x.shape

        if self.if_abs_pos_embed:
            H, W = math.ceil(H / self.patch_size), math.ceil(W / self.patch_size)
            if H != self.token_size[0] or W != self.token_size[1]:
                # downstream tasks such as det and seg may have various input resolutions
                pos_embed = MM_FastVim.resize_pos_embed(
                    self.pos_embed, (H, W), self.token_size, "bicubic"
                )
            else:
                pos_embed = self.pos_embed

            x = x + pos_embed
            x = self.pos_drop(x)

        outs = []

        # mamba impl
        residual = None
        hidden_states = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if out_indices is not None and layer_idx in out_indices:
                outs.append(hidden_states)

        if out_indices is not None:
            assert len(outs) == len(out_indices)
            return outs, (H, W)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.final_pool_type == "none":
            return hidden_states[:, -1, :]
        elif self.final_pool_type == "mean":
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == "max":
            return hidden_states
        elif self.final_pool_type == "all":
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None):
        x = self.forward_features(x, inference_params)
        if return_features:
            return x
        x = self.head(x)
        if self.final_pool_type == "max":
            x = x.max(dim=1)[0]
        return x


@BACKBONES.register_module()
@MODELS.register_module()
class MM_FastVim(VisionMamba):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=192,
        depth=24,
        pretrained=None,
        out_indices=[5, 11, 17, 23],
        scanpath_type="rowwise",
        load_ema=True,
        **kwargs,
    ):
        super().__init__(
            img_size,
            patch_size,
            stride,
            depth,
            embed_dim,
            in_chans,
            **kwargs,
        )
        self.load_ema = load_ema
        self.scanpath_type = scanpath_type

        self.out_indices = out_indices
        for i in range(len(out_indices)):
            layer = nn.LayerNorm(self.embed_dim)
            layer_name = f"outnorm_{i}"
            self.add_module(layer_name, layer)

        # del the parent class's head
        del self.head
        del self.norm_f

        self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained):
        if pretrained is None:
            return

        if self.load_ema is True:
            try:
                state_dict = torch.load(pretrained, map_location="cpu")[
                    "state_dict_ema"
                ]
            except Exception as e:
                print(f"ema state doesn't exists, loading non-ema state_dict : {e}")
                state_dict = torch.load(pretrained, map_location="cpu")["state_dict"]
        else:
            state_dict = torch.load(pretrained, map_location="cpu")["state_dict"]

        state_dict_model = {
            key.replace("backbone.", ""): value for key, value in state_dict.items()
        }

        if "pos_embed" in state_dict_model:
            pos_size = int(math.sqrt(state_dict_model["pos_embed"].shape[1]))

            state_dict_model["pos_embed"] = self.resize_pos_embed(
                state_dict_model["pos_embed"],
                self.token_size,
                (pos_size, pos_size),
                "bicubic",
                self.scanpath_type,
            )

        print(
            self.patch_embed.patch_size[-1],
            state_dict_model["patch_embed.proj.weight"].shape[-1],
        )
        if (
            self.patch_embed.patch_size[-1]
            != state_dict_model["patch_embed.proj.weight"].shape[-1]
        ):
            state_dict_model.pop("patch_embed.proj.weight")
            state_dict_model.pop("patch_embed.proj.bias")

        res = self.load_state_dict(state_dict_model, strict=False)
        print(res)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shape, pos_shape, mode, scanpath_type):
        from mmseg.models.utils import resize

        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shape (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, "shape of pos_embed must be [B, L, C]"
        pos_h, pos_w = pos_shape
        pos_embed_weight = pos_embed
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]
        ).permute(0, 3, 1, 2)

        if scanpath_type == "colwise":
            pos_embed_weight = pos_embed_weight.transpose(2, 3)

        pos_embed_weight = resize(
            pos_embed_weight, size=input_shape, align_corners=False, mode=mode
        )

        if scanpath_type == "colwise":
            pos_embed_weight = pos_embed_weight.transpose(2, 3)

        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return pos_embed_weight

    def forward(self, x):
        C = self.embed_dim
        outs, (H, W) = self.forward_features(x, out_indices=self.out_indices)
        outs = [getattr(self, f"outnorm_{i}")(o) for i, o in enumerate(outs)]
        outs = [o.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous() for o in outs]
        if len(self.out_indices) == 1:
            return outs[0]
        return outs


######################################################################
@register_model
def vim_tiny_patch16_224_final_pool_mean_abs_pos_embed_with_noclstok_div2(
    pretrained=False, img_size=224, patch_size=16, stride=16, **kwargs
):
    model = VisionMamba(
        img_size=img_size,
        patch_size=patch_size,
        stride=stride,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def vim_small_patch16_224_final_pool_mean_abs_pos_embed_with_noclstok_div2(
    pretrained=False, img_size=224, patch_size=16, stride=16, **kwargs
):
    model = VisionMamba(
        img_size=img_size,
        patch_size=patch_size,
        stride=stride,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def vim_base_patch16_224_final_pool_mean_abs_pos_embed_with_noclstok_div2(
    pretrained=False,
    img_size=224,
    patch_size=16,
    stride=16,
    pretrained_checkpoint_path=None,
    **kwargs,
):
    model = VisionMamba(
        img_size=img_size,
        patch_size=patch_size,
        stride=stride,
        embed_dim=768,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        **kwargs,
    )
    torch.cuda.empty_cache()
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    if pretrained_checkpoint_path is not None:  # for transfer from MAE/SSL
        checkpoint = torch.load(pretrained_checkpoint_path)["state_dict"]
        state_dict_model = {
            key.replace("backbone.", ""): value for key, value in checkpoint.items()
        }

        if "pos_embed" in state_dict_model:
            print("true", state_dict_model["pos_embed"].shape)
            orig_size = int(math.sqrt(state_dict_model["pos_embed"].shape[1]))
            new_size = int(img_size // patch_size)

            # assuming no class_token in MAE pretraining
            if orig_size != new_size:
                embedding_size = state_dict_model["pos_embed"].shape[-1]
                print(
                    "Position interpolate from %dx%d to %dx%d"
                    % (orig_size, orig_size, new_size, new_size)
                )
                # only the position tokens are interpolated
                pos_tokens = state_dict_model["pos_embed"]
                pos_tokens = pos_tokens.reshape(
                    -1, orig_size, orig_size, embedding_size
                ).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

                state_dict_model["pos_embed"] = pos_tokens

                print("changing true", state_dict_model["pos_embed"].shape)

        print_results = model.load_state_dict(state_dict_model, strict=False)
        print(print_results)

        torch.cuda.empty_cache()

    return model


@register_model
def vim_large_patch16_224_final_pool_mean_abs_pos_embed_with_noclstok_div2(
    pretrained=False,
    img_size=224,
    patch_size=16,
    stride=16,
    pretrained_checkpoint_path=None,
    **kwargs,
):
    model = VisionMamba(
        img_size=img_size,
        patch_size=patch_size,
        stride=stride,
        embed_dim=1024,
        depth=48,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        **kwargs,
    )
    torch.cuda.empty_cache()
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    if pretrained_checkpoint_path is not None:  # for transfer from MAE/SSL
        checkpoint = torch.load(pretrained_checkpoint_path)["state_dict"]
        state_dict_model = {
            key.replace("backbone.", ""): value for key, value in checkpoint.items()
        }

        if "pos_embed" in state_dict_model:
            print("true", state_dict_model["pos_embed"].shape)
            orig_size = int(math.sqrt(state_dict_model["pos_embed"].shape[1]))
            new_size = int(img_size // patch_size)

            # assuming no class_token in MAE pretraining
            if orig_size != new_size:
                embedding_size = state_dict_model["pos_embed"].shape[-1]
                print(
                    "Position interpolate from %dx%d to %dx%d"
                    % (orig_size, orig_size, new_size, new_size)
                )
                # only the position tokens are interpolated
                pos_tokens = state_dict_model["pos_embed"]
                pos_tokens = pos_tokens.reshape(
                    -1, orig_size, orig_size, embedding_size
                ).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

                state_dict_model["pos_embed"] = pos_tokens

                print("changing true", state_dict_model["pos_embed"].shape)

        print_results = model.load_state_dict(state_dict_model, strict=False)
        print(print_results)

        torch.cuda.empty_cache()

    return model


@register_model
def vim_huge_patch14_224_final_pool_mean_abs_pos_embed_with_noclstok_div2(
    pretrained=False,
    img_size=224,
    patch_size=14,
    stride=14,
    pretrained_checkpoint_path=None,
    **kwargs,
):
    model = VisionMamba(
        img_size=img_size,
        patch_size=patch_size,
        stride=stride,
        embed_dim=1280,
        depth=64,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        **kwargs,
    )
    torch.cuda.empty_cache()
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    if pretrained_checkpoint_path is not None:  # for transfer from MAE/SSL
        checkpoint = torch.load(pretrained_checkpoint_path)["state_dict"]
        state_dict_model = {
            key.replace("backbone.", ""): value for key, value in checkpoint.items()
        }

        if "pos_embed" in state_dict_model:
            print("true", state_dict_model["pos_embed"].shape)
            orig_size = int(math.sqrt(state_dict_model["pos_embed"].shape[1]))
            new_size = int(img_size // patch_size)

            # assuming no class_token in MAE pretraining
            if orig_size != new_size:
                embedding_size = state_dict_model["pos_embed"].shape[-1]
                print(
                    "Position interpolate from %dx%d to %dx%d"
                    % (orig_size, orig_size, new_size, new_size)
                )
                # only the position tokens are interpolated
                pos_tokens = state_dict_model["pos_embed"]
                pos_tokens = pos_tokens.reshape(
                    -1, orig_size, orig_size, embedding_size
                ).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode="bicubic",
                    align_corners=False,
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

                state_dict_model["pos_embed"] = pos_tokens
                print("changing true", state_dict_model["pos_embed"].shape)

                if new_size == 448:
                    trunc_normal_(model.head.weight, std=2e-5)  # taken from MAE

        print_results = model.load_state_dict(state_dict_model, strict=False)
        print(print_results)

        torch.cuda.empty_cache()

    return model
