# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple_faster import Mamba
from mamba_ssm.modules.mamba_simple_masked_faster_prune import Mamba_masked
from timm.layers import lecun_normal_, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg, _load_weights
from torch import Tensor

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import torch.nn.functional as F
from mmdet.registry import MODELS


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
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
        self.stride = to_2tuple(stride)

        if scanpath_type == "colwise":  # colwise denotes Pool_row in FastVim paper
            self.grid_size = tuple(
                [
                    (self.img_size[1] - self.patch_size[1])//self.stride[1] + 1,
                    (self.img_size[0] - self.patch_size[0])//self.stride[0] + 1,
                ]
            )
        elif scanpath_type == "rowwise":  # rowwise denotes Pool_col in FastVim paper
            self.grid_size = tuple(
                [
                    (self.img_size[0] - self.patch_size[0])//self.stride[0] + 1,
                    (self.img_size[1] - self.patch_size[1])//self.stride[1] + 1,
                ]
            )

        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.scanpath_type = scanpath_type

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        assert (
            H == self.img_size[0]
        ), f"Input height ({H}) doesn't match model ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input width ({W}) doesn't match model ({self.img_size[1]})."

        x = self.proj(x)

        if self.scanpath_type == "colwise":  # colwise denotes Pool_row in FastVim paper
            x = x.transpose(2, 3)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x




class Block_masked(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        rotate_every_block=True,
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
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.rotate_indices = self.compute_rotate_indices(*token_size)

    @staticmethod
    def compute_rotate_indices(H, W):
        indices = torch.arange(H * W, dtype=torch.long)
        i, j = indices.div(W, rounding_mode="floor"), indices % W
        return j * H + i

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        ids_keep=None,
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
                residual = residual + hidden_states

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
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        if self.rotate_every_block is True and self.layer_idx % 2 != 0:
            rotate_indices = self.rotate_indices.to(hidden_states.device)

            batch_indices = (
                torch.arange(hidden_states.shape[0])
                .unsqueeze(-1)
                .to(hidden_states.device)
            )

            ids_keep = rotate_indices[ids_keep].contiguous()
            rotated_ids = torch.argsort(
                ids_keep, dim=1
            )  # .contiguous()   agrsort returns contiguous output
            inv_rotated_ids = torch.argsort(rotated_ids, 1)  # .contiguous()
            ids_keep = ids_keep[batch_indices, rotated_ids].contiguous()
            hidden_states = hidden_states[batch_indices, rotated_ids].contiguous()

        hidden_states, scores = self.mixer(
            hidden_states, ids_keep, inference_params=inference_params
        )

        if self.rotate_every_block is True and self.layer_idx % 2 != 0:
            hidden_states = hidden_states[batch_indices, inv_rotated_ids].contiguous()

        return hidden_states, residual, scores

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )




def create_block_masked(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
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
    collapse_method="mean",
    token_size=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    if rotate_every_block is True and layer_idx % 2 != 0:
        mixer_cls = partial(
            Mamba_masked,
            layer_idx=layer_idx,
            init_layer_scale=init_layer_scale,
            scanpath_type=scanpath_type,
            use_norm_after_ssm=use_norm_after_ssm,
            token_size=[token_size[1], token_size[0]],
            collapse_method=collapse_method,
            **ssm_cfg,
            **factory_kwargs,
        )
    else:
        mixer_cls = partial(
            Mamba_masked,
            layer_idx=layer_idx,
            init_layer_scale=init_layer_scale,
            scanpath_type=scanpath_type,
            use_norm_after_ssm=use_norm_after_ssm,
            token_size=token_size,
            collapse_method=collapse_method,
            **ssm_cfg,
            **factory_kwargs,
        )

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block_masked(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
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
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        if_abs_pos_embed=True,
        init_layer_scale=None,
        embed_layer=PatchEmbed,
        scanpath_type="rowwise",  # rowwise denotes Pool_col in FastVim paper
        use_norm_after_ssm=True,
        rotate_every_block=True,  # for acorss col and row alternate pooling
        collapse_method="mean",
        token_keep_ratio=0.8,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_abs_pos_embed = if_abs_pos_embed
        self.rotate_every_block = rotate_every_block
        self.token_keep_ratio = token_keep_ratio

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.patch_size = patch_size
        self.stride = stride
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim,
            strict_img_size=False,
            dynamic_img_pad=True,
            scanpath_type=scanpath_type,
        )

        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.token_size = self.patch_embed.grid_size
        print('self.token_size:', self.token_size )

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )


        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block_masked(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    init_layer_scale=init_layer_scale,
                    scanpath_type=scanpath_type,
                    use_norm_after_ssm=use_norm_after_ssm,
                    rotate_every_block=rotate_every_block,
                    collapse_method=collapse_method,
                    token_size=self.token_size,
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

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        x = self.patch_embed(x)
        batch, N, D = x.shape

        if self.if_abs_pos_embed:
            pos_embed = self.pos_embed

            x = x + pos_embed
            x = self.pos_drop(x)

        ids_keep = torch.arange(N, device=x.device).unsqueeze(0).repeat(batch, 1)

        # mamba impl
        residual = None
        hidden_states = x
        layernum = 0

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual, scores = layer(
                hidden_states, residual, ids_keep, inference_params=inference_params
            )
            layernum += 1

            if (layernum - 5) >= 0 and (layernum - 5) % 5 == 0 and layernum <= 20:
                _, N, D = hidden_states.shape
                # print('start', layernum, N, hidden_states.shape, residual.shape, ids_keep.shape)

                num_keep_node = math.ceil(N * self.token_keep_ratio)     # 196 r

                _, top_score_indices = scores.topk(num_keep_node, dim=1, largest=True)
                top_score_indices, _ = torch.sort(top_score_indices, dim=-1)

                hidden_states = torch.gather(hidden_states, dim=1, index=top_score_indices.unsqueeze(-1).repeat(1, 1, D))
                residual = torch.gather(residual, dim=1, index=top_score_indices.unsqueeze(-1).repeat(1, 1, D))
                ids_keep = torch.gather(ids_keep, 1, top_score_indices)
                # print('end', layernum, N, hidden_states.shape, residual.shape, ids_keep.shape)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states.mean(dim=1)

    def forward(self, x, return_features=False, inference_params=None):
        x = self.forward_features(x, inference_params)
        if return_features:
            return x
        x = self.head(x)

        return x



######################################################################
@register_model
def vim_small_patch16_224_final_pool_mean_abs_pos_embed_with_noclstok_div2(
    pretrained=False,
    img_size=224,
    patch_size=16,
    stride=16,
    pretrained_checkpoint_path=None,
    load_ema=False,
    **kwargs,
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
        if_abs_pos_embed=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])


    if pretrained_checkpoint_path is not None:  # for transfer from MAE/SSL
        checkpoint = torch.load(pretrained_checkpoint_path)["state_dict"]

        if load_ema is False:
            # state_dict_model = {
            #     key.replace("backbone.", ""): value for key, value in checkpoint.items()
            # }

            state_dict_model = {}
            for key, value in checkpoint.items():
                # Remove keys without 'ema'
                if 'backbone' in key:
                    # Remove 'backbone.' from keys
                    state_dict_model[key.replace('backbone.', '')] = value

        else:
            state_dict_model = {}
            for key, value in checkpoint.items():
                # Remove keys without 'ema'
                if 'ema' in key:
                    # Remove 'backbone.' from keys
                    state_dict_model[key.replace('ema.module.', '')] = value


        if "pos_embed" in state_dict_model:
            print("true", state_dict_model["pos_embed"].shape)
            orig_size = int(math.sqrt(state_dict_model["pos_embed"].shape[1]))
            new_size = (img_size - patch_size) // stride + 1

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

