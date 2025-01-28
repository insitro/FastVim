# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
import random
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from timm.layers import DropPath, lecun_normal_, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg, _load_weights
from torch import Tensor

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class PatchEmbedPerChannel(nn.Module):
    """
    A PyTorch module for embedding images into a sequence of patches.
    Each patch is embedded into a vector. The module also includes a
    mechanism for hierarchical channel sampling (HCS), which randomly
    selects a subset of channels for each batch during training.

    Attributes
    ----------
    img_size : int
        The size of an individual image. Default is 224.
    patch_size : int
        The size of the image patches. Default is 16.
    num_patches : int
        The total number of patches in an image. It is calculated as
        (img_size // patch_size) * (img_size // patch_size) * in_chans.
    hcs : bool
        If True (default), use hierarchical channel sampling during training.
    proj : nn.Conv3d
        A 3D convolution layer used for projecting the patches into the
        embedding dimension. All channels share the same filter weights.
    channel_embed : nn.Embedding
        An embedding layer for creating channel-specific embeddings.

    Methods
    -------
    forward(x: Tensor, extra_tokens={}) -> tuple[Tensor, int]
        Takes an input image tensor and the extra token dictionary (that contains the
        channel information), applies the patch embedding and hierarchical channel
        sampling (if enabled), and returns the embedded patches and the number of
        channels used.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        stride: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        hcs: bool = False,
        scan_order: str = "Channel-First",  # Spatial-First, Channel-First
        sort_channels=True,  # sorted HCS
    ):
        """
        Initializes the PatchEmbedPerChannel module.

        Parameters
        ----------
        img_size : Optional[int]
            The size of an individual image. Default is 224.
        patch_size : Optional[int]
            The size of the image patches. Default is 16.
        in_chans : Optional[int]
            The number of maximum input channels. Note that during forward, the model
            may take inputs with a subset of all channels. Default is 3.
        embed_dim : Optional[int]
            The dimension of the image embedding. Default is 768.
        hcs : Optional[bool]
            If True (default), use hierarchical channel sampling during training.

        Returns
        -------
        Tensor, int
            The positional encoding for the input patch, and the number of tokens per patch.
        """
        super().__init__()
        self.patch_size = patch_size
        self.hcs = hcs
        self.scan_order = scan_order
        self.sort_channels = sort_channels

        # Here we use 3D convolution because all channels share the same filter weights
        # --> The channel dimension is treated as the 3rd dimension.
        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, stride, stride),
        )

        # We maintain an embedding lookup table to store channel-specific embeddings
        self.channel_embed = nn.Embedding(in_chans, embed_dim)

        # nn.init.trunc_normal_(self.channel_embed.weight, std=0.02)  3 being handled in segm_init_weights

        # patch_grid_size = img_size // patch_size
        patch_grid_size = (
            img_size - patch_size
        ) // stride + 1  # to accomodate for stride like 8 and patch size like 16
        self.grid_size = patch_grid_size

        self.num_patches = patch_grid_size * patch_grid_size

    def forward(
        self, x: Tensor, input_channel_order: Optional[Tensor] = None
    ) -> tuple[Tensor, int]:
        """
        Takes an input image tensor and input channel ids,  optional extra tokens, applies
        the patch embedding and hierarchical channel sampling (if enabled),
        and returns the embedded patches and the number of channels used.
        The extra tokens is required if hcs is enabled.

        Parameters
        ----------
        x : Tensor
            The input image tensor. The shape is [batch, channels, width, height].
            The number of channels must be smaller than or equal
            to the maximum number of channels (in_chans).
        input_channel_order : Optional[Tensor]
            Tensor providing the input channel ordering to fetch the right channel embeddings
            for the input image. The shape is [batch, channels].
            If None, the channel embedding ordering is assumed to be the same as the length
            of the input image in order

        Returns
        -------
        tuple[Tensor, int]
            A tuple containing the embedded images and the number of channels used.
        """
        B, num_channels, h, w = x.shape

        if input_channel_order is None:
            channel_embed = self.channel_embed(
                torch.arange(0, num_channels).repeat(B, 1).to(x.device)
            )  # B, C, embed_dim
        else:
            channel_embed = self.channel_embed(input_channel_order)

        channel_embed = channel_embed.permute(0, 2, 1)  # B embed_dim C

        if self.training and self.hcs:
            # Run HCS for this batch.
            # Step 1: randomly sample the number of channels for this batch
            C_new = random.randint(1, num_channels)

            # Step 2: randomly sample the selected channels
            channels = random.sample(range(num_channels), k=C_new)
            if self.sort_channels is True:
                channels.sort()

            # Update the number of channels, input image, and channel embeddings
            num_channels = C_new

            x = x[:, channels, :, :]
            channel_embed = channel_embed[:, :, channels]

        else:
            channels = random.sample(range(num_channels), k=num_channels)
            channels.sort()

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B embed_dim C H/ps W/ps

        # channel specific offsets
        x += channel_embed.unsqueeze(-1).unsqueeze(-1)  # B embed_dim C H/ps W/ps

        if self.scan_order == "Spatial-First":
            x = x.flatten(2)  # B embed_dim seq_len

        elif self.scan_order == "Channel-First":
            # B embed_dim C H/ps W/ps
            x = x.permute(0, 1, 3, 4, 2)
            x = x.flatten(2)  # B embed_dim seq_len

        # preparing the output sequence
        x = x.transpose(1, 2)  # B seq_len embed_dim

        return x, num_channels, h, w, channels


# x = x.flatten(2) is 'Spatial-First'
# >>> a =torch.tensor([[[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]]]) as follows:
# >>> a.shape
# torch.Size([3, 2, 2])
# >>> a
# tensor([[[ 1,  2],
#          [ 3,  4]],

#         [[ 5,  6],
#          [ 7,  8]],

#         [[ 9, 10],
#          [11, 12]]])
# >>> b = a.flatten()
# >>> b
# tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

# whereas # x = x.permute(0, 1, 3, 4, 2); x = x.flatten(2) is 'Channel-First' as follows:
# >>> c = a.permute(1,2,0)
# >>> c.shape
# torch.Size([2, 2, 3])
# >>> c
# tensor([[[ 1,  5,  9],
#          [ 2,  6, 10]],

#         [[ 3,  7, 11],
#          [ 4,  8, 12]]])
# >>> c.flatten()
# tensor([ 1,  5,  9,  2,  6, 10,  3,  7, 11,  4,  8, 12])


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
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
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
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
    use_norm_after_ssm=False,
    init_layer_scale=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        Mamba,
        layer_idx=layer_idx,
        use_norm_after_ssm=use_norm_after_ssm,
        init_layer_scale=init_layer_scale,
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
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        final_pool_type="none",
        if_abs_pos_embed=False,
        if_cls_token=False,
        init_layer_scale=None,
        scan_order="Channel-First",  # 'Channel-First', 'Spatial-First'
        hcs=True,
        sort_channels=True,  # sorted HCS
        use_norm_after_ssm=True,
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
        self.if_cls_token = if_cls_token

        self.channels = channels

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.scan_order = scan_order
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedPerChannel(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim,
            hcs=hcs,
            scan_order=scan_order,
            sort_channels=sort_channels,
        )
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_cls_token:
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
                    use_norm_after_ssm=use_norm_after_ssm,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)

        if if_cls_token:
            trunc_normal_(self.cls_token, std=0.02)

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
        return {
            "pos_embed",
            "pos_embed_obj",
            "cls_token",
            "dist_token",
            "cls_token_head",
            "cls_token_tail",
        }

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x, tokens_per_patch, h, w, channels_list = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_abs_pos_embed:
            if self.scan_order == "Spatial-First":
                x = x + self.pos_embed.expand(
                    tokens_per_patch, -1, self.embed_dim
                ).reshape(1, -1, self.embed_dim)

            elif self.scan_order == "Channel-First":
                x = x + torch.repeat_interleave(self.pos_embed, tokens_per_patch, 1)  #

        if self.if_abs_pos_embed:
            x = self.pos_drop(x)

        if self.if_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            token_position = (
                M // 2
            )  # in channel mamba it will be middle channel middle token
            # add cls token in the middle
            x = torch.cat(
                (x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1
            )

        # mamba impl
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
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

        # return only cls token if it exists
        if self.if_cls_token:
            return hidden_states[:, token_position, :]
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None):
        x = self.forward_features(x, inference_params)
        if return_features:
            return x
        x = self.head(x)
        return x


@register_model
def channelvim_small_patch16_224_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False, patch_size=16, stride=16, if_abs_pos_embed=True, **kwargs
):
    model = VisionMamba(
        patch_size=patch_size,
        stride=stride,
        if_abs_pos_embed=if_abs_pos_embed,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_cls_token=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
