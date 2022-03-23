from typing import Dict, List
from math import sqrt, log
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.maskformer.transformer_decoder import TransformerDecoderLayer, TransformerDecoder
from utils.misc import get_model


class MaskFormer(nn.Module):
    def __init__(
            self,
            n_queries: int = 100,
            arch: str = "vit_small",
            patch_size: int = 8,
            training_method: str = "dino",
            n_decoder_layers: int = 6,
            normalize_before: bool = False,
            return_intermediate: bool = False,
            learnable_pixel_decoder: bool = False,
            lateral_connection: bool = False,
            scale_factor: int = 2,
            abs_2d_pe_init: bool = False,
            use_binary_classifier: bool = False
    ):
        """Define a encoder and decoder along with queries to be learned through the decoder."""
        super(MaskFormer, self).__init__()

        if arch == "vit_small":
            self.encoder = get_model(arch=arch, patch_size=patch_size, training_method=training_method)
            n_dims: int = self.encoder.n_embs
            n_heads: int = self.encoder.n_heads
            mlp_ratio: int = self.encoder.mlp_ratio
        else:
            self.encoder = get_model(arch=arch, training_method=training_method)
            n_dims_resnet: int = self.encoder.n_embs
            n_dims: int = 384
            n_heads: int = 6
            mlp_ratio: int = 4
            self.linear_layer = nn.Conv2d(n_dims_resnet, n_dims, kernel_size=1)

        decoder_layer = TransformerDecoderLayer(
            n_dims, n_heads, n_dims * mlp_ratio, 0., activation="relu", normalize_before=normalize_before
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            n_decoder_layers,
            norm=nn.LayerNorm(n_dims),
            return_intermediate=return_intermediate
        )

        self.query_embed = nn.Embedding(n_queries, n_dims).weight  # initialized with gaussian(0, 1)

        if use_binary_classifier:
            # self.ffn = MLP(n_dims, n_dims, n_dims, num_layers=3)
            # self.linear_classifier = nn.Linear(n_dims, 1)
            self.ffn = MLP(n_dims, n_dims, 1, num_layers=3)
            # self.norm = nn.LayerNorm(n_dims)
        else:
            # self.ffn = None
            # self.linear_classifier = None
            # self.norm = None
            self.ffn = MLP(n_dims, n_dims, n_dims, num_layers=3)
            self.linear_classifier = nn.Linear(n_dims, 2)
            self.norm = nn.LayerNorm(n_dims)

        self.arch = arch
        self.use_binary_classifier = use_binary_classifier
        self.lateral_connection = lateral_connection
        self.learnable_pixel_decoder = learnable_pixel_decoder
        self.scale_factor = scale_factor

    # copy-pasted from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    @staticmethod
    def positional_encoding_2d(n_dims: int, height: int, width: int):
        """
        :param n_dims: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if n_dims % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(n_dims))
        pe = torch.zeros(n_dims, height, width)
        # Each dimension use half of d_model
        d_model = int(n_dims / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe

    def forward_encoder(self, x: torch.Tensor):
        """
        :param x: b x c x h x w
        :return patch_tokens: b x depth x hw x n_dims
        """
        if self.arch == "vit_small":
            encoder_outputs: Dict[str, torch.Tensor] = self.encoder(x)  # [:, 1:, :]
            all_patch_tokens: List[torch.Tensor] = list()
            for layer_name in [f"layer{num_layer}" for num_layer in range(1, self.encoder.depth + 1)]:
                patch_tokens: torch.Tensor = encoder_outputs[layer_name][:, 1:, :]  # b x hw x n_dims
                all_patch_tokens.append(patch_tokens)

            all_patch_tokens: torch.Tensor = torch.stack(all_patch_tokens, dim=0)  # depth x b x hw x n_dims
            all_patch_tokens = all_patch_tokens.permute(1, 0, 3, 2)  # b x depth x n_dims x hw
            return all_patch_tokens
        else:
            encoder_outputs = self.linear_layer(self.encoder(x)[-1])  # b x n_dims x h x w
            return encoder_outputs

    def forward_transformer_decoder(self, patch_tokens: torch.Tensor, skip_decoder: bool = False) -> torch.Tensor:
        """Forward transformer decoder given patch tokens from the encoder's last layer.
        :param patch_tokens: b x n_dims x hw -> hw x b x n_dims
        :param skip_decoder: if True, skip the decoder and produce mask predictions directly by matrix multiplication
        between learnable queries and encoder features (i.e., patch tokens). This is for the purpose of an overfitting
        experiment.
        :return queries: n_queries x b x n_dims -> b x n_queries x n_dims or b x n_layers x n_queries x n_dims
        """
        b = patch_tokens.shape[0]
        patch_tokens = patch_tokens.permute(2, 0, 1)  # b x n_dims x hw -> hw x b x n_dims

        # n_queries x n_dims -> n_queries x b x n_dims
        queries: torch.Tensor = self.query_embed.unsqueeze(1).repeat(1, b, 1)
        queries: torch.Tensor = self.decoder.forward(
            tgt=torch.zeros_like(queries),
            memory=patch_tokens,
            query_pos=queries
        ).squeeze(dim=0)

        if len(queries.shape) == 3:
            queries: torch.Tensor = queries.permute(1, 0, 2)  # n_queries x b x n_dims -> b x n_queries x n_dims
        elif len(queries.shape) == 4:
            # n_layers x n_queries x b x n_dims -> b x n_layers x n_queries x n_dims
            queries: torch.Tensor = queries.permute(2, 0, 1, 3)
        return queries

    def forward_pixel_decoder(self, patch_tokens: torch.Tensor, input_size=None):
        """ Upsample patch tokens by self.scale_factor and produce mask predictions
        :param patch_tokens: b (x depth) x n_dims x hw -> b (x depth) x n_dims x h x w
        :param queries: b x n_queries x n_dims
        :return mask_predictions: b x n_queries x h x w
        """

        if input_size is None:
            # assume square shape features
            hw = patch_tokens.shape[-1]
            h = w = int(sqrt(hw))
        else:
            # arbitrary shape features
            h, w = input_size
        patch_tokens = patch_tokens.view(*patch_tokens.shape[:-1], h, w)

        assert len(patch_tokens.shape) == 4
        patch_tokens = F.interpolate(patch_tokens, scale_factor=self.scale_factor, mode="bilinear")
        return patch_tokens

    def forward(self, x, encoder_only=False, skip_decoder: bool = False):
        """
        x: b x c x h x w
        patch_tokens: b x n_patches x n_dims -> n_patches x b x n_dims
        query_emb: n_queries x n_dims -> n_queries x b x n_dims
        """
        dict_outputs: dict = dict()

        # b x depth x n_dims x hw (vit) or b x n_dims x h x w (resnet50)
        features: torch.Tensor = self.forward_encoder(x)

        if self.arch == "vit_small":
            # extract the last layer for decoder input
            last_layer_features: torch.Tensor = features[:, -1, ...]  # b x n_dims x hw
        else:
            # transform the shape of the features to the one compatible with transformer decoder
            b, n_dims, h, w = features.shape
            last_layer_features: torch.Tensor = features.view(b, n_dims, h * w)  # b x n_dims x hw

        if encoder_only:
            _h, _w = self.encoder.make_input_divisible(x).shape[-2:]
            _h, _w = _h // self.encoder.patch_size, _w // self.encoder.patch_size

            b, n_dims, hw = last_layer_features.shape
            dict_outputs.update({"patch_tokens": last_layer_features.view(b, _h, _w, n_dims)})
            return dict_outputs

        # transformer decoder forward
        queries: torch.Tensor = self.forward_transformer_decoder(
            last_layer_features,
            skip_decoder=skip_decoder
        )  # b x n_queries x n_dims or b x n_layers x n_queries x n_dims

        # pixel decoder forward (upsampling the patch tokens by self.scale_factor)
        if self.arch == "vit_small":
            _h, _w = self.encoder.make_input_divisible(x).shape[-2:]
            _h, _w = _h // self.encoder.patch_size, _w // self.encoder.patch_size
        else:
            _h, _w = h, w
        features: torch.Tensor = self.forward_pixel_decoder(
            patch_tokens=features if self.lateral_connection else last_layer_features,
            input_size=(_h, _w)
        )  # b x n_dims x h x w

        # queries: b x n_queries x n_dims or b x n_layers x n_queries x n_dims
        # features: b x n_dims x h x w
        # mask_pred: b x n_queries x h x w or b x n_layers x n_queries x h x w
        if len(queries.shape) == 3:
            mask_pred = torch.einsum("bqn,bnhw->bqhw", queries, features)
        else:
            if self.use_binary_classifier:
                mask_pred = torch.sigmoid(torch.einsum("bdqn,bnhw->bdqhw", queries, features))
            else:
                mask_pred = torch.sigmoid(torch.einsum("bdqn,bnhw->bdqhw", self.ffn(queries), features))

        if self.use_binary_classifier:
            # queries: b x n_layers x n_queries x n_dims -> n_layers x b x n_queries x n_dims
            queries = queries.permute(1, 0, 2, 3)
            objectness: List[torch.Tensor] = list()
            for n_layer, queries_per_layer in enumerate(queries):  # queries_per_layer: b x n_queries x n_dims
                # objectness_per_layer = self.linear_classifier(
                #     self.ffn(self.norm(queries_per_layer))
                # )  # b x n_queries x 1
                objectness_per_layer = self.ffn(queries_per_layer)  # b x n_queries x 1
                objectness.append(objectness_per_layer)
            # n_layers x b x n_queries x 1 -> # b x n_layers x n_queries x 1
            objectness: torch.Tensor = torch.stack(objectness).permute(1, 0, 2, 3)
            dict_outputs.update({
                "objectness": torch.sigmoid(objectness),
                "mask_pred": mask_pred
            })

        return dict_outputs


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, n_groups=32, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.ReLU()
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(self.block(x), scale_factor=self.scale_factor, mode="bilinear")