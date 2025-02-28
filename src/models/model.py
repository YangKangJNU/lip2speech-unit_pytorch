from torch import nn
import torch
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet
from espnet.nets.pytorch_backend.transformer.embedding import (
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from src.models.attention import RelPositionMultiHeadedAttention
from src.utils import lens_to_mask

class L2SUnit(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.relu_type = cfg.relu_type
        self.num_layer = cfg.num_layer
        self.attention_dim = cfg.attention_dim
        self.attention_head = cfg.attention_head
        self.kernel_size = cfg.kernel_size
        self.feedforward_dim = cfg.feedforward_dim
        self.dropout_rate = cfg.dropout_rate
        self.unit_num = cfg.unit_num
        self.spk_dim = cfg.spk_dim

        # resnet
        self.resnet = Conv3dResNet(relu_type=self.relu_type)
        # conformer
        encoder_attn_layer = RelPositionMultiHeadedAttention
        encoder_attn_layer_args = (self.attention_head, self.attention_dim, self.dropout_rate)
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (self.attention_dim, self.feedforward_dim, self.dropout_rate)
        convolution_layer = ConvolutionModule
        convolution_layer_args = (self.attention_dim, self.kernel_size)
        self.conformer = repeat(
            self.num_layer,
            lambda: EncoderLayer(
                self.attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args),
                self.dropout_rate,
                normalize_before=True,
                concat_after=True,
                macaron_style=True,                
            ),
        )
        pos_enc_class = RelPositionalEncoding
        self.embed = nn.Sequential(
            nn.Linear(self.attention_dim, self.attention_dim),
            pos_enc_class(self.attention_dim, self.dropout_rate),
        )        
        # mel
        self.mel_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.attention_dim+self.spk_dim,out_channels=self.attention_dim,kernel_size=3,stride=1,padding=1),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Conv1d(in_channels=self.attention_dim,out_channels=self.attention_dim,kernel_size=3,stride=1,padding=1),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Conv1d(in_channels=self.attention_dim,out_channels=self.attention_dim,kernel_size=3,stride=1,padding=1),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
        )
        self.mel_proj = Linear(self.attention_dim, 160)

        # unit
        self.proj_out = MLP(self.attention_dim, [self.attention_dim, self.attention_dim, self.unit_num], self.dropout_rate, nn.GELU)

        self.final_dropout = nn.Dropout(self.dropout_rate)

    def forward(self, video, video_len, spk_emb):
        T_video = video.size(2)
        x = self.resnet(video)
        x = x.repeat_interleave(2, dim=1)
        x = self.embed(x)
        x_mask = lens_to_mask(video_len*2, T_video*2).unsqueeze(1)
        x, x_mask = self.conformer(x, x_mask)
        spk_x = torch.cat([spk_emb.unsqueeze(1).repeat(1, T_video*2, 1), x[0]], dim=-1)
        encoder_out_mel = self.mel_proj(self.mel_conv(spk_x.transpose(1,2)).transpose(1,2))
        B, T, D = encoder_out_mel.shape
        encoder_out_mel = encoder_out_mel.reshape(B, T, D//2, 2).transpose(-1,-2).reshape(B, T*2, D//2)
        x = self.final_dropout(x[0])
        unit_logit = self.proj_out(x)
        mel_mask = lens_to_mask(video_len*4, T_video*4).unsqueeze(-1)
        return encoder_out_mel*mel_mask, unit_logit

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims,
        dropout: float = 0.1,
        nonlinearity = nn.ReLU,
        normalization = None, #nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super(MLP, self).__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x):
        x = self.projection(x)
        x = self.last_layer(x)
        return x

class Conformer(nn.Module):
    def __init__(self,
                input_dim=1024,
                num_layer=6,
                attention_dim=256,
                attention_head=4,
                kernel_size=31,
                feedforward_dim=2048,
                dropout_rate=0.1,
                ):
        super().__init__()

        pos_enc_class = RelPositionalEncoding
        encoder_attn_layer = RelPositionMultiHeadedAttention
        encoder_attn_layer_args = (attention_head, attention_dim, dropout_rate)
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, feedforward_dim, dropout_rate)
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, kernel_size)
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(input_dim, attention_dim),
            pos_enc_class(attention_dim, dropout_rate),
        )
        self.encoder = repeat(
            num_layer,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args),
                dropout_rate,
                normalize_before=True,
                concat_after=True,
                macaron_style=True,                
            ),
        )

    def forward(self, x, x_len):
        x = self.embed(x)
        x_mask = lens_to_mask(x_len).unsqueeze(1)
        x, x_mask = self.encoder(x, x_mask)

        return x[0]*(x_mask.permute(0, 2, 1)), x_mask.squeeze()

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m