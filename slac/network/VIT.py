import torch
import torch.nn as nn
import warnings
import math
import torchvision.transforms as T
import time

class VisionTransformer(nn.Module):
    "vision transformer"
    def __init__(self, img_size=[84], patch_size=14, in_chans=3, num_classes=0, embed_dim=384, depth=8,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_feature = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chan=in_chans, embeded_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm1 = norm_layer(embed_dim)
        self.Data_Augmentation = Data_Augmentation()
        # classifier head/change head for other tasks
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.linear_img = nn.Sequential(nn.Linear(embed_dim, embed_dim//2),
                                          nn.ReLU(),
                                          nn.Linear(embed_dim//2, embed_dim//4),
                                          nn.ReLU(),
                                          nn.Linear(embed_dim//4, embed_dim//12))
        self.final_layers = nn.Sequential(nn.Linear(37*32, 37*16),
                                          nn.ReLU(),
                                          nn.Linear(37*16, 256))
        self.norm2 = norm_layer(256)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w: int, h: int):
        npatch = x.shape[2] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        else:
            raise ValueError('Position Encoder does not mactch dimension')

    def prepare_tokens(self, x, Data_Augmentation: bool):
        B, S, nc, w, h = x.shape
        # may be later, see how original solution goes
        if Data_Augmentation:
            x = x.view(B*S, nc, w, h)
            x = self.Data_Augmentation(x)
            x = x.view(B, S, nc, w, h)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, S, -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, Data_Augmentation: bool):
        x = self.prepare_tokens(x, Data_Augmentation)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm1(x)
        x = self.linear_img(x)
        B, S, patches, dim = x.size()
        img_x = x.view(B, S, -1)
        img_x = self.final_layers(img_x)
        img_x = self.norm2(img_x)
        return img_x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, S, N, C = x.shape
        qkv = self.qkv(x).reshape(B*S, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        attn = attn.view(B, S, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)

        # x shape: B, S, Patches, latent_dim
        # attn shape: B, S, Heads, Patches, latent_dim
        return x, attn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_chan=3, embeded_dim=384):
        super().__init__()
        num_patches = int((img_size/patch_size)*(img_size/patch_size))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embeded_dim = embeded_dim
        self.proj = nn.Conv2d(in_chan, embeded_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input shape batch, Sequence, in_Channels H#W
        # Output shape batch, Sequence, correlation & out_Channels

        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        x = self.proj(x).flatten(2).transpose(1, 2).view(B, S, -1, self.embeded_dim)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,)*(x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.MLP = nn.Sequential(nn.Linear(in_features, hidden_features),
                            act_layer(),
                            nn.Dropout(drop),
                            nn.Linear(hidden_features, out_features),
                            nn.Dropout(drop))

    def forward(self, x):
        x = self.MLP(x)
        return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class Data_Augmentation(nn.Module):
    """Create crops of an input image together with additional augmentation.

    It generates 2 global crops and `n_local_crops` local crops.

    Parameters
    ----------
    global_crops_scale : tuple
        Range of sizes for the global crops.

    local_crops_scale : tuple
        Range of sizes for the local crops.

    n_local_crops : int
        Number of local crops to create.

    size : int
        The size of the final image.

    Attributes
    ----------
    global_1, global_2 : transforms.Compose
        Two global transforms.

    local : transforms.Compose
        Local transform. Note that the augmentation is stochastic so one
        instance is enough and will lead to different crops.
    """
    def __init__(
        self,
        global_crops_scale=(0.9, float(1.0)),
        size=84,
    ):
        super().__init__()
        self.transform = nn.Sequential(T.RandomApply(nn.ModuleList([
                T.RandomResizedCrop(size,scale=global_crops_scale,),
                T.RandomAffine(degrees=(-5, 5), translate=(float(0.01), 0.1))]),p=0.3),
                # T.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.2,hue=0.1,)]),p=0.3),
                # T.RandomErasing(p=0.5, scale=(0.02, 0.05), ratio = (0.3, 0.9), value='random'),
                # T.RandomErasing(p=0.5, scale=(0.02, 0.05), ratio=(0.3, 0.9), value='random'),
                # T.RandomErasing(p=0.5, scale=(0.02, 0.05), ratio=(0.3, 0.9), value='random'),
                # T.RandomErasing(p=0.5, scale=(0.02, 0.05), ratio=(0.3, 0.9), value='random'),
        )

    def forward(self, img):
        """
        Apply transformation.
        """
        data_aug = self.transform(img)
        return data_aug

# dimension tests
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# a = torch.ones(32, 8, 3, 84, 84).to(device)
#
# # start = time.time()
# # VIT = VisionTransformer().to(device)
# # load_data_time = time.time()
# # test = VIT(a)
# # computational_time = time.time()
# # print(computational_time - load_data_time)
# # print(load_data_time - start)
# # print(test[0].size())
# # print(test[1].size())
# # VIT = Attention(dim=384)
# # test = PatchEmbed(a)
# # print(test.size())
# # Trans_block = Block(dim=384, num_heads=8)
# # VIT_test = Trans_block(test)
#
# # print(VIT_test[0].size())
# # print(VIT_test[1].size())