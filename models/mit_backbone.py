import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x.transpose(1, 2).view(x.shape[0], -1, H, W)).flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class MiTBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MiTBackbone, self).__init__()
        self.mit = MixVisionTransformer(
            img_size=256,
            embed_dims=[32, 64, 160, 256], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[2, 2, 2, 2], 
            sr_ratios=[8, 4, 2, 1]
        )
        
        if pretrained:
            self._load_pretrained()
            
    def _load_pretrained(self):
        try:
            local_path = 'mit_b0.pth'
            if os.path.exists(local_path):
                print(f"Loading pretrained weights from local file: {local_path}...")
                state_dict = torch.load(local_path, map_location='cpu')
            else:
                url = "https://huggingface.co/nvidia/mit-b0/resolve/main/pytorch_model.bin"
                print(f"Downloading/Loading pretrained weights from {url}...")
                state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=False)
            
            new_state_dict = {}
            kv_buffer = {} 
            
            for k, v in state_dict.items():
                if k.startswith('segformer.encoder.'):
                    k = k[18:]
                elif k.startswith('encoder.'):
                    k = k[8:]
                
                if k.startswith('patch_embeddings.'):
                    parts = k.split('.')
                    idx = int(parts[1])
                    if len(parts) > 3 and parts[2] == 'layer_norm':
                        new_k = f"patch_embed{idx+1}.norm.{parts[3]}"
                    else:
                        new_k = f"patch_embed{idx+1}.{'.'.join(parts[2:])}"
                    new_state_dict[new_k] = v
                    
                elif k.startswith('layer_norm.'):
                    parts = k.split('.')
                    idx = int(parts[1])
                    new_k = f"norm{idx+1}.{'.'.join(parts[2:])}"
                    new_state_dict[new_k] = v
                    
                elif k.startswith('block.'):
                    parts = k.split('.')
                    stage = int(parts[1])
                    blk_id = int(parts[2])
                    rest = parts[3:]
                    prefix = f"block{stage+1}.{blk_id}"
                    
                    if rest[0] == 'attention':
                        if rest[1] == 'self':
                            if rest[2] == 'query':
                                new_k = f"{prefix}.attn.q.{rest[3]}"
                                new_state_dict[new_k] = v
                            elif rest[2] in ['key', 'value']:
                                param_type = rest[3]
                                buffer_key = f"{prefix}.attn.kv.{param_type}"
                                if buffer_key not in kv_buffer:
                                    kv_buffer[buffer_key] = {}
                                kv_buffer[buffer_key][rest[2]] = v
                            elif rest[2] == 'sr':
                                new_k = f"{prefix}.attn.sr.{rest[3]}"
                                new_state_dict[new_k] = v
                            elif rest[2] == 'layer_norm':
                                new_k = f"{prefix}.attn.norm.{rest[3]}"
                                new_state_dict[new_k] = v
                        elif rest[1] == 'output' and rest[2] == 'dense':
                            new_k = f"{prefix}.attn.proj.{rest[3]}"
                            new_state_dict[new_k] = v
                            
                    elif rest[0] == 'mlp':
                        if rest[1] in ['linear1', 'dense1']:
                            new_k = f"{prefix}.mlp.fc1.{rest[2]}"
                        elif rest[1] in ['linear2', 'dense2']:
                            new_k = f"{prefix}.mlp.fc2.{rest[2]}"
                        elif rest[1] == 'dwconv':
                            if rest[2] == 'dwconv':
                                new_k = f"{prefix}.mlp.dwconv.{rest[3]}"
                            else:
                                new_k = f"{prefix}.mlp.dwconv.{rest[2]}"
                        new_state_dict[new_k] = v
                        
                    elif rest[0] == 'layer_norm_1':
                        new_k = f"{prefix}.norm1.{rest[1]}"
                        new_state_dict[new_k] = v
                    elif rest[0] == 'layer_norm_2':
                        new_k = f"{prefix}.norm2.{rest[1]}"
                        new_state_dict[new_k] = v

            for k, val_dict in kv_buffer.items():
                if 'key' in val_dict and 'value' in val_dict:
                    merged = torch.cat([val_dict['key'], val_dict['value']], dim=0)
                    new_state_dict[k] = merged
            
            model_state_dict = self.mit.state_dict()
            filtered_state_dict = {}
            for k, v in new_state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
            
            msg = self.mit.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded HF pretrained weights with msg: {msg}")
            
        except Exception as e:
            print(f"Warning: Failed to load HF pretrained weights: {e}")
            print("Training from scratch...")
        
    def forward(self, x):
        outs = self.mit(x)
        return outs[0], outs[1], outs[2]