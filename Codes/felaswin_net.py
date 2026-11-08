import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# 1. Novel Modules (FETLE & AMSFA)
# ══════════════════════════════════════════════════════════════════════════════
class FETLE(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.texture_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.edge_dw    = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False)
        self.hf_smooth  = nn.AvgPool2d(3, stride=1, padding=1)
        self.hf_scale   = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.branch_w   = nn.Parameter(torch.ones(3))
        self.bn         = nn.BatchNorm2d(channels)
        self.gelu       = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.texture_dw(x)
        e = self.edge_dw(x)
        h = self.hf_scale(x - self.hf_smooth(x)) 
        w = F.softmax(self.branch_w, dim=0)
        return self.gelu(self.bn(w[0] * t + w[1] * e + w[2] * h))

class AMSFA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, asym_k: int = 5):
        super().__init__()
        p = asym_k // 2
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.path3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.path4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, (1, asym_k), padding=(0, p), groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, (asym_k, 1), padding=(p, 0), groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.path_w = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.path_w, dim=0)
        return F.gelu(w[0]*self.path1(x) + w[1]*self.path2(x) + w[2]*self.path3(x) + w[3]*self.path4(x))

# ══════════════════════════════════════════════════════════════════════════════
# 2. Residual FETLE Connection Block (Stable ResNet Pattern)
# ══════════════════════════════════════════════════════════════════════════════
class ResNetFetleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        
        self.resnet_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.fetle_path = FETLE(channels)
        
        # 2 weights: only for learned branches
        self.branch_w = nn.Parameter(torch.ones(2))
        self.final_activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.branch_w, dim=0)
        out = x + (w[0] * self.resnet_path(x)) + (w[1] * self.fetle_path(x))
        return self.final_activation(out)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Native Swin Components with DropPath & Attention Dropout
# ══════════════════════════════════════════════════════════════════════════════
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0.1, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinBlockNative(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, input_resolution=(28, 28), 
                 dropout_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.0, init_values=1e-4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, attn_drop=attn_drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout_rate)
        )

        # Learnable LayerScale parameters for self-stabilization
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        
        # LayerScale applied directly to residual feature flows
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

# ══════════════════════════════════════════════════════════════════════════════
# 4. Modified Architecture: FeLASwin-Net
# ══════════════════════════════════════════════════════════════════════════════
class FeLASwinNet(nn.Module):
    def __init__(self, num_classes=3, drop_path_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        # Stem: 224x224 -> 56x56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1 (Residual FETLE): 56x56 -> 28x28 via AMSFA
        self.stage1_fetle = ResNetFetleBlock(48)
        self.stage1_down = AMSFA(48, 96)
        
        # Stage 2: 28x28 -> 14x14 via AMSFA
        self.stage2_fetle = ResNetFetleBlock(96)
        self.stage2_down = AMSFA(96, 128)
        
        # Stochastic depth decay rule across all 6 Swin blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 6)]

        # Stage 3 (Swin Bottleneck 1): 14x14 grid
        self.stage3_swin = nn.Sequential(
            SwinBlockNative(128, num_heads=4, window_size=7, shift_size=0, input_resolution=(14, 14), 
                            attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[0]),
            SwinBlockNative(128, num_heads=4, window_size=7, shift_size=3, input_resolution=(14, 14), 
                            attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[1]),
            SwinBlockNative(128, num_heads=4, window_size=7, shift_size=0, input_resolution=(14, 14), 
                            attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[2]),
            SwinBlockNative(128, num_heads=4, window_size=7, shift_size=3, input_resolution=(14, 14), 
                            attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[3])
        )
        
        # Stage 4 (Swin Bottleneck 2): 14x14 -> 7x7 via AMSFA
        self.stage4_down = AMSFA(128, 256)
        
        # Grid is 7x7, window is 7, shift_size is 0
        self.stage4_swin = nn.Sequential(
            SwinBlockNative(256, num_heads=8, window_size=7, shift_size=0, input_resolution=(7, 7), 
                            attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[4]),
            SwinBlockNative(256, num_heads=8, window_size=7, shift_size=0, input_resolution=(7, 7), 
                            attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[5])
        )
        
        # Classification Head
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. CNN Stages
        x = self.stem(x)
        x = self.stage1_down(self.stage1_fetle(x))
        x = self.stage2_down(self.stage2_fetle(x))
        
        # 2. Stage 3 Swin: (B, C, H, W) -> (B, L, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage3_swin(x)
        
        # 3. Stage 4 Downsampling: (B, L, C) -> (B, C, H, W)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.stage4_down(x)
        
        # 4. Stage 4 Swin
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.stage4_swin(x)
        
        # 5. Global Avg Pooling & Classification
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

if __name__ == "__main__":
    model = FeLASwinNet(num_classes=3, drop_path_rate=0.1, attn_drop_rate=0.1)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params/1e6:.2f} M")