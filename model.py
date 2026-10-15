"""CIUM-Net: HVI dual-branch restoration, BiCIU and LCUM (paper Sections 3.1-3.2)."""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class NormDownsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x

class NormUpsample(nn.Module):
    def __init__(self, in_ch,out_ch,scale=2,use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm=use_norm
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch*2,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
            
    def forward(self, x,y):
        x = self.up_scale(x)
        x = torch.cat([x, y],dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x


# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = out.reshape(b, c, h, w)

        out = self.project_out(out)
        return out
    

# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


pi = 3.141592653589793

class HVITransform(nn.Module):
    def __init__(self):
        super(HVITransform, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # k is reciprocal to the paper mentioned
        
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        k = self.density_k
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I],dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]
        
        # clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
        
        v = I
        k = self.density_k.detach()  # Preserve the supplied PHVIT gradient convention.
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V + eps,H + eps) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)
        
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb


class GatedLocalMixing(nn.Module):
    """Bidirectional gated local mixing, paper Eq. (2).

    Given a target branch x and a guidance branch y, the block performs:
    concat(x, y) -> 1x1 projection -> channel attention -> spatial attention
    -> pair-aware gating.
    """

    def __init__(self, dim, reduce_ratio=4):
        super(GatedLocalMixing, self).__init__()
        hidden = max(dim // reduce_ratio, 4)

        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Sigmoid(),
        )

        self.pair_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        pair = torch.cat([x, y], dim=1)
        fused = self.fuse(pair)

        channel_weight = self.channel_attn(fused)
        avg_map = torch.mean(fused, dim=1, keepdim=True)
        max_map, _ = torch.max(fused, dim=1, keepdim=True)
        spatial_weight = self.spatial_attn(torch.cat([avg_map, max_map], dim=1))
        pair_weight = self.pair_gate(pair)

        return fused * channel_weight * spatial_weight * pair_weight


class BiCIU(nn.Module):
    """Bidirectional Chromaticity-Intensity Update Unit, paper Eq. (3).

    Input order is fixed as:
        hv_feat, i_feat -> hv_out, i_out
    """

    def __init__(self, dim, num_heads, bias=False, reduce_ratio=4):
        super(BiCIU, self).__init__()

        # Local bidirectional fusion.
        self.norm_hv_local = LayerNorm(dim)
        self.norm_i_local = LayerNorm(dim)
        self.mafm_hv = GatedLocalMixing(dim, reduce_ratio=reduce_ratio)
        self.mafm_i = GatedLocalMixing(dim, reduce_ratio=reduce_ratio)

        # Cross-attention interaction, reusing the repository's CAB implementation.
        self.norm_hv_attn = LayerNorm(dim)
        self.norm_i_attn = LayerNorm(dim)
        self.hv_cross_attn = CAB(dim, num_heads, bias=bias)
        self.i_cross_attn = CAB(dim, num_heads, bias=bias)

        # Branch-wise refinement, reusing the repository's IEL implementation.
        self.norm_hv_iel = LayerNorm(dim)
        self.norm_i_iel = LayerNorm(dim)
        self.hv_iel = IEL(dim)
        self.i_iel = IEL(dim)

    def forward(self, hv_feat, i_feat):
        hv_local = self.norm_hv_local(hv_feat)
        i_local = self.norm_i_local(i_feat)
        hv_feat = hv_feat + self.mafm_hv(hv_local, i_local)
        i_feat = i_feat + self.mafm_i(i_local, hv_local)

        hv_attn_in = self.norm_hv_attn(hv_feat)
        i_attn_in = self.norm_i_attn(i_feat)
        hv_feat = hv_feat + self.hv_cross_attn(hv_attn_in, i_attn_in)
        i_feat = i_feat + self.i_cross_attn(i_attn_in, hv_attn_in)

        hv_feat = hv_feat + self.hv_iel(self.norm_hv_iel(hv_feat))
        i_feat = i_feat + self.i_iel(self.norm_i_iel(i_feat))

        return hv_feat, i_feat


class LCUM(nn.Module):
    def __init__(self, kernel=7, tau=0.12, alpha=8.0, gamma=0.10, eps=1e-6):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError('kernel must be a positive odd integer')
        self.kernel, self.tau, self.alpha = kernel, tau, alpha
        self.gamma, self.eps = gamma, eps

    def forward(self, intensity):
        # Preserve the supplied implementation's zero-padded local mean.
        x = intensity.float()
        mu = F.avg_pool2d(x, self.kernel, 1, self.kernel // 2)
        w = (x - mu).abs() / (mu.abs() + self.eps)
        lo = w.amin(dim=(-2, -1), keepdim=True)
        hi = w.amax(dim=(-2, -1), keepdim=True)
        w = (w - lo) / (hi - lo + self.eps)
        return (1 + self.gamma * (torch.sigmoid(self.alpha * (w - self.tau)) - 0.5)).to(intensity.dtype)


def inject_update(inputs, candidates, coefficient):
    """Apply a shared spatial coefficient to each COMPLETE branch update."""
    if torch.is_tensor(coefficient) and coefficient.ndim == 4:
        coefficient = F.interpolate(coefficient, size=inputs[0].shape[-2:],
                                    mode='bilinear', align_corners=False)
    return tuple(x + coefficient * (y - x) for x, y in zip(inputs, candidates))


class CIUMNet(nn.Module):
    def __init__(self, channels=(36, 36, 72, 144), heads=(1, 2, 4, 8),
                 use_lcum=True):
        super().__init__()
        if len(channels) != 4 or len(heads) != 4:
            raise ValueError('Four channel widths and attention head counts are required')
        if any(c % h for c, h in zip(channels, heads)):
            raise ValueError('Each channel width must be divisible by its head count')
        self.model_config = dict(channels=list(channels), heads=list(heads), use_lcum=use_lcum)
        self.trans = HVITransform()
        self.lcum = LCUM() if use_lcum else None
        # Figure 2 explicitly assigns two H/V input channels and one I channel.
        self.hv_stem = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(2, channels[0], 3, bias=False))
        self.i_stem = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(1, channels[0], 3, bias=False))
        self.hv_down = nn.ModuleList([NormDownsample(a, b) for a, b in zip(channels, channels[1:])])
        self.i_down = nn.ModuleList([NormDownsample(a, b) for a, b in zip(channels, channels[1:])])
        self.hv_up = nn.ModuleList([NormUpsample(channels[j], channels[j-1]) for j in (3, 2, 1)])
        self.i_up = nn.ModuleList([NormUpsample(channels[j], channels[j-1]) for j in (3, 2, 1)])
        self.bicius = nn.ModuleList([BiCIU(channels[j], heads[j]) for j in (1, 2, 3, 3, 2, 1)])
        self.hv_head = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(channels[0], 2, 3, bias=False))
        self.i_head = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(channels[0], 1, 3, bias=False))

    def forward(self, x, return_hvi=False, interaction_alpha3=1.0, interaction_alpha4=1.0):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError('Expected a nonempty Bx3xHxW RGB tensor in [0, 1]')
        height, width = x.shape[-2:]
        # No geometric resizing: pad to the three-stage stride and crop afterwards.
        x = F.pad(x, (0, (-width) % 8, 0, (-height) % 8), mode='replicate')
        hvi = self.trans.HVIT(x)
        hv, i = self.hv_stem(hvi[:, :2]), self.i_stem(hvi[:, 2:])
        skips = [(hv, i)]
        for stage in range(3):
            hv, i = self.hv_down[stage](hv), self.i_down[stage](i)
            if stage < 2:
                hv, i = self.bicius[stage](hv, i)
                skips.append((hv, i))
        modulation = self.lcum(hvi[:, 2:]) if self.lcum is not None else 1.0
        for index, alpha in ((2, interaction_alpha3), (3, interaction_alpha4)):
            inputs = (hv, i)
            candidates = self.bicius[index](*inputs)
            hv, i = inject_update(inputs, candidates, modulation * alpha)
        for stage in range(3):
            skip_hv, skip_i = skips[2-stage]
            hv, i = self.hv_up[stage](hv, skip_hv), self.i_up[stage](i, skip_i)
            if stage < 2:
                hv, i = self.bicius[4+stage](hv, i)
        restored_hvi = hvi + torch.cat((self.hv_head(hv), self.i_head(i)), dim=1)
        rgb = self.trans.PHVIT(restored_hvi)[..., :height, :width]
        restored_hvi = restored_hvi[..., :height, :width]
        return (rgb, restored_hvi) if return_hvi else rgb

    def HVIT(self, x):
        return self.trans.HVIT(x)
