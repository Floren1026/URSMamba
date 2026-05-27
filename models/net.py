import math
import torch
from torch import nn
from models.vmamba_modify import VSSM
# from models.vmamba import VSSM
from models.wavelet import DWT_2D, IDWT_2D

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Spatial(nn.Module):
    def __init__(self, hidden_dim, depths=[2,4], use_residual=True, use_proj=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        
        self.mamba = VSSM(depths=depths, embed_dim=hidden_dim)
                    
        if self.use_proj:
            self.proj = nn.Sequential(
                LayerNorm2d(hidden_dim),
                nn.SiLU()
            )

    def forward(self,x):
        x_spa = self.mamba(x)

        if self.use_proj:
            x_proj = self.proj(x_spa)
        if self.use_residual:
            return x_proj + x
        else:
            return x_proj


class SpatialMb(nn.Module):
    def __init__(self, in_chans, out_chans, hidden_dim=64, depths=[4, 4]):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.dwt = DWT_2D('haar')
        self.iwt = IDWT_2D('haar')
        d = len(depths)
        n = int(d/2)
        self.lowfre = Lowfre(in_chans, hidden_dim*4, depths=depths[:n], use_residual=True, use_proj=True)
        self.highfre = Hifre(in_chans*3, hidden_dim*4, depths=depths[n:], use_residual=True, use_proj=True)

    def forward(self, x):
        x_dwt = self.dwt(x)
        x_LL, x_high = x_dwt[:,:self.in_chans, ...], x_dwt[:,self.in_chans:, ...]

        mamba_LL = self.lowfre(x_LL)
        mamba_high = self.highfre(x_high)

        x_mamba = mamba_LL + mamba_high

        x_iwt = self.iwt(x_mamba)

        return x_iwt


class SpectralMb(nn.Module):
    def __init__(self, imgsize, hidden_dim, depths=[2,4], group_num=16, byrow=True, use_residual=True):
        super().__init__()
        self.group_num = group_num
        self.byrow = byrow
        self.use_residual = use_residual

        H, W = imgsize, imgsize
        assert H % self.group_num == 0, f"Image's height should be divided by group_num"
        assert W % self.group_num == 0, f"Image's width should be divided by group_num"
        self.Wg = int(W / self.group_num)
        self.Hg = int(H / self.group_num)
      
        self.mamba = VSSM(depths=depths, embed_dim=self.group_num)

        self.proj = nn.Sequential(
            LayerNorm2d(hidden_dim),
            nn.SiLU()
        )

    def forward(self, x):
        if self.byrow:
            x_trans = x.permute(3, 0, 1, 2).contiguous()
        else:
            x_trans = x.permute(2, 0, 1, 3).contiguous()

        W, B, C, H = x_trans.shape
        x_trans = x_trans.view(self.Wg, self.group_num, B*C, H)
        x_spe = self.mamba(x_trans)
        x_spe = x_spe.view(W, B, C, H)
        x_spe = x_spe.permute(1, 2, 3, 0).contiguous()
        x_proj = self.proj(x_spe)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class Hifre(nn.Module):
    def __init__(self, in_chans, hidden_dim=64, depths=[4, 4], use_residual=True, use_proj=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.patch_embedding = nn.Sequential(nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
                                             nn.Identity(),
                                             LayerNorm2d(hidden_dim)
                                             )
        
        self.mamba = VSSM(depths=depths, embed_dim=hidden_dim)

        if self.use_proj:
            self.proj = nn.Sequential(
                LayerNorm2d(hidden_dim),
                nn.SiLU()
            )

    def forward(self, x):
        x_embed = self.patch_embedding(x)
        x_mamba = self.mamba(x_embed)

        if self.use_proj:
            x_proj = self.proj(x_mamba)
        if self.use_residual:
            return x_proj + x_embed
        else:
            return x_proj


class Lowfre(nn.Module):
    def __init__(self, in_chans, hidden_dim=64, depths=[4, 4], use_residual=True, use_proj=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.patch_embedding = nn.Sequential(nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
                                             nn.Identity(),
                                             LayerNorm2d(hidden_dim)
                                             )
        
        self.mamba = VSSM(depths=depths, embed_dim=hidden_dim)

        if self.use_proj:
            self.proj = nn.Sequential(
                LayerNorm2d(hidden_dim),
                nn.SiLU()
            )

    def forward(self,x):
        x_embed = self.patch_embedding(x)
        x_mamba = self.mamba(x_embed)

        if self.use_proj:
            x_proj = self.proj(x_mamba)
        if self.use_residual:
            return x_proj + x_embed
        else:
            return x_proj


class SSDF(nn.Module):
    def __init__(self, in_chans, out_chans, imgsize, hidden_dim, depths=[2, 2, 4, 4], group_num=4, use_residual=True, use_att=True):
        super().__init__()  
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.patch_embedding = nn.Sequential(nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
                                             nn.Identity(),
                                             LayerNorm2d(hidden_dim)
                                             )

        depths_spe = depths[1::2]
        self.spa_mamba = SpatialMb(in_chans, out_chans, hidden_dim, depths)

        self.spe_mambaR = SpectralMb(imgsize, hidden_dim, depths=depths_spe, group_num=group_num, byrow=True, use_residual=use_residual)
        self.spe_mambaC = SpectralMb(imgsize, hidden_dim, depths=depths_spe, group_num=group_num, byrow=False, use_residual=use_residual)

    def forward(self, x):
        x_embed = self.patch_embedding(x)

        spa_x = self.spa_mamba(x)
        
        spe_xC = self.spe_mambaC(x_embed)
        spe_xR = self.spe_mambaR(x_embed)
        spe_x = spe_xC + spe_xR


        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x

        if self.use_residual:
            return fusion_x + x_embed
        else:
            return fusion_x

            


class Net(nn.Module):
    def __init__(self, in_chans=6, out_chans=3, imgsize=144, hidden_dim=64, depths=[2, 4], 
                 group_num=4, mode='ssdf', use_residual=True, use_att=True):
        super().__init__()
        self.mode = mode
        self.patch_embedding = nn.Sequential(nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),
                                             nn.Identity(),
                                             LayerNorm2d(hidden_dim)
                                             )

        if mode == 'spectral':
            self.mamba = SpectralMb(imgsize, hidden_dim, depths, group_num=group_num, use_residual=use_residual)
        elif mode == 'spatial':
            self.mamba = Spatial(hidden_dim, depths, use_residual=use_residual)
        elif mode=='ssdf':
            self.mamba = SSDF(in_chans, out_chans, imgsize, hidden_dim, depths=depths, group_num=group_num, use_residual=use_residual, use_att=use_att)

        self.img_cons = nn.Sequential(
            LayerNorm2d(hidden_dim), # B,H,W,C
            nn.Identity(),
            nn.Conv2d(hidden_dim, out_chans, 3, 1, 1)
            )

    def forward(self, x):
        x_embed = self.patch_embedding(x)
        if self.mode == 'ssdf':
            x_mamba = self.mamba(x)
        else:
            x_mamba = self.mamba(x_embed)

        res = x_mamba + x_embed
        x_mamba = self.img_cons(res)

        return x_mamba

