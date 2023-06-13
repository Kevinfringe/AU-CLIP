'''
    Customized latent mapper by increasing the number of layers and dimension of each layer.
'''
import torch
from torch import nn
from torch.nn import Module

from stylegan2.model import EqualLinear, PixelNorm

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]


class Mapper(Module):
    '''
        Base model of a mapper. 
        Modify the number of layers in the network.
    '''
    def __init__(self, opts, au_dim=8, latent_dim=512):
        super(Mapper, self).__init__()
        dimin = 5
        self.opts = opts
        # PixelNorm aims to divides each element of the input tensor by the square root of the mean of squared elements of that pixel's feature vector. This ensures that the pixel features have zero mean and unit variance
        layers = [PixelNorm()]
        layers.append(
                        EqualLinear(
                            au_dim, au_dim *2, lr_mul=0.01, activation='fused_lrelu'
                        )
                     )
        layers.append(
                        EqualLinear(
                            au_dim * 2, au_dim *4, lr_mul=0.01, activation='fused_lrelu'
                        )
                     )
        layers.append(
                        EqualLinear(
                            au_dim * 4, au_dim * 8, lr_mul=0.01, activation='fused_lrelu'
                        )
                     )
        layers.append(
                        EqualLinear(
                            au_dim * 8, au_dim * 16, lr_mul=0.01, activation='fused_lrelu'
                        )
                     )
        layers.append(
                        EqualLinear(
                            au_dim * 16, au_dim * 32, lr_mul=0.01, activation='fused_lrelu'
                        )
                     )
        layers.append(
                        EqualLinear(
                            au_dim * 32, au_dim * 64, lr_mul=0.01, activation='fused_lrelu'
                        )
                     )

        for i in range(3):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x


class SingleMapper(Module):
    '''
        A single mapper for mapping the au code to w code directly.
    '''

    def __init__(self, opts):
        super(SingleMapper, self).__init__()

        self.opts = opts

        self.mapping = Mapper(opts)

    def forward(self, x):
        out = self.mapping(x)
        return out


class LevelsMapper(Module):

    def __init__(self, opts):
        super(LevelsMapper, self).__init__()

        self.opts = opts

        if not opts.no_coarse_mapper:
            self.course_mapping = Mapper(opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = Mapper(opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = Mapper(opts)

    def forward(self, x, w):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]
        
        w_coarse = w[:, :4, :]
        w_medium = w[:, 4:8, :]
        w_fine = w[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(w_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(w_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(w_fine)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out

class FullStyleSpaceMapper(Module):

    def __init__(self, opts):
        super(FullStyleSpaceMapper, self).__init__()

        self.opts = opts

        for c, c_dim in enumerate(STYLESPACE_DIMENSIONS):
            setattr(self, f"mapper_{c}", Mapper(opts, latent_dim=c_dim))

    def forward(self, x):
        out = []
        for c, x_c in enumerate(x):
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            out.append(x_c_res)

        return out


class WithoutToRGBStyleSpaceMapper(Module):

    def __init__(self, opts):
        super(WithoutToRGBStyleSpaceMapper, self).__init__()

        self.opts = opts

        indices_without_torgb = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
        self.STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in indices_without_torgb]

        for c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
            setattr(self, f"mapper_{c}", Mapper(opts, latent_dim=STYLESPACE_DIMENSIONS[c]))

    def forward(self, x):
        out = []
        for c in range(len(STYLESPACE_DIMENSIONS)):
            x_c = x[c]
            if c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            else:
                x_c_res = torch.zeros_like(x_c)
            out.append(x_c_res)

        return out