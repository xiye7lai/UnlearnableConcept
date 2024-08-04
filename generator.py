import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, ngf=64, ex_dim=None, norm='batch',
                 use_dropout=False, n_blocks=24, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ngf = ngf

        use_bias = norm == 'instance'

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d

        self.act = nn.ReLU(True)

        encoder_list = [
            # 3*224*224
            nn.ReflectionPad2d(3),
            # 3*230*230
            nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0,
                      bias=use_bias),
            # 64*224*224
            norm_layer(ngf),
            self.act]

        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder_list += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                       stride=2, padding=1, bias=use_bias),
                             norm_layer(ngf * mult * 2),
                             self.act]
        # 256*56*56
        # 512*28*28

        mult = 2 ** n_downsampling

        if ex_dim is not None:
            Csqueeze_list = [
                nn.Conv2d(ngf * mult + ex_dim, ngf * mult, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ngf * mult),
                self.act
            ]

        bottle_neck_list = []
        for i in range(n_blocks):
            bottle_neck_list += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        decoder_list = []

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder_list += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                                                padding=1, output_padding=1, bias=use_bias),
                             norm_layer(int(ngf * mult / 2)),
                             self.act]
        decoder_list += [nn.ReflectionPad2d(3)]
        decoder_list += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        decoder_list += [nn.Tanh()]

        self.encoder = nn.Sequential(*encoder_list)
        if ex_dim is not None:
            self.Csqueeze = nn.Sequential(*Csqueeze_list)
        self.bottle_neck = nn.Sequential(*bottle_neck_list)
        self.decoder = nn.Sequential(*decoder_list)

    def forward(self, x, embedding):
        x = self.encoder(x)
        new_emb = embedding.unsqueeze(2).expand(x.shape[0], embedding.shape[1], x.shape[2]).unsqueeze(2).expand(
            x.shape[0], embedding.shape[1], x.shape[2], x.shape[3])
        x = torch.cat([x, new_emb], dim=1)
        x = self.Csqueeze(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


class Generator_one_hot(nn.Module):
    def __init__(self):
        super(Generator_one_hot, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        #
        # # Intermediate layer (10-dimensional data)
        # self.intermediate = nn.Linear(128 * 56 * 56 + 10, 128 * 56 * 56)
        #
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            nn.Linear(522, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 3 * 224 * 224),
            nn.Unflatten(1, (3, 224, 224)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.encoder(x)
        #
        # # Flatten the output of the encoder
        # x = x.view(x.size(0), -1)
        #
        # # Concatenate the additional data
        # x = torch.cat((x, additional_data), dim=1)
        #
        # # Pass through the intermediate layer
        # x = self.intermediate(x)
        #
        # # Reshape back to the shape of the encoder output
        # x = x.view(x.size(0), 128, 56, 56)

        x = self.decoder(x)
        return x
