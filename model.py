##
import torch
import torch.nn as nn

##
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model =nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(1, 1), bias=False)
        )

    def forward(self, input):
        output = self.model(input)
        return output


##
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.model(input)
        return output


##
class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.GD = Decoder()


    def forward(self,latent):
        img = self.GD(latent)
        return img

##
class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.encoder = Encoder()
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1)
        )


    def forward(self, x):
        features = self.encoder(x) #[4,4]
        # print(features.shape)
        pred = self.classifier(features)
        # pred = F.sigmoid(pred)
        return features, pred


if __name__ == '__main__':
    net = Decoder()
    x = torch.rand(1,64,1,1)
    y = net(x)
    print(y.shape)
