from torch.autograd import Variable
from DeepLearning.net.architecture.unet.unet_layers import *
import numpy as np

class unet(nn.Module):
    def __init__(self, channels_in, classes_out):
        super(unet, self).__init__()
        self.inc = inconv(channels_in, 64)
        self.down1 = down_block(64, 128)
        self.down2 = down_block(128, 256)
        self.down3 = down_block(256, 512)
        self.down4 = down_block(512, 1024)
        self.down5 = down_block(1024, 2048)
        self.down6 = down_block(2048, 2048)
        self.up1 = up_block(4096, 1024)
        self.up2 = up_block(2048, 512)
        self.up3 = up_block(1024, 256)
        self.up4 = up_block(512, 128)
        self.up5 = up_block(256, 64)
        self.up6 = up_block(128, 64)
        self.outc = outconv(64, classes_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)

        #print("after up1, before up2: ",x.shape)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        #print(x.shape)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        #print(x.shape)
        x = self.outc(x)
        return x

if __name__ == "__main__":
    model = unet(3,1).cuda()
    x = Variable(torch.FloatTensor(np.random.random((4, 3, 512, 512))).cuda())
    out = model(x)
    loss = torch.sum(out)
    loss.backward()