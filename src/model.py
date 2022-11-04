import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, add_avg_pool = True):
        super(ConvBlock, self).__init__()
        self.add_avg_pool = add_avg_pool

        self.conv_layer = torch.nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=1
        )
        self.batch_norm = torch.nn.BatchNorm1d(num_features=out_channels)
        self.elu = torch.nn.ELU()

        if self.add_avg_pool:
            self.avg_pool = torch.nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.batch_norm(out)
        out = self.elu(out)
        
        if self.add_avg_pool:
            out = self.avg_pool(out)

        return out

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        self.block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1024
        )

        self.block2 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=1024
        )

        self.block3 = ConvBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=512
        )

        self.block4 = ConvBlock(
            in_channels=256,
            out_channels=512,
            kernel_size=512
        )

        self.block5 = ConvBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=256
        )

        self.block6 = ConvBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=256
        )

        self.block7 = ConvBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=264,
            add_avg_pool= False
        )

        self.linear = torch.nn.Linear(
            in_features=512,
            out_features=128
        )

        self.elu = torch.nn.ELU()

        self.output = torch.nn.Linear(
            in_features=128,
            out_features=1
        )

    def forward(self, x):
        out = self.block1(x)
        #print(out.shape)
        out = self.block2(out)
        #print(out.shape)
        out = self.block3(out)
        #print(out.shape)
        out = self.block4(out)
        #print(out.shape)
        out = self.block5(out)
        #print(out.shape)
        out = self.block6(out)
        #print(out.shape)
        out = self.block7(out)
        #print(out.shape)
        out = torch.flatten(out, start_dim = 1)
        out = self.linear(out)
        out = self.elu(out)
        out = self.output(out)
        out = torch.sigmoid(out)
        #print(out.shape)
        return out