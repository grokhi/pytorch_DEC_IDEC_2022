import torch
import torch.nn as nn


'''
AUTOENCODER
'''

class Block(nn.Module):
    def __init__(
        self, 
        inp_dim: int, 
        out_dim: int,
        activation: nn.Module,
        dropout_value: float
    ):
        super().__init__()
        self.fc1 = nn.Linear(inp_dim, out_dim)
        self.act = activation()
        self.dropout = nn.Dropout(dropout_value)
 
    def forward(self, x):
        return self.dropout(
            self.act(self.fc1(x))
        )

class StackedAutoEncoder(nn.Module):
    '''
    Fully connected auto-encoder model, symmetric.

    Arguments:
        dims (list): list of number of units in each layer of encoder. dims[0] is input dims, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        activation (nn.Module): activation function, not applied to Input, Hidden and Output layers
        dropout_value (float): dropout value mostly used for controlling the learning curves
        base_block (nn.Module): layer of neural net
    
    Return:
        Model of autoencoder
    '''
    def __init__(
        self,
        dims:list,
        activation:nn.Module = nn.ReLU,
        dropout_value:float = .0,
        base_block:nn.Module = Block,
    ):

        super().__init__()
        self.dims = dims
        self.inp_dim = dims[0]
        self.hid_dim = dims[-1]
        self.dropout_value = dropout_value
        self.activation = activation

        # encoder
        encoder_blocks = []
        encoder_blocks.extend([
            base_block(self.inp_dim, dims[1], nn.Identity, dropout_value),
            base_block(dims[1], dims[2], activation, dropout_value),
            base_block(dims[2], dims[3], activation, dropout_value),
            base_block(dims[3], self.hid_dim, nn.Identity, dropout_value),
        ])

        self.encoder = nn.Sequential(*encoder_blocks) 

        # decoder
        decoder_blocks = []
        decoder_blocks.extend([
            base_block(self.hid_dim, dims[3], activation, dropout_value),
            base_block(dims[3], dims[2], activation, dropout_value),
            base_block(dims[2], dims[1], activation, dropout_value),
            base_block(dims[1], self.inp_dim, nn.Identity, dropout_value), # nn.Identity is when f(x)==x
        ])
        
        self.decoder = nn.Sequential(*decoder_blocks)


    def set_dropout_value(self, value:float):
        self.dropout_value = value

        for i in range(len(self.encoder)):
            self.encoder[i].dropout.p = self.dropout_value
        for i in range(len(self.decoder)):
            self.decoder[i].dropout.p = self.dropout_value


    def forward(self, x):
        self.set_dropout_value(self.dropout_value) # inplace

        return self.decoder(self.encoder(x))

'''
DENOISING AUTOENCODER
'''

class DenoisingBlock(Block):
    def __init__(
        self, 
        inp_dim: int, 
        out_dim: int,
        activation: nn.Module,
        dropout_value: float
    ):
        super().__init__(inp_dim, out_dim, activation, dropout_value)
                
    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.05

        return self.act(
            self.dropout(self.fc1(x))
        )


class StackedDenoisingAutoEncoder(StackedAutoEncoder):
    '''
    Fully connected auto-encoder model, symmetric.

    Arguments:
        dims (list): list of number of units in each layer of encoder. dims[0] is input dims, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        activation (nn.Module): activation function, not applied to Input, Hidden and Output layers
        dropout_value (float): dropout value mostly used for controlling the learning curves
        base_block (nn.Module): layer of neural net
        
    Return:
        Model of denoising autoencoder
    '''
    def __init__(
        self, 
        dims:list, 
        activation: nn.Module = nn.ReLU,
        dropout_value: float = .0,
        base_block: nn.Module = DenoisingBlock,
    ):
        super().__init__(dims, activation, dropout_value, base_block,)

    def forward(self, x):
        if self.training:
            x = torch.clip(x + torch.randn_like(x) * 0.1, min=0, max=1)

        return self.decoder(self.encoder(x))

