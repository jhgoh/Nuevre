## Model based on the Perciever-IO
## Base code taken from https://github.com/krasserm/perceiver-io
## or https://github.com/lucidrains/perceiver-pytorch
## I will take the necessary part only
import torch
import einops
from typing import List, Optional, Tuple

## MultiHeadAttention for the Perciever-IO
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_qry, d_key, d_val, n_head, d_head, dropout=0.):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.d_hidden = n_head*d_head
        self.scale = d_head**0.5

        self.fc_qry = torch.nn.Linear(d_qry, self.d_hidden, bias=False)
        self.fc_key = torch.nn.Linear(d_key, self.d_hidden, bias=False)
        self.fc_val = torch.nn.Linear(d_val, self.d_hidden, bias=False)

        self.fc_out = torch.nn.Linear(self.d_hidden, d_qry)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, qry, key, val, mask=None):
        batch_size = key.shape[0]

        ## Transform query, key, value, to fit with the shape of hidden dimension
        qry = self.fc_qry(qry)
        key = self.fc_key(key)
        val = self.fc_val(val)
        
        ## Reshape input
        #qry = qry.view(batch_size, -1, self.n_head, self.d_head).permute(0,2,1,3)
        ##key = key.view(batch_size, -1, self.n_head, self.d_head).permute(0,2,1,3)
        #val = val.view(batch_size, -1, self.n_head, self.d_head).permute(0,2,1,3)
        #keyT = key.view(batch_size, -1, self.n_head, self.d_head).permute(0,2,3,1)
        qry = einops.rearrange(qry, 'b n (h d) -> (b h) n d', h=self.n_head)
        key = einops.rearrange(key, 'b n (h d) -> (b h) n d', h=self.n_head)
        val = einops.rearrange(val, 'b n (h d) -> (b h) n d', h=self.n_head)

        ## Get the similarity(energy) matrix, shape = (batch, n_head, d_head)
        #sim = torch.matmul(qry, keyT)
        sim = torch.einsum('b i d, b j d -> b i j', qry, key)
        sim /= self.scale
        
        if not mask is None:
            #mask = mask[:,:,0]
            #mask = torch.reshape(mask, mask.shape[0],1,1,mask.shape[1])
            #mask = mask.repeat(1, sim.shape[1], sim.shape[2], 1)
            mask = einops.rearrange(mask, 'b ... -> b (...)')
            mask = einops.repeat(mask, 'b j -> (b h) () j', h=self.n_head)
            sim.masked_fill_(~mask, -1e32)

        attn = torch.softmax(sim, dim=-1)
        sim = self.dropout(sim)

        #x = torch.matmul(attn, val)
        #x = x.permute(0,2,1,3).contiguous()
        #x = x.view(batch_size, -1, self.d_hidden)
        x = torch.einsum('b i j, b j d -> b i d', attn, val)
        x = einops.rearrange(x, '(b h) n d -> b n (h d)', h=self.n_head)
        x = self.fc_out(x)

        return x

class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()

        hidden = mult*dim

        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden, bias=True),
            torch.nn.GELU(), ## Gaussian Error Linear Unit, SOTA in language models
            #torch.nn.SiLU(), ## Sigmoid Linear Unit, also called as swish function
            #torch.nn.ReLU(),
            torch.nn.Linear(hidden, dim, bias=True),
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return x

class SelfAttention(MultiHeadAttention):
    def __init__(self, d_qkv, n_head, d_head, dropout=0.):
        super().__init__(d_qkv, d_qkv, d_qkv, n_head, d_head, dropout)
    
    def forward(self, x):
        return super().forward(x,x,x)

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, d_input, n_head, d_head, nLayers, dropout=0.):
        super().__init__()
        self.nLayers = nLayers

        #layers = []
        #for i in range(nLayers):
        #    layers.extend([
        #        SelfAttention(d_input, n_head, d_head, dropout),
        #    ])
        #self.layers = torch.nn.Sequential(layers)
        self.attention = SelfAttention(d_input, n_head, d_head, dropout)

    def forward(self, x):
        #x = self.layers(x)
        for m in range(self.nLayers):
            x = self.attention(x)
        return x

class PerceiverIO(torch.nn.Module):
    def __init__(self, d_input, d_latent, d_out,
                 n_head, d_head, n_attnLayers,
                 dropout=0.):
        super().__init__()

        self.latent = torch.nn.Parameter(torch.randn(d_latent))
        self.outQry = torch.nn.Parameter(torch.randn(d_out))

        self.encodeCA = MultiHeadAttention(d_latent, d_input, d_input, n_head, d_head, dropout)
        self.encodeFF = FeedForward(d_latent, d_latent)
        self.encodeCANorm = torch.nn.LayerNorm(d_latent)
        self.encodeFFNorm = torch.nn.LayerNorm(d_latent)
        self.processSA = SelfAttentionLayer(d_latent, n_head, d_head, n_attnLayers, dropout)
        self.decodeCA = MultiHeadAttention(d_out, d_latent, d_latent, n_head, d_head, dropout)
                                            
    def forward(self, input):
        batch_size = input.shape[0]
        #input = input.view(batch, -1)

        #latent = self.latent.repeat(batch,1,1)
        #outQry = self.outQry.repeat(batch,1,1)
        latent = einops.repeat(self.latent, '... -> b ...', b=batch_size)
        outQry = einops.repeat(self.outQry, '... -> b () ...', b=batch_size)
        latent = latent.view(batch_size,1,-1)

        x = self.encodeCA(latent, input, input)
        x = self.encodeCANorm(x)
        x = x + self.encodeFF(x)
        x = self.encodeFFNorm(x)

        x = self.processSA(x)
        x = self.decodeCA(outQry, x, x)
        
        return x

