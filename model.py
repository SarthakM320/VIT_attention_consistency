import torch
import timm
from torch import nn
from einops import rearrange

class Model(nn.Module):
    def __init__(self,num_classes = 30, pretrained = True, freeze = True, layer = -1):
        super().__init__()
        self.layer = layer
        # self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained = pretrained)
        self.model = timm.create_model('vit_small_patch16_224',num_classes = 0, pretrained=pretrained)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            # for p in range(len(self.model.blocks)):
            #     for param in self.model.blocks[p].attn.qkv.parameters():
            #         param.requires_grad = True
            #     for param in self.model.blocks[p].attn.parameters():
            #         param.requires_grad = True
                

        self.head = nn.Linear(self.model.embed_dim, num_classes)
        self.num_heads = self.model.blocks[layer].attn.num_heads
    
    def forward(self, inp1, inp2):
        feats = []
        # Using multiple layers 
        B,C,H,W = inp1.shape

        def hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0]
            k = qkv[1]
            v = qkv[2]
            scale = q.shape[-1] ** -0.5
            attn = (q@k.transpose(-2, -1))*scale
            attn = attn.softmax(dim=-1)
            feats.append(attn)

        id = self.model.blocks[self.layer].attn.register_forward_hook(hook)

        output_1 = self.head(self.model(inp1))
        # removing the cls token
        self_attn_1 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_1.shape[-2]**0.5)
        # self_attn_1 = self_attn_1.reshape(B,self.num_heads,dim,dim,dim,dim) # b,12,14,14,14,14
        self_attn_1 = rearrange(self_attn_1, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)
        

        output_2 = self.head(self.model(inp2))
        # removing the cls token
        self_attn_2 = feats[-1][:,:,1:,1:]
        dim = int(self_attn_2.shape[-2]**0.5)
        self_attn_2 = rearrange(self_attn_2, 'b h (h1 w1) (h2 w2) -> b h h1 w1 h2 w2', h2 = dim, h1 = dim)
        # print(self_attn_2.shape)

        id.remove()

        return self_attn_1, output_1, self_attn_2, output_2




