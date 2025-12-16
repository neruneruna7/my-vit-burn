import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

#image param
imageWH = 32
channel=3

#vit hyperparam
patchWH=8
splitRow=imageWH//8
splitCol=imageWH//8
patchTotal=(imageWH//patchWH)**2 #(32 / 8)^2 = 16
patchVectorLen=channel*(patchWH**2) #3 * 64 = 192
embedVectorLen=int(patchVectorLen/2)

#transformer layer hyperparam
head=12
dim_feedforward=embedVectorLen
activation="gelu"
layers=12

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.patchEmbedding = nn.Linear(patchVectorLen,embedVectorLen)
        self.cls = nn.Parameter(torch.zeros(1, 1, embedVectorLen))
        self.positionEmbedding = nn.Parameter(torch.zeros(1, patchTotal + 1, embedVectorLen))
        encoderLayer = TransformerEncoderLayer(
            d_model=embedVectorLen,
            nhead=head,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformerEncoder = TransformerEncoder(encoderLayer,layers)
        self.mlpHead=nn.Linear(embedVectorLen,10)

    def patchify(self,img):
        horizontal = torch.stack(torch.chunk(img,splitRow,dim=2),dim=1)
        patches = torch.cat(torch.chunk(horizontal,splitCol,dim=4),dim=1)
        return patches

    def forward(self,x):
        x=self.patchify(x)
        x=torch.flatten(x,start_dim=2)
        x=self.patchEmbedding(x)
        clsToken = self.cls.repeat_interleave(x.shape[0],dim=0)
        x=torch.cat((clsToken,x),dim=1)
        x+=self.positionEmbedding
        x=self.transformerEncoder(x)
        x=self.mlpHead(x[:,0,:])
        return x

def main():
    print("Hello from chore-vit!")

    vit = ViT()


if __name__ == "__main__":
    main()
