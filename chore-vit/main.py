import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

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
        chunked = torch.chunk(img,splitRow,dim=2)
        print("After Chunk Shape:", [c.shape for c in chunked])
        horizontal = torch.stack(chunked,dim=1)
        print("After Stack Shape:", horizontal.shape)

        col_chunked = torch.chunk(horizontal,splitCol,dim=4)
        print("After Column Chunk Shape:", [c.shape for c in col_chunked])
        patches = torch.cat(col_chunked,dim=1)
        print("After Concat Patches Shape:", patches.shape)
        return patches

    def forward(self,x):
        # それぞれのテンソルの形状を確認する
        print("=== Forward Pass ===")
        print("Input Shape:", x.shape)
        x=self.patchify(x)
        print("After Patchify Shape:", x.shape)
        x=torch.flatten(x,start_dim=2)
        print("After Flatten Shape:", x.shape)
        x=self.patchEmbedding(x)
        print("After Patch Embedding Shape:", x.shape)
        clsToken = self.cls.repeat_interleave(x.shape[0],dim=0)
        x=torch.cat((clsToken,x),dim=1)
        x+=self.positionEmbedding
        x=self.transformerEncoder(x)
        print("After Transformer Encoder Shape:", x.shape)
        print("CLS Token Shape:", x[:,0,:].shape)
        x=self.mlpHead(x[:,0,:])
        print("Output Shape:", x.shape)
        print("=====================\n")
        return x

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") # Mac用
    else:
        return torch.device("cpu")

def main():
    # 1. デバイスの設定
    device = get_device()
    print(f"Using device: {device}")

    # 2. データセットの前処理と読み込み
    print("Preparing Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Data Augmentation
        transforms.RandomHorizontalFlip(),    # Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR10のダウンロード
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    # 3. モデル、損失関数、オプティマイザの定義
    print("Initializing Model...")
    model = ViT().to(device)
    
    criterion = nn.CrossEntropyLoss()
    # ViTはAdamWなどがよく使われますが、ここではシンプルなAdamを使用
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. 学習ループ
    epochs = 1 # お試しなので少なめに設定
    print(f"Start Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 99:    # 100ミニバッチごとに表示
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}, acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0
             
        # エポックごとの検証（Validation）
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        print(f'--> Epoch {epoch + 1} Test Accuracy: {100.*test_correct/test_total:.2f}%')

    print("Finished Training")

if __name__ == "__main__":
    main()