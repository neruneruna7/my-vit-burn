import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm

from torchvision.models.vision_transformer import VisionTransformer

def create_custom_vit():
    # 1. パラメータ定義 (Rustの定数をPythonに移植)
    IMAGE_WH = 32
    CHANNEL = 3

    # ViT Hyper Params
    PATCH_WH = 8
    # SPLIT_ROWS/COLS はモデル定義には直接不要ですが計算ロジックとして記載
    SPLIT_ROWS = IMAGE_WH // PATCH_WH
    SPLIT_COLS = IMAGE_WH // PATCH_WH
    PATCH_TOTAL = (IMAGE_WH // PATCH_WH) ** 2
    
    PATCH_VECTOR_LEN = CHANNEL * (PATCH_WH ** 2)  # 3 * 64 = 192
    EMBED_VECTOR_LEN = PATCH_VECTOR_LEN // 2      # 192 / 2 = 96

    # Transformer Params
    HEAD = 12
    # 注意: 通常ViTでは mlp_dim = 4 * hidden_dim ですが，
    # ここでは指定通り EMBED_VECTOR_LEN (96) とします
    DIM_FEEDFORWARD = EMBED_VECTOR_LEN 
    ACTIVATION = "gelu" # torchvisionではデフォルトでgeluが使用されます
    LAYERS = 12
    
    NUM_CLASSES = 10 # CIFAR-10用

    # 2. パラメータの整合性チェック
    # hidden_dim は num_heads で割り切れる必要があります
    if EMBED_VECTOR_LEN % HEAD != 0:
        raise ValueError(f"EMBED_VECTOR_LEN ({EMBED_VECTOR_LEN}) must be divisible by HEAD ({HEAD})")

    # 3. VisionTransformerクラスの初期化
    # torchvisionの汎用クラスを使用することで任意の構成を実現します
    model = VisionTransformer(
        image_size=IMAGE_WH,          # 32
        patch_size=PATCH_WH,          # 8
        num_layers=LAYERS,            # 12
        num_heads=HEAD,               # 12
        hidden_dim=EMBED_VECTOR_LEN,  # 96
        mlp_dim=DIM_FEEDFORWARD,      # 96
        num_classes=NUM_CLASSES,      # 10
        dropout=0.0,                  # 指定がなければデフォルト0
        attention_dropout=0.0         # 指定がなければデフォルト0
    )

    return model

def main():
    # 1. デバイスの設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    print(f"Using device: {device}")

    # 2. 前処理の設定
    # ViT-B/16はデフォルトで224x224の入力を想定しているためリサイズを行う
    # 注意: 32x32を224x224に拡大するため，計算負荷が高く，情報は増えない
    transform = transforms.Compose([
        # transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 3. データセットの読み込み
    batch_size = 64 # GPUメモリに応じて調整が必要
    

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # 4. モデルの定義 (Vision Transformer)
    # weights=Noneを指定してスクラッチから学習する
    model = create_custom_vit() 
    model = model.to(device)    
    # # CIFAR-10用に出力層(ヘッド)を1000クラスから10クラスに変更
    # # ViTの実装によっては heads.head あるいは head など構造が異なる場合があるため確認が必要
    # # torchvisionのViTでは `heads` モジュールの中に `head` (Linear) がある
    # model.heads.head = nn.Linear(model.heads.head.in_features, 10)
    
    # model = model.to(device)

    # 5. 損失関数とオプティマイザの設定
    criterion = nn.CrossEntropyLoss()
    # ViTの学習にはAdamWが推奨される
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # 学習スケジューラ (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 6. 学習ループ
    epochs = 10
    print(f"Start training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
            
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 統計情報の更新
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # プログレスバーの末尾に現在のLossとAccを表示
                # リアルタイムで値が更新されていきます
                current_loss = running_loss / (total / inputs.size(0)) # 平均Lossの概算
                current_acc = 100 * correct / total
                tepoch.set_postfix(loss=current_loss, acc=f"{current_acc:.2f}%")
        scheduler.step()
        
        # エポックごとの検証
        validate(model, testloader, device)

    print('Finished Training')

def validate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')

if __name__ == '__main__':
    main()