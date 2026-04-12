import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm

from torchvision.models.vision_transformer import VisionTransformer

SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 2


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


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

    seed_everything(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    # 3. データセットの読み込み
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS,
                                              worker_init_fn=seed_worker, generator=generator)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS,
                                             worker_init_fn=seed_worker, generator=generator)

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
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 6. 学習ループ
    print(f"Start training for {EPOCHS} epochs...")
    history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{EPOCHS}")
            
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 統計情報の更新
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                current_loss = running_loss / total
                current_acc = 100 * correct / total
                tepoch.set_postfix(loss=f"{current_loss:.3f}", acc=f"{current_acc:.2f}%")
        
        # エポックごとの検証
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        valid_loss, valid_acc = validate(model, testloader, criterion, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
        })

    print_summary(history)
    print('Finished Training')


def validate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss = running_loss / total
    valid_acc = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {valid_acc:.2f} %')
    return valid_loss, valid_acc


def print_summary(history):
    rows = [
        ("Train", "Accuracy", "train_acc"),
        ("Train", "Loss", "train_loss"),
        ("Valid", "Accuracy", "valid_acc"),
        ("Valid", "Loss", "valid_loss"),
    ]

    print()
    print("| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |")
    print("|-------|----------|----------|----------|----------|----------|")
    for split, metric, key in rows:
        minimum = min(history, key=lambda item: item[key])
        maximum = max(history, key=lambda item: item[key])
        print(
            f"| {split:<5} | {metric:<8} | {minimum[key]:<8.3f} | "
            f"{minimum['epoch']:<8} | {maximum[key]:<8.3f} | {maximum['epoch']:<8} |"
        )

if __name__ == '__main__':
    main()
