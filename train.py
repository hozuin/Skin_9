import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ⭐️ 같은 경로에 있는 model_hybrid.py에서 모델을 불러옵니다.
from model_hybrid import HybridSkinModel

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정 (수정됨)
# ==========================================
# 캡처해주신 폴더 구조에 맞춘 정확한 상대 경로입니다.
TRAIN_DIR = './Skin_9_data/Skin_train/augmented_data'
VAL_DIR = './Skin_9_data/Skin_validation/augmented_data'

BATCH_SIZE = 16               # VRAM 64GB 환경이므로 충분합니다.
EPOCHS = 10                   
LEARNING_RATE = 1e-4          

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 데이터 준비 및 전처리
# ==========================================
# 이미지를 512x512로 맞추고 텐서로 변환 (Train/Val 공통)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def get_dataloaders():
    print("지정된 폴더에서 데이터셋을 불러오는 중입니다...")
    
    # ⭐️ 폴더가 이미 분리되어 있으므로, 각각의 경로에서 데이터를 읽어옵니다.
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)
    
    class_names = train_dataset.classes
    print(f"인식된 클래스(6개): {class_names}")
    print(f"학습용(Train) 이미지: {len(train_dataset)}장")
    print(f"검증용(Validation) 이미지: {len(val_dataset)}장")

    # 데이터를 배치(Batch) 단위로 묶어주는 DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, class_names

# ==========================================
# 3. 메인 학습 루프 (Training Loop)
# ==========================================
def train_model():
    train_loader, val_loader, class_names = get_dataloaders()
    
    # 모델 불러오기 및 GPU 올리기
    model = HybridSkinModel(num_classes=len(class_names)).to(device)
    print(f"\n[{device}] 환경에서 RTX 5070 Ti 가속 학습을 시작합니다!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # 🟢 [학습 단계]
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="학습 중(Train)"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()       
            outputs = model(inputs)     
            loss = criterion(outputs, labels) 
            loss.backward()             
            optimizer.step()            
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 🔵 [검증 단계] 
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="검증 중(Val)"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs, 1) 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_acc = 100 * correct / total
        print(f"결과 ➔ Train Loss: {epoch_loss:.4f} | Validation Accuracy: {epoch_acc:.2f}%")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_skin_model.pth')
            print("⭐️ 최고 정확도 갱신! 모델이 'best_skin_model.pth'로 저장되었습니다.")

    print(f"\n모든 학습 완료! 최종 최고 정확도: {best_acc:.2f}%")

if __name__ == '__main__':
    train_model()