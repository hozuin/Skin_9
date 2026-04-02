import torch
import torch.nn as nn
import torch.nn.functional as F  # ⭐️ 이미지를 축소/확대하는 마법의 도구 추가
import timm

class HybridSkinModel(nn.Module):
    def __init__(self, num_classes=6):
        super(HybridSkinModel, self).__init__()
        
        # 1. EfficientNetV2 (현미경)
        self.cnn = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', 
                                     pretrained=True, 
                                     num_classes=0)
        cnn_feature_dim = self.cnn.num_features  

        # 2. Swin Transformer (망원경)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', 
                                      pretrained=True, 
                                      num_classes=0)
        swin_feature_dim = self.swin.num_features  

        # 3. Feature Fusion 
        self.dropout = nn.Dropout(p=0.3) 
        self.classifier = nn.Linear(cnn_feature_dim + swin_feature_dim, num_classes)

    def forward(self, x):
        # x는 현재 train.py에서 넘어온 512x512 이미지입니다.
        
        # 🟢 Step A-1: EfficientNet에는 512x512 해상도 그대로 통과 (질감 파악)
        feat_cnn = self.cnn(x)    
        
        # 🔵 Step A-2: Swin Transformer에는 224x224로 축소해서 통과 (전체 맥락 파악) ⭐️ 에러 해결!
        x_swin = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feat_swin = self.swin(x_swin)  
        
        # Step B: 두 특징(단서) 융합
        fused_features = torch.cat((feat_cnn, feat_swin), dim=1) 
        
        # Step C: 최종 분류
        out = self.dropout(fused_features)
        output = self.classifier(out) 
        
        return output

if __name__ == "__main__":
    model = HybridSkinModel(num_classes=6)
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print("✅ 에러 해결! 512x512 이미지가 두 갈래로 나뉘어 성공적으로 처리되었습니다.")