import cv2
import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def apply_semantic_segmentation(image_path):
    print("1. AI 모델을 불러오는 중입니다... (최초 1회 로딩 시 시간이 걸립니다)")
    
    # 🔥 에러의 원인이었던 pipeline()을 버리고, 모델의 진짜 이름(Segformer)을 직접 호출합니다!
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    
    # 2. 이미지 읽기
    original_image = Image.open(image_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    h, w, _ = image_cv.shape

    print("2. 픽셀 단위로 얼굴을 분석하는 중입니다...")
    
    # 3. AI 모델에 이미지 입력 준비 및 실행
    inputs = processor(images=original_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 4. 분석 결과 크기를 원본 이미지 크기와 똑같이 맞추기 (비율 조정)
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=original_image.size[::-1], # 원본 이미지의 (세로, 가로) 크기
        mode="bilinear",
        align_corners=False,
    )
    
    # 5. 각 픽셀별로 가장 확률이 높은 부위 번호 추출
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    # 강한 블러 배경 생성
    blurred_image = cv2.GaussianBlur(image_cv, (99, 99), 30)
    
    # 빈 마스크 도화지 생성 (검은색)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # 6. 타겟 부위 자동 탐색 (눈, 코, 입, 눈썹)
    # 모델에 저장된 부위 이름들 중에서, 우리가 가리고 싶은 부위의 번호만 쏙쏙 뽑아냅니다.
    target_names = [
        'nose', 'mouth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'u_lip', 'l_lip',
        'left_eye', 'right_eye', 'upper_lip', 'lower_lip', 'left_eyebrow', 'right_eyebrow'
    ]
    
    target_ids = []
    for id_num, label_name in model.config.id2label.items():
        if label_name.lower() in target_names:
            target_ids.append(int(id_num))

    # 7. 타겟 부위 번호에 해당하는 픽셀만 하얗게 칠하기 (마스크 완성)
    for label_id in target_ids:
        final_mask[pred_seg == label_id] = 255

    # 8. 원본 이미지와 블러 이미지 합성
    mask_3d = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    final_result = np.where(mask_3d == 255, blurred_image, image_cv)

    return final_result

# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    # 캡처 화면에 있던 test.png에 맞췄습니다.
    result_img = apply_semantic_segmentation("test.jpg")
    
    if result_img is not None:
        cv2.imwrite("segmentation_result.jpg", result_img)
        print("--- 픽셀 단위 마스킹이 완벽하게 끝났습니다! ---")
        
        # 결과 이미지를 창으로 띄워서 확인
        cv2.imshow("Semantic Segmentation Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()