# Skin_9
# 🩺 딥러닝 기반 안면 피부질환 예측 프로그램 (Hybrid Ensemble Model)
**선문대학교 종합 프로젝트 (2026)**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)
![React Native](https://img.shields.io/badge/React_Native-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)

## 📌 프로젝트 개요 (Overview)
본 프로젝트는 기존 근접 촬영 방식의 피부 진단 앱이 가지는 한계를 극복하고, **사용자가 스마트폰으로 안면 전체를 촬영(셀카)하는 것만으로 정밀한 피부 상태 확인이 가능한 차세대 진단 애플리케이션**입니다. 

불필요한 병원 방문을 줄이고 조기 진단을 유도하여 의료 접근성을 혁신하며, 특히 진단 과정에서 발생할 수 있는 개인정보 문제를 원천 차단하는 **'Privacy-by-Design'** 아키텍처를 도입했습니다.

### 🎯 목표 성능
* 주요 피부 질환 5종(건선, 아토피, 여드름, 주사, 지루성 피부염)에 대해 **90% 이상의 테스트 정확도** 달성.

## 💡 핵심 기술 파이프라인 (Core 3-Step Pipeline)

### 1️⃣ On-Device 기반 안면 마스킹 (Privacy Protection)
* **기술:** Google MediaPipe Face Landmarker API
* **특징:** 서버 전송 전, 클라이언트(모바일) 단에서 478개의 랜드마크를 추출하여 눈, 코, 입 등 비질환 영역을 즉시 마스킹(제외)합니다. 이를 통해 불필요한 연산 부하를 줄이고, 민감한 개인 얼굴 정보 유출을 사전에 차단합니다.

### 2️⃣ 순수 피부 무손실 분할 (Lossless Segmentation)
* **기술:** OpenCV, NumPy
* **특징:** 원본 고화질을 유지한 채, 마스킹되지 않은 유효 피부 구역을 512x512 크기의 패치 단위로 슬라이딩 윈도우 방식으로 분할하여 병변의 미세 질감을 보존합니다.

### 3️⃣ 하이브리드 AI 앙상블 진단 (Hybrid Architecture)
* **기술:** EfficientNetV2 + Swin Transformer
* **특징:** 이미지의 국소적 특징(Local Features) 추출에 강한 CNN(EfficientNetV2)과, 안면 전체의 문맥적 관계(Global Context) 파악에 뛰어난 Transformer 모델을 앙상블(결합)하여 진단 성능을 극대화했습니다.

## 📊 데이터셋 및 전처리 (Data Pipeline)
* **출처:** AI HUB 안면부 피부질환 이미지 데이터셋 (연구 목적 활용)
* **클래스 구성 (총 6종):** 건선, 아토피, 여드름, 주사, 지루성 피부염 + 정상 피부
* **데이터 증강 (Data Augmentation):** 의료 데이터 특성상 부족한 데이터와 클래스 불균형을 해결하기 위해, 원본 6,000장 데이터를 3배 증강하여 **총 18,000장**의 대규모 학습 파이프라인을 구축했습니다.

## ⚙️ 기술 스택 및 인프라 (Tech Stack & Infra)
| 분야 | 사용 기술 |
| :--- | :--- |
| **Deep Learning** | PyTorch |
| **Main Model** | EfficientNetV2, Swin Transformer |
| **Backend** | FastAPI |
| **Frontend / App** | React Native, Android Native |
| **Database** | PostgreSQL (Supabase) |
| **Server Hosting** | Google Cloud Run |
| **Computing H/W** | NVIDIA GeForce RTX 5070 Ti, 64GB RAM |

## 🛡 협업 및 버전 관리 (Collaboration)
* **Git & GitHub:** 기능 단위(Branch) 개발 전략을 통해 팀원 간 코드 충돌을 방지하고 소스코드의 이력을 투명하게 관리합니다.

## ⚠️ 라이선스 및 데이터 윤리 (Ethics Guidelines)
* 본 프로젝트의 소스 코드는 **MIT License**를 따릅니다.
* **데이터 보안:** AI HUB에서 제공하는 공공 데이터는 외부 유출 방지를 위해 철저히 보안 관리되며, 연구 목적으로만 사용됩니다. 학습된 데이터(`original_data`, `augmented_data`)는 `.gitignore` 처리되어 깃허브에 업로드되지 않습니다.
