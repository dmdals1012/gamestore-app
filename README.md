# 🎮 Steam 게임 추천 시스템

> **당신의 다음 최애 게임을 찾는 여정을 도와주는 AI 기반 추천 시스템**

Streamlit App : https://gamestore-app-8zysuajnm3ohhzjprxpkkj.streamlit.app/



---

## 🧠 주요 기능

- 🔍 **게임 이름 기반 추천**  
  → NLP 및 협업 필터링을 사용해 비슷한 게임 추천

- 🎲 **장르 기반 추천**  
  → 선택한 장르 내에서 랜덤 게임 추천

- 🏗️ **개발사 기반 추천**  
  → 좋아하는 개발사의 다른 게임 탐색

- 🤖 **AI 챗봇 추천**  
  → 자연어 질문을 통해 맞춤형 게임 추천

- 🖥️ **Streamlit UI**  
  → 누구나 쉽게 사용할 수 있는 웹 인터페이스 제공

---

## 📦 사용된 기술 스택

| 분류      | 기술 |
|-----------|------|
| **백엔드** | Python, Streamlit |
| **모델**   | 🤗 HuggingFace LLM (`gemma-2-9b-it`), Sentence Transformers |
| **데이터** | [Steam Games Dataset](https://huggingface.co/datasets/swamysharavana/steam_games.csv) |
| **모델 저장소** | Hugging Face Hub |
| **LLM 연동** | `llama-index`, `HuggingFaceInferenceAPI` |
| **추천 방식** | NLP 기반 유사도 + 협업 필터링 앙상블 |

---

## 🛠️ 설치 및 실행 방법

1. **필수 패키지 설치**

```bash
pip install -r requirements.txt

- streamlit
- datasets
- llama-index
- huggingface_hub
- scikit-learn
- joblib
- streamlit_option_menu
- streamlit-lottie
```


2. **Hugging Face Token 설정**

.streamlit/secrets.toml 파일 생성 후 다음 내용 추가:
 - HUGGINGFACE_API_TOKEN = "huggingface_token"



**🧠 추천 방식 상세**
1. 🔠 NLP 기반 추천
- 문장 임베딩 모델: paraphrase-multilingual-MiniLM-L12-v2

- 게임 설명을 임베딩하여 입력 게임과 코사인 유사도 계산

2. 👥 협업 필터링
- 사용자-게임 평점 행렬을 기반으로 ALS (Alternating Least Squares) 모델 사용

3. 🔀 앙상블 추천
- 두 방식의 결과를 가중 평균으로 결합
- 기본 가중치: weights = [0.5, 0.5]

4. 🤖 AI 챗봇 기능
- 모델: google/gemma-2-9b-it (HuggingFace Inference API 기반)

- 역할: 전문 게임 추천 AI

- 검색: llama-index 기반 벡터 검색

예시 질문:

- "스토리 몰입감 높은 게임 추천해줘"

- "친구랑 할 수 있는 멀티 게임 알려줘"

- "디아블로 같은 게임 더 있어?"








