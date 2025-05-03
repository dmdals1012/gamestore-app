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

본 프로젝트의 추천 시스템은 콘텐츠 기반 필터링과 협업 필터링을 결합한 하이브리드(앙상블) 구조로, 각각의 장점을 살려 더 정확하고 다양한 게임 추천을 제공합니다.



1. 🔠  콘텐츠 기반 추천 (NLP 임베딩)
- 각 게임의 이름, 설명, 태그, 장르, 개발사 등 다양한 텍스트 정보를 BERT, SentenceTransformer 등 최신 임베딩 모델로 벡터화합니다.

- 사용자가 입력한 게임과 텍스트 유사도가 높은 게임을 코사인 유사도로 계산하여 추천합니다.

2. 👥 협업 필터링 추천 (SVD)
- 실제 또는 시뮬레이션된 유저-게임 평점 데이터를 기반으로 SVD(특이값 분해) 모델을 학습합니다.

- 특정 사용자가 아직 플레이하지 않은 게임 중, 평점 예측값이 높은 게임을 추천합니다.

3. 🔀 앙상블(하이브리드) 추천 방식
- 두 추천 결과(콘텐츠 기반, 협업 필터링)를 가중 평균 방식으로 결합합니다.

- 각 추천 결과의 점수를 0~1 사이로 정규화 후, 아래와 같이 최종 점수를 계산합니다:
    
    - 최종 점수 = 콘텐츠 기반 점수 × w1 + 협업 필터링 점수 × w2
(w1, w2는 가중치, 기본값 0.5:0.5, 사용 환경에 따라 조정 가능)

- 최종 점수가 높은 순으로 게임을 추천합니다.

- 추천 결과가 충분히 다양하지 않거나, 추천이 실패할 경우 인기 게임/장르 기반 폴백 추천을 제공합니다.



4. 🤖 AI 챗봇 기능
- 모델: google/gemma-2-9b-it (HuggingFace Inference API 기반)

- 역할: 전문 게임 추천 AI

- 검색: llama-index 기반 벡터 검색

예시 질문:

- "스토리 몰입감 높은 게임 추천해줘"

- "친구랑 할 수 있는 멀티 게임 알려줘"

- "디아블로 같은 게임 더 있어?"



**🛠️ 개발 과정 상세 설명**
제공된 코드를 기반으로 Steam 게임 추천 시스템의 개발 과정을 단계별로 설명합니다. 전체 프로세스는 데이터 준비 → 콘텐츠 기반 추천 → 협업 필터링 → 모델 저장/배포 순으로 진행됩니다.

1. 데이터 준비 및 전처리

    1-1 데이터셋 로드

    ```bash
    from datasets import load_dataset
    dataset = load_dataset('swamysharavana/steam_games.csv')
    df = pd.DataFrame(dataset['train'])
    ```
    1.2 텍스트 정제 및 특성 결합

    ```bash
    def clean_text(text):
        return re.sub(r'[^\w\s]', '', text.lower()) if isinstance(text, str) else ''

    df['combined_features'] = (
        df['name'].fillna('') + ' ' +
        df['desc_snippet'].fillna('') + ' ' +
        df['game_description'].fillna('') + ' ' +
        df['popular_tags'].fillna('') + ' ' +
        df['genre'].fillna('')
    ).apply(clean_text)
    ```

2. 콘텐츠 기반 추천 시스템

    2.1 문장 임베딩 생성

    ```bash
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(df['combined_features'].tolist(), show_progress_bar=True)
    ```

    2.2 코사인 유사도 계산

    ```bash
    cosine_sim = cosine_similarity(embeddings)
    ```

    2.3 추천 함수 구현

    ```bash
    def get_recommendations(title, cosine_sim=cosine_sim, df=df):
    idx = df[df['name'] == title].index[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:11]
    return df.iloc[[i[0] for i in sim_scores]][['name', 'genre', 'developer']]
    ```

    2.4 모델 테스트

    ```bash
    테스트 함수

    def get_recommendations_with_info(title, df, cosine_sim=cosine_sim):
        idx = df[df['name'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # 상위 10개 추천
        game_indices = [i[0] for i in sim_scores]
        
        recommendations = df.iloc[game_indices][['name', 'genre', 'developer', 'original_price']]
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        return recommendations
    ```

    ```bash
    테스트 결과

                                        name                       genre  \
    269                       NieR:Automata™                  Action,RPG   
    1887  Darksiders II Deathinitive Edition        Action,Adventure,RPG   
    1683                 GRIP: Combat Racing         Action,Indie,Racing   
    467                            Far Cry 3            Action,Adventure   
    1282               Worms Ultimate Mayhem                    Strategy   
    2772                         The Vagrant  Action,Adventure,Indie,RPG   
    1254             Trine Enchanted Edition      Action,Adventure,Indie   
    385                       Risk of Rain 2   Action,Indie,Early Access   
    8980                    SYNTHETIK: Arena   Action,Free to Play,Indie   
    3006                Superfighters Deluxe                Action,Indie   

                                                developer original_price  \
    269                      Square Enix,PlatinumGames Inc.         $39.99   
    1887               Gunfire Games,Vigil Games,THQ Nordic         $29.99   
    1683                                 Caged Element Inc.         $29.99   
    467   Ubisoft Montreal, Massive Entertainment, and U...         $19.99   
    1282                                 Team17 Digital Ltd         $14.99   
    2772                                        O.T.K Games          $3.99   
    1254                                         Frozenbyte         $14.99   
    385                                         Hopoo Games         $19.99   
    8980                                    Flow Fire Games   Free to Play   
    3006                             MythoLogic Interactive          $9.99   

        similarity_score  
    ...
    1254          0.969935  
    385           0.969844  
    8980          0.969818  
    3006          0.969677  
    유사도 점수 평균 0.96
    ```



    
    

3. 협업 필터링 시스템

    3.1 사용자-게임 평점 행렬 생성

    ```bash
    # 리뷰 점수 5점 척도 변환
    def extract_review_score(review_text):
        match = re.search(r'(\d+)%', review_text)
        return int(match.group(1))/20 if match else None

    df['score'] = df['all_reviews'].apply(extract_review_score)
    ```

    3.2 SVD 모델 학습
    ```bash
    from surprise import SVD, Dataset, Reader

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'name', 'score']], reader)
    svd = SVD()
    svd.fit(data.build_full_trainset())
    ```

    3.3 사용자별 추천 생성

    ```bash
    def get_collab_recommendations(user_id, n=10):
    predictions = [svd.predict(user_id, game) for game in df['name']]
    return sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    ```

    3.4 모델 테스트

    ```bash
    테스트 함수
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    테스트 결과

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                        Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.9190  0.9291  0.9450  0.9265  0.9390  0.9317  0.0092  
    MAE (testset)     0.7419  0.7473  0.7611  0.7490  0.7573  0.7513  0.0070  
    Fit time          0.18    0.17    0.16    0.17    0.16    0.17    0.01    
    Test time         0.02    0.02    0.02    0.02    0.02    0.02    0.00 
    ```
4. 시스템 통합 및 테스트

    4.1 하이브리드 추천 엔진

    ```bash
    def hybrid_recommend(user_id=None, game_title=None, weights=[0.5, 0.5]):
        content_rec = get_recommendations(game_title) if game_title else []
        collab_rec = get_collab_recommendations(user_id) if user_id else []
        
        # 점수 정규화 및 가중 평균
        combined = pd.merge(content_rec, collab_rec, on='name', how='outer')
        combined['final_score'] = weights[0]*combined['content_score'] + weights[1]*combined['collab_score']
        
        return combined.sort_values('final_score', ascending=False).head(10)
    ```

    4.2 테스트

    ```bash
    def hybrid_recommend(game_title, user_id, n=5, w_content=0.5, w_collab=0.5):
        """
        콘텐츠 기반 + 협업 필터링 앙상블 추천 함수 예시

        Parameters:
            game_title (str): 기준이 되는 게임 이름 (예: 'DOOM')
            user_id (int): 사용자 ID
            n (int): 추천 개수
            w_content (float): 콘텐츠 기반 추천 가중치
            w_collab (float): 협업 필터링 추천 가중치

        Returns:
            pd.DataFrame: 추천 게임 리스트 (게임명, 장르, 개발사, 설명 등)
        """

        # 1. 콘텐츠 기반 추천 점수 (예시: 유사도)
        content_df = get_content_recommendations(game_title, topk=n*3)  # ['name', 'genre', 'developer', 'desc', 'content_score']

        # 2. 협업 필터링 추천 점수 (예시: 예측 평점)
        collab_df = get_collab_recommendations(user_id, topk=n*3)      # ['name', 'collab_score']

        # 3. 점수 정규화
        content_df['content_score'] = (content_df['content_score'] - content_df['content_score'].min()) / (content_df['content_score'].max() - content_df['content_score'].min() + 1e-8)
        collab_df['collab_score'] = (collab_df['collab_score'] - collab_df['collab_score'].min()) / (collab_df['collab_score'].max() - collab_df['collab_score'].min() + 1e-8)

        # 4. 점수 병합
        merged = pd.merge(content_df, collab_df, on='name', how='outer')
        merged = merged.fillna(0)

        # 5. 앙상블 점수 계산
        merged['ensemble_score'] = w_content * merged['content_score'] + w_collab * merged['collab_score']

        # 6. 상위 n개 추천
        result = merged.sort_values('ensemble_score', ascending=False).head(n)

        # 7. 필요한 컬럼만 반환
        return result[['name', 'genre', 'developer','ensemble_score']]

        테스트 결과

        게임명	           장르	                              개발사	                        앙상블 점수
        NieR:Automata™,	Action,                              RPG	Square Enix,                 0.925
        Darksiders II Deathinitive Edition,	Action, Gunfire Games, Vigil Games, THQ Nordic,	     0.903
        Dreaming Sarah OST,	[Adventure, Casual, Indie],	[Andre Chagas Silva, Anthony Septim],    0.881
        Half-Life 2 Soundtrack,	    Action,	                      Valve                          0.858
        Half-Life Soundtrack,	    Action,	                      Valve                          0.836




    ```


5. 모델 최적화 및 배포

    5.1 모델 저장

    ```bash
    import joblib

    model_data = {
        'content_embeddings': embeddings,
        'collab_model': svd,
        'game_names': df['name'].values
    }
    joblib.dump(model_data, 'steam_recommender.pkl')
    ```

    5.2 Hugging Face Hub 배포

    ```bash
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj='steam_recommender.pkl',
        repo_id="dmdals1012/steam-game-recommender",
        repo_type="model"
    )
    ```











