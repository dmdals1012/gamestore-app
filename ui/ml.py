import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
import joblib
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
import os

def get_huggingface_token():   
    token = os.environ.get('HUGGINGFACE_API_TOKEN')
    if token is None:
        token = st.secrets.get('HUGGINGFACE_API_TOKEN')
    return token

@st.cache_resource
def initialize_models():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    token = get_huggingface_token()
    llm = HuggingFaceInferenceAPI(
        model_name=model_name,
        max_new_tokens=2048,
        temperature=0.5,
        system_prompt = """
당신은 Steam 플랫폼에서 제공되는 다양한 게임에 대한 깊은 지식을 가진 전문적인 게임 추천 AI 어시스턴트입니다. 다음 지침을 따라 사용자에게 최적의 Steam 게임 추천 서비스를 제공하세요:

1. 사용자의 질문을 주의 깊게 분석하여 선호하는 장르, 플레이 스타일, 선호하는 그래픽 스타일, 멀티플레이어 여부, 난이도, 출시 연도 등을 파악하세요.
2. Steam에서 제공하는 게임을 우선적으로 추천하되, 게임의 주요 특징(장르, 스토리, 그래픽 스타일, 시스템 요구 사항 등)을 고려하여 사용자의 취향에 맞는 게임을 선별하세요.
3. 추천하는 게임에 대해 간략한 설명과 함께 추천 이유를 제시하세요. 예를 들어, "이 게임은 뛰어난 스토리텔링과 몰입감 있는 게임 플레이를 제공하며, 특히 [특정 요소]를 선호하는 플레이어에게 적합합니다."와 같이 설명하세요.
4. 사용자의 요청에 따라 인기 있는 최신 게임, 숨겨진 명작(인디 게임 포함), 특정 태그(예: 로그라이크, 오픈 월드, 협동 플레이 등)에 맞는 게임을 추천하세요.
5. Steam의 주요 기능(할인 이벤트, DLC, 모드 지원 여부, 플레이 타임 등)에 대한 정보도 필요할 경우 제공하세요.
6. 특정 게임을 문의하면 해당 게임의 핵심 특징과 함께 비슷한 게임도 추가로 추천하세요.
7. 게임 관련 전문 용어를 사용할 때는 간단한 설명을 덧붙여 초보자도 이해할 수 있도록 하세요.
8. 답변은 항상 완전한 문장으로 작성하고, 가독성을 높이기 위해 적절히 단락을 나누세요.
9. 사용자의 연령대, 게임 경험 수준, 취향 등을 고려하여 적절한 게임을 추천하세요.
10. 최신 Steam 트렌드, 세일 정보, 플레이어 평가 등을 바탕으로 신뢰성 높은 추천을 제공하세요.
11. 대답은 항상 한국어로 작성하며, 답변이 중간에 끊기지 않도록 문장을 완전하게 끝내고, 필요한 경우 추가 설명을 덧붙이세요.

당신의 목표는 사용자가 Steam에서 자신에게 가장 적합하고 재미있는 게임을 찾을 수 있도록 돕는 것입니다. 친절하고 열정적인 태도로 사용자와 소통하세요.
""",
        token=token
    )
    embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    Settings.llm = llm
    Settings.embed_model = embed_model

@st.cache_resource
def get_index_from_huggingface():
    repo_id = "dmdals1012/game-index"
    local_dir = "./game_index_storage"
    token = get_huggingface_token()
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type='dataset',
        token=token
    )
    storage_context = StorageContext.from_defaults(persist_dir=local_dir)
    index = load_index_from_storage(storage_context)
    return index

@st.cache_resource
def load_data_and_models():
    dataset = load_dataset("swamysharavana/steam_games.csv")
    df = pd.DataFrame(dataset['train'])
    df['genre'] = df['genre'].fillna('알 수 없음')
    df['developer'] = df['developer'].fillna('알 수 없음')
    nlp_model_path = hf_hub_download(repo_id="dmdals1012/steam-game-nlp-recommender", filename="nlp_model.pkl")
    nlp_model_data = joblib.load(nlp_model_path)
    nlp_embeddings = nlp_model_data['embeddings']
    collab_model_path = hf_hub_download(repo_id="dmdals1012/steam-game-collaborative-recommender", filename="collaborative_filtering_model.pkl")
    collab_model = joblib.load(collab_model_path)
    return df, nlp_embeddings, collab_model

def get_nlp_recommendations(game_name, top_n=5):
    try:
        idx = df[df['name'].str.lower() == game_name.lower()].index[0]
    except IndexError:
        return pd.DataFrame()
    game_embedding = nlp_embeddings[idx]
    similarities = cosine_similarity([game_embedding], nlp_embeddings)[0]
    sim_scores = list(enumerate(similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    game_indices = [i[0] for i in sim_scores]
    return df.iloc[game_indices][['name', 'genre', 'developer']]

def get_collaborative_recommendations(game_name, top_n=5):
    try:
        game_id = df[df['name'].str.lower() == game_name.lower()].index[0]
    except IndexError:
        return pd.DataFrame()
    all_games = df.index.tolist()
    predictions = [collab_model.predict(game_id, other_game_id).est for other_game_id in all_games]
    top_indices = np.argsort(predictions)[::-1][1:top_n+1]
    return df.iloc[top_indices][['name', 'genre', 'developer']]

def ensemble_recommendations(nlp_recs, collab_recs, weights=[0.5, 0.5], top_n=5):
    all_games = set(nlp_recs['name'].tolist() + collab_recs['name'].tolist())
    scores = {}
    for game in all_games:
        score = 0
        if game in nlp_recs['name'].values:
            score += weights[0] * (len(nlp_recs) - nlp_recs[nlp_recs['name'] == game].index[0])
        if game in collab_recs['name'].values:
            score += weights[1] * (len(collab_recs) - collab_recs[collab_recs['name'] == game].index[0])
        scores[game] = score
    top_games = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return df[df['name'].isin(top_games)][['name', 'genre', 'developer']]

def get_recommendations_by_genre(genre, top_n=5):
    games_in_genre = df[df['genre'] == genre]
    return games_in_genre.sample(min(len(games_in_genre), top_n))[['name', 'genre', 'developer']]

def get_recommendations_by_developer(developer, top_n=5):
    games_by_developer = df[df['developer'] == developer]
    return games_by_developer.sample(min(len(games_by_developer), top_n))[['name', 'genre', 'developer']]

def generate_game_description(game):
    return f"{game['name']}은(는) {game['genre']} 장르의 게임으로, {game['developer']}에서 개발했습니다. 이 게임은 {game['genre']}의 특징을 잘 보여주며, 플레이어들에게 독특한 경험을 제공합니다. 자세한 정보는 Steam 스토어에서 확인할 수 있습니다."

def display_game_cards(games):
    for _, game in games.iterrows():
        description = game.get('description', generate_game_description(game))
        st.markdown(f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            width: 100%;
        ">
            <h3 style="color: #1e88e5;">{game['name']}</h3>
            <p><strong>장르:</strong> {game['genre']}</p>
            <p><strong>개발사:</strong> {game['developer']}</p>
            <p><strong>설명:</strong> {description}</p>
        </div>
        """, unsafe_allow_html=True)

def app():
    global df, nlp_embeddings, collab_model
    df, nlp_embeddings, collab_model = load_data_and_models()

    if 'description' not in df.columns:
        df['description'] = df.apply(generate_game_description, axis=1)

    st.title('🎮 Steam 게임 추천')

    initialize_models()
    index = get_index_from_huggingface()

    search_method = st.radio("🔍 검색 방법 선택:", ('게임 이름', '장르', '개발사', '챗봇'))

    if search_method == '챗봇':
        st.subheader("🤖 게임 추천 챗봇")
        user_input = st.text_input("게임에 대해 무엇이든 물어보세요:")
        if user_input:
            with st.spinner('AI가 답변을 생성 중입니다... 🤖'):
                query_engine = index.as_query_engine()
                response = query_engine.query(user_input)
                st.write(response.response)

    elif search_method == '게임 이름':
        game_name = st.text_input('🕹️ 게임 이름을 입력하세요:')
        if game_name:
            with st.spinner('AI가 게임을 분석 중입니다... 🤖'):
                nlp_recommendations = get_nlp_recommendations(game_name)
                collab_recommendations = get_collaborative_recommendations(game_name)
            
            if not nlp_recommendations.empty:
                st.subheader(f'🌟 {game_name}와(과) 유사한 게임 추천:')
                
                ensemble_recs = ensemble_recommendations(nlp_recommendations, collab_recommendations)
                display_game_cards(ensemble_recs)
                
                st.info("ℹ️ 이 추천 목록은 NLP 기반, 협업 필터링 기반 추천 시스템의 결과를 종합하여 만들어졌습니다.")
            else:
                st.warning('⚠️ 게임을 찾을 수 없습니다. 철자를 확인하고 다시 시도해주세요.')

    elif search_method == '장르':
        genres = sorted(df['genre'].unique())
        selected_genre = st.selectbox('🎭 장르를 선택하세요:', genres)
        if st.button('추천 받기 🚀'):
            with st.spinner('AI가 게임을 분석 중입니다... 🤖'):
                genre_recommendations = get_recommendations_by_genre(selected_genre)
            
            st.subheader(f'🌟 {selected_genre} 장르의 추천 게임:')
            display_game_cards(genre_recommendations)
            
            st.info("ℹ️ 이 추천 목록은 선택한 장르를 기반으로 만들어졌습니다.")

    elif search_method == '개발사':
        developers = sorted(df['developer'].unique())
        selected_developer = st.selectbox('🏢 개발사를 선택하세요:', developers)
        if st.button('추천 받기 🚀'):
            with st.spinner('AI가 게임을 분석 중입니다... 🤖'):
                developer_recommendations = get_recommendations_by_developer(selected_developer)
            
            st.subheader(f'🌟 {selected_developer}의 추천 게임:')
            display_game_cards(developer_recommendations)
            
            st.info("ℹ️ 이 추천 목록은 선택한 개발사를 기반으로 만들어졌습니다.")

    st.sidebar.subheader('📊 데이터셋 정보')
    st.sidebar.write(f"전체 게임 수: {len(df)}")
    st.sidebar.write(f"고유 장르 수: {df['genre'].nunique()}")
    st.sidebar.write(f"고유 개발사 수: {df['developer'].nunique()}")
