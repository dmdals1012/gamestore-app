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
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu

# 앱 타이틀과 설명
st.set_page_config(page_title="Steam 게임 추천", page_icon="🎮", layout="wide")

def get_huggingface_token():   
    token = os.environ.get('HUGGINGFACE_API_TOKEN')
    if token is None:
        token = st.secrets.get('HUGGINGFACE_API_TOKEN')
    return token

@st.cache_resource
def initialize_models(llm_model_name="google/gemma-2-9b-it"): # 기본 모델 변경
    token = get_huggingface_token()
    llm = HuggingFaceInferenceAPI(
        model_name=llm_model_name,
        temperature=0.5,
        system_prompt = """
당신은 Steam 플랫폼에서 제공되는 다양한 게임에 대한 깊은 지식을 가진 전문적인 게임 추천 AI 어시스턴트입니다. 다음 지침을 **반드시** 따르세요:

    1. 사용자의 질문을 **정확하고 완전하게** 분석하여 사용자의 의도를 파악하세요.
    2. Steam에서 제공하는 게임을 **최대한 활용**하여 사용자의 취향에 맞는 게임을 선별하세요.
    3. 추천하는 게임에 대해 **자세한 설명**과 함께 **구체적인 추천 이유**를 제시하세요.
    4. 사용자의 요청에 따라 다양한 게임(인기 게임, 숨겨진 명작, 특정 태그 게임 등)을 **제한 없이** 추천하세요.
    5. Steam의 주요 기능에 대한 정보도 **필요한 경우 자세하게** 제공하세요.
    6. 특정 게임에 대한 문의에는 핵심 특징과 **다양한 유사 게임**을 **충분히** 추천하세요.
    7. 게임 관련 용어는 초보자도 이해할 수 있도록 **쉽게 설명**하세요.
    8. 답변은 항상 **완전한 문장**으로 작성하고, 가독성을 위해 **적절히 단락을 나누세요**.
    9. 사용자의 특성(연령, 경험, 취향)을 고려하여 **가장 적합한 게임**을 추천하세요.
    10. 최신 트렌드, 세일 정보, 사용자 평가를 **반영**하여 신뢰성 높은 추천을 제공하세요.
    11. **[답변이 중간에 끊기지 않도록 모든 문장을 완전하게 마무리하고, 필요한 경우 추가 설명을 덧붙여 답변을 풍부하게 만드세요. 답변 길이에 제한을 두지 말고, 필요한 만큼 충분히 자세하게 설명하세요. 답변을 생성할 때 필요한 경우 추가적인 정보를 검색하거나 추론하여 제공하세요. 당신은 항상 완전하고 자세한 답변을 제공해야 합니다. 답변이 불완전하거나 중간에 끊기는 일이 없도록 하세요.](pplx://action/followup)**
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

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def app():
    global df, nlp_embeddings, collab_model
    df, nlp_embeddings, collab_model = load_data_and_models()
    
    # 헤더 섹션
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("🎮 Steam 게임 추천 시스템")
        st.subheader("당신의 다음 최애 게임을 찾아보세요!")
    with header_col2:
        lottie_gaming = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_xyadoh9h.json")
        st_lottie(lottie_gaming, height=150)

    # 사이드바 설정
    with st.sidebar:
        st.subheader("📊 데이터셋 정보")
        st.info(f"🎮 전체 게임 수: {len(df)}")
        st.info(f"🏷️ 고유 장르 수: {df['genre'].nunique()}")
        st.info(f"🏢 고유 개발사 수: {df['developer'].nunique()}")
        
        st.markdown("---")
        st.subheader("🔍 검색 방법 선택")
        
        # option_menu를 사용하여 검색 방법 선택
        search_method = option_menu(
            menu_title=None,
            options=["게임 이름", "장르", "개발사", "챗봇"],
            icons=["controller", "tags", "building", "robot"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#0083B8"},
            }
        )

        st.markdown("---")
        st.subheader("ℹ️ 추천 방법 안내")
        if search_method == "게임 이름":
            st.write("재밌게 플레이 했던 게임과 비슷한 게임을 찾고 싶다면 추천 받아보세요!")
        elif search_method == "장르":
            st.write("특정 장르의 인기 게임을 찾고 있다면 추천 받아보세요!")
        elif search_method == "개발사":
            st.write("좋아하는 개발사의 다른 게임이 궁금하다면 추천 받아보세요!")
        elif search_method == "챗봇":
            st.write("궁금한 점이나 원하는 게임 스타일을 자유롭게 질문해보세요!")

    initialize_models()
    index = get_index_from_huggingface()

    # 메인 컨텐츠
    if search_method == '챗봇':
        st.subheader("🤖 게임 추천 챗봇")
        st.write("어떤 게임을 찾고 있는지 챗봇에게 자유롭게 물어보세요. 몇 가지 예시 질문:")
        st.write("- '스토리 위주의 RPG 게임 추천해줘'")
        st.write("- '친구와 함께 할 수 있는 협동 게임 알려줘'")
        st.write("- '최근에 인기 있는 인디 게임 추천해줘'")
        user_input = st.text_input("질문을 입력하세요:")
        if user_input:
            with st.spinner('AI가 답변을 생성 중입니다... 🤖'):
                query_engine = index.as_query_engine()
                response = query_engine.query(user_input)
                st.success(response.response)

    elif search_method == '게임 이름':
        game_names = sorted(df['name'].unique(), key=lambda x: (x is None, x))
        st.subheader("🎮 어떤 게임과 비슷한 게임을 찾으시나요?")  # 제목 추가
        selected_game = st.selectbox('🕹️ 게임을 선택하세요:', game_names)
        if st.button('추천 받기 🚀', key='game_name_button'):
            with st.spinner('AI가 게임을 분석 중입니다... 🤖'):
                nlp_recommendations = get_nlp_recommendations(selected_game)
                collab_recommendations = get_collaborative_recommendations(selected_game)
            
            if not nlp_recommendations.empty:
                st.subheader(f'🌟 {selected_game}와(과) 유사한 게임 추천:')
                
                ensemble_recs = ensemble_recommendations(nlp_recommendations, collab_recommendations)
                display_game_cards(ensemble_recs)
                
                st.info("ℹ️ 이 추천 목록은 NLP 기반, 협업 필터링 기반 추천 시스템의 결과를 종합하여 만들어졌습니다.")
            else:
                st.warning('⚠️ 선택한 게임에 대한 추천을 생성할 수 없습니다.')

    elif search_method == '장르':
        genres = sorted(df['genre'].unique())
        st.subheader("🎭 어떤 장르의 게임을 찾으시나요?")  # 제목 추가
        selected_genre = st.selectbox('🎭 장르를 선택하세요:', genres)
        if st.button('추천 받기 🚀', key='genre_button'):
            with st.spinner('AI가 게임을 분석 중입니다... 🤖'):
                genre_recommendations = get_recommendations_by_genre(selected_genre)
            
            st.subheader(f'🌟 {selected_genre} 장르의 추천 게임:')
            display_game_cards(genre_recommendations)
            
            st.info("ℹ️ 이 추천 목록은 선택한 장르를 기반으로 만들어졌습니다.")

    elif search_method == '개발사':
        developers = sorted(df['developer'].unique())
        st.subheader("🏢 어떤 개발사의 게임을 찾으시나요?")  # 제목 추가
        selected_developer = st.selectbox('🏢 개발사를 선택하세요:', developers)
        if st.button('추천 받기 🚀', key='developer_button'):
            with st.spinner('AI가 게임을 분석 중입니다... 🤖'):
                developer_recommendations = get_recommendations_by_developer(selected_developer)
            
            st.subheader(f'🌟 {selected_developer}의 추천 게임:')
            display_game_cards(developer_recommendations)
            
            st.info("ℹ️ 이 추천 목록은 선택한 개발사를 기반으로 만들어졌습니다.")

    # 푸터
    st.markdown("---")
    st.markdown("Made with ❤️ by Your Game Recommendation Team")

if __name__ == "__main__":
    app()
