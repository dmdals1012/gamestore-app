import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import joblib

@st.cache_resource
def load_data_and_models():
    dataset = load_dataset("swamysharavana/steam_games.csv")
    df = pd.DataFrame(dataset['train'])
    df['genre'] = df['genre'].fillna('알 수 없음')
    df['developer'] = df['developer'].fillna('알 수 없음')
    
    nlp_model_path = hf_hub_download(repo_id="dmdals1012/steam-game-nlp-recommender", filename="nlp_model.pkl")
    nlp_model_data = joblib.load(nlp_model_path)
    nlp_embeddings = nlp_model_data['embeddings']
    
    content_model_path = hf_hub_download(repo_id="dmdals1012/steam-game-content-recommender", filename="content_based_model.pkl")
    content_model_data = joblib.load(content_model_path)
    content_embeddings = content_model_data['embeddings']
    
    collab_model_path = hf_hub_download(repo_id="dmdals1012/steam-game-collaborative-recommender", filename="collaborative_filtering_model.pkl")
    collab_model = joblib.load(collab_model_path)
    
    return df, nlp_embeddings, content_embeddings, collab_model

df, nlp_embeddings, content_embeddings, collab_model = load_data_and_models()

@st.cache_data
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

@st.cache_data
def get_content_recommendations(game_name, top_n=5):
    try:
        idx = df[df['name'].str.lower() == game_name.lower()].index[0]
    except IndexError:
        return pd.DataFrame()
    game_embedding = content_embeddings[idx]
    similarities = cosine_similarity([game_embedding], content_embeddings)[0]
    sim_scores = list(enumerate(similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    game_indices = [i[0] for i in sim_scores]
    return df.iloc[game_indices][['name', 'genre', 'developer']]

@st.cache_data
def get_collaborative_recommendations(game_name, top_n=5):
    try:
        game_id = df[df['name'].str.lower() == game_name.lower()].index[0]
    except IndexError:
        return pd.DataFrame()
    
    all_games = df.index.tolist()
    predictions = [collab_model.predict(game_id, other_game_id).est for other_game_id in all_games]
    
    top_indices = np.argsort(predictions)[::-1][1:top_n+1]
    
    return df.iloc[top_indices][['name', 'genre', 'developer']]

@st.cache_data
def ensemble_recommendations(nlp_recs, content_recs, collab_recs, weights=[1/3, 1/3, 1/3], top_n=5):
    all_games = set(nlp_recs['name'].tolist() + content_recs['name'].tolist() + collab_recs['name'].tolist())
    
    scores = {}
    for game in all_games:
        score = 0
        if game in nlp_recs['name'].values:
            score += weights[0] * (len(nlp_recs) - nlp_recs[nlp_recs['name'] == game].index[0])
        if game in content_recs['name'].values:
            score += weights[1] * (len(content_recs) - content_recs[content_recs['name'] == game].index[0])
        if game in collab_recs['name'].values:
            score += weights[2] * (len(collab_recs) - collab_recs[collab_recs['name'] == game].index[0])
        scores[game] = score
    
    top_games = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return df[df['name'].isin(top_games)][['name', 'genre', 'developer']]

def show():
    st.title('🎮 Steam 게임 추천')

    search_method = st.radio("🔍 검색 방법 선택:", ('게임 이름', '장르', '개발사'))

    if search_method == '게임 이름':
        game_name = st.text_input('🕹️ 게임 이름을 입력하세요:')
        if game_name:
            with st.spinner('AI가 게임을 분석 중입니다... 🤖'):
                nlp_recommendations = get_nlp_recommendations(game_name)
                content_recommendations = get_content_recommendations(game_name)
                collab_recommendations = get_collaborative_recommendations(game_name)
            
            if not nlp_recommendations.empty:
                st.subheader(f'🌟 {game_name}와(과) 유사한 게임 추천:')
                
                ensemble_recs = ensemble_recommendations(nlp_recommendations, content_recommendations, collab_recommendations)
                st.table(ensemble_recs.rename(columns={'name': '게임 이름', 'genre': '장르', 'developer': '개발사'}))
                
                st.info("ℹ️ 이 추천 목록은 NLP 기반, 콘텐츠 기반, 협업 필터링 기반 추천 시스템의 결과를 종합하여 만들어졌습니다.")
            else:
                st.warning('⚠️ 게임을 찾을 수 없습니다. 철자를 확인하고 다시 시도해주세요.')

    elif search_method == '장르':
        genres = sorted(df['genre'].unique())
        selected_genre = st.selectbox('🎭 장르를 선택하세요:', genres)
        games_in_genre = df[df['genre'] == selected_genre]['name'].tolist()
        selected_game = st.selectbox('🎮 게임을 선택하세요:', games_in_genre)
        if st.button('추천 받기 🚀'):
            with st.spinner('AI가 게임을 분석 중입니다... 🤖'):
                nlp_recommendations = get_nlp_recommendations(selected_game)
                content_recommendations = get_content_recommendations(selected_game)
                collab_recommendations = get_collaborative_recommendations(selected_game)
            
            ensemble_recs = ensemble_recommendations(nlp_recommendations, content_recommendations, collab_recommendations)
            st.subheader(f'🌟 {selected_game}와(과) 유사한 게임 추천:')
            st.table(ensemble_recs.rename(columns={'name': '게임 이름', 'genre': '장르', 'developer': '개발사'}))
            
            st.info("ℹ️ 이 추천 목록은 NLP 기반, 콘텐츠 기반, 협업 필터링 기반 추천 시스템의 결과를 종합하여 만들어졌습니다.")

    elif search_method == '개발사':
        developers = sorted(df['developer'].unique())
        selected_developer = st.selectbox('🏢 개발사를 선택하세요:', developers)
        games_by_developer = df[df['developer'] == selected_developer]['name'].tolist()
        selected_game = st.selectbox('🎮 게임을 선택하세요:', games_by_developer)
        if st.button('추천 받기 🚀'):
            with st.spinner('AI가 게임을 분석 중입니다... 🤖'):
                nlp_recommendations = get_nlp_recommendations(selected_game)
                content_recommendations = get_content_recommendations(selected_game)
                collab_recommendations = get_collaborative_recommendations(selected_game)
            
            ensemble_recs = ensemble_recommendations(nlp_recommendations, content_recommendations, collab_recommendations)
            st.subheader(f'🌟 {selected_game}와(과) 유사한 게임 추천:')
            st.table(ensemble_recs.rename(columns={'name': '게임 이름', 'genre': '장르', 'developer': '개발사'}))
            
            st.info("ℹ️ 이 추천 목록은 NLP 기반, 콘텐츠 기반, 협업 필터링 기반 추천 시스템의 결과를 종합하여 만들어졌습니다.")

    st.sidebar.subheader('📊 데이터셋 정보')
    st.sidebar.write(f"전체 게임 수: {len(df)}")
    st.sidebar.write(f"고유 장르 수: {df['genre'].nunique()}")
    st.sidebar.write(f"고유 개발사 수: {df['developer'].nunique()}")
