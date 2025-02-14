import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import joblib

@st.cache_resource
def load_data_and_model():
    dataset = load_dataset("swamysharavana/steam_games.csv")
    df = pd.DataFrame(dataset['train'])
    df['genre'] = df['genre'].fillna('알 수 없음')
    df['developer'] = df['developer'].fillna('알 수 없음')
    model_path = hf_hub_download(repo_id="dmdals1012/steam-game-nlp-recommender", filename="nlp_model.pkl")
    model_data = joblib.load(model_path)
    embeddings = model_data['embeddings']
    return df, embeddings

df, embeddings = load_data_and_model()

def get_recommendations(game_name, top_n=5):
    try:
        idx = df[df['name'].str.lower() == game_name.lower()].index[0]
    except IndexError:
        return pd.DataFrame()
    game_embedding = embeddings[idx]
    similarities = cosine_similarity([game_embedding], embeddings)[0]
    sim_scores = list(enumerate(similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    game_indices = [i[0] for i in sim_scores]
    return df.iloc[game_indices][['name', 'genre', 'developer']]

st.title('Steam 게임 추천 시스템')

search_method = st.radio("검색 방법 선택:", ('게임 이름', '장르', '개발사'))

if search_method == '게임 이름':
    game_name = st.text_input('게임 이름을 입력하세요:')
    if game_name:
        recommendations = get_recommendations(game_name)
        if not recommendations.empty:
            st.subheader(f'{game_name}와(과) 유사한 게임 추천:')
            st.table(recommendations.rename(columns={'name': '게임 이름', 'genre': '장르', 'developer': '개발사'}))
        else:
            st.warning('게임을 찾을 수 없습니다. 철자를 확인하고 다시 시도해주세요.')

elif search_method == '장르':
    genres = sorted(df['genre'].unique())
    selected_genre = st.selectbox('장르를 선택하세요:', genres)
    games_in_genre = df[df['genre'] == selected_genre]['name'].tolist()
    selected_game = st.selectbox('게임을 선택하세요:', games_in_genre)
    if st.button('추천 받기'):
        recommendations = get_recommendations(selected_game)
        st.subheader(f'{selected_game}와(과) 유사한 게임 추천:')
        st.table(recommendations.rename(columns={'name': '게임 이름', 'genre': '장르', 'developer': '개발사'}))

elif search_method == '개발사':
    developers = sorted(df['developer'].unique())
    selected_developer = st.selectbox('개발사를 선택하세요:', developers)
    games_by_developer = df[df['developer'] == selected_developer]['name'].tolist()
    selected_game = st.selectbox('게임을 선택하세요:', games_by_developer)
    if st.button('추천 받기'):
        recommendations = get_recommendations(selected_game)
        st.subheader(f'{selected_game}와(과) 유사한 게임 추천:')
        st.table(recommendations.rename(columns={'name': '게임 이름', 'genre': '장르', 'developer': '개발사'}))

st.sidebar.subheader('데이터셋 정보')
st.sidebar.write(f"전체 게임 수: {len(df)}")
st.sidebar.write(f"고유 장르 수: {df['genre'].nunique()}")
st.sidebar.write(f"고유 개발사 수: {df['developer'].nunique()}")
