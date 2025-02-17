import streamlit as st
from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def app():
    st.title('🎮 Steam 게임 추천 시스템에 오신 것을 환영합니다!')
    
    # 애니메이션 추가
    lottie_gaming = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_xyadoh9h.json")
    st_lottie(lottie_gaming, height=200)
    
    st.write("""
    ## 🚀 당신의 다음 최애 게임을 찾아보세요!
    
    최첨단 AI 기술을 활용하여 당신의 취향에 꼭 맞는 게임을 추천해 드립니다.
    
    ### 🔍 주요 기능:
    - 게임 이름, 장르, 개발사 기반 맞춤형 추천
    - AI 챗봇을 통한 인터랙티브 게임 추천
    - NLP와 협업 필터링을 결합한 앙상블 추천 시스템
    - 데이터 기반 게임 트렌드 및 인사이트 제공
    
    ### 🎯 시작하기:
    1. 사이드바에서 원하는 추천 방식을 선택하세요.
    2. 게임 이름, 장르, 개발사를 입력하거나 AI 챗봇과 대화해보세요.
    3. AI가 분석한 맞춤형 게임 추천을 확인하세요!
    
    ### 💡 우리의 추천 시스템:
    - 자연어 처리(NLP) 기술로 게임의 숨겨진 특징을 파악합니다.
    - 협업 필터링으로 유저들의 선호도 패턴을 분석합니다.
    - 고급 앙상블 기법으로 정확하고 다양한 추천을 제공합니다.
    - AI 챗봇이 실시간으로 당신의 취향을 파악하여 게임을 추천합니다.
    
    게임의 무한한 세계로 여러분을 초대합니다! 🕹️
    """)
    
    st.image("https://cdn.akamai.steamstatic.com/steam/apps/593110/header.jpg", caption="Steam 게임의 세계로 오신 것을 환영합니다!")

    col1, col2 = st.columns(2)
    with col1:
        st.info("ℹ️ 이 앱은 Streamlit, Hugging Face, LlamaIndex 등의 최신 기술로 개발되었습니다.")
    with col2:
        st.success("🌟 자세한 시스템 소개는 '시스템 소개' 페이지를 확인해주세요!")

    st.markdown("---")
    st.markdown("© 2025 Steam 게임 추천 시스템 | 모든 권리 보유")
