import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go


def create_feature_card(icon, title, description):
    st.markdown(f"""
    <div style="
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h3 style="color: #1e88e5;">{icon} {title}</h3>
        <p style="color: #333;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def app():
    st.title('📖 Steam 게임 추천 시스템 소개')
    
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["AI 추천 시스템", "기술 스택", "주요 기능", "성능 분석", "향후 계획"],
            icons=["robot", "gear", "list-check", "graph-up", "calendar"],
            menu_icon="house",
            default_index=0,
        )
    
    if selected == "AI 추천 시스템":
        st.header("🧠 첨단 AI 기반 추천 시스템")
        
        create_feature_card("📚", "NLP 기반 추천", 
            "자연어 처리 기술을 활용하여 게임 설명과 리뷰를 심층 분석합니다. 이를 통해 게임의 숨겨진 특징을 파악하고 유사한 게임을 정확하게 찾아냅니다.")
        
        create_feature_card("🧩", "콘텐츠 기반 추천", 
            "게임의 장르, 개발사, 태그 등 다양한 메타데이터를 분석하여 사용자의 취향에 맞는 게임을 추천합니다. 정교한 알고리즘을 통해 게임 간의 유사성을 계산합니다.")
        
        create_feature_card("🔀", "앙상블 추천 기법", 
            "NLP 기반 추천과 콘텐츠 기반 추천의 결과를 고급 앙상블 알고리즘을 통해 통합합니다. 이를 통해 각 모델의 장점을 극대화하고 더욱 정확하고 다양한 추천을 제공합니다.")
        
        create_feature_card("🤖", "AI 챗봇 기반 인터랙티브 추천", 
            "최신 대화형 AI 모델을 활용한 챗봇 시스템을 통해 사용자와 자연스러운 대화를 나눕니다. 사용자의 취향, 플레이 스타일, 선호하는 게임 요소 등을 심층적으로 파악하여 맞춤형 게임 추천을 제공합니다. 또한, 게임에 대한 상세한 정보와 인사이트를 실시간으로 제공하여 사용자의 게임 선택을 돕습니다.")
        

    elif selected == "기술 스택":
        st.header("🛠️ 최신 기술 스택")
        
        tech_stack = {
            "Python": "핵심 프로그래밍 언어",
            "Streamlit": "반응형 웹 애플리케이션 프레임워크",
            "NumPy & Pandas": "고성능 데이터 처리 및 분석",
            "Scikit-learn": "머신러닝 알고리즘 및 모델 평가",
            "Hugging Face Transformers": "최신 NLP 모델 활용",
            "LlamaIndex": "대규모 언어 모델 기반 인덱싱 및 쿼리 시스템",
            "PyTorch": "딥러닝 모델 구현 및 학습",
            "Plotly": "인터랙티브 데이터 시각화"
        }
        
        for tech, desc in tech_stack.items():
            st.markdown(f"**{tech}**: {desc}")
        
        st.info("우리 팀은 지속적으로 최신 기술을 도입하여 시스템을 개선하고 있습니다.")

    elif selected == "주요 기능":
        st.header("🔍 혁신적인 주요 기능")
        
        features = [
            ("🎮 게임 이름 기반 정밀 추천", "사용자가 입력한 게임과 가장 유사한 게임들을 AI가 분석하여 추천합니다."),
            ("🏷️ 장르 기반 스마트 추천", "선택한 장르 내에서 최적의 게임을 AI가 선별하여 제안합니다."),
            ("🏢 개발사 기반 큐레이션", "특정 개발사의 게임들 중 사용자 취향에 맞는 게임을 추천합니다."),
            ("🤖 AI 챗봇 기반 인터랙티브 추천", "사용자와의 대화를 통해 취향을 파악하고 맞춤형 게임을 추천합니다."),
            ("📊 데이터 기반 인사이트 제공", "게임 트렌드, 인기 장르 등 다양한 통계 정보를 제공합니다.")
        ]
        
        for title, description in features:
            expander = st.expander(title)
            expander.write(description)

    elif selected == "성능 분석":
        st.header("📊 시스템 성능 분석")
        
        metrics = {
            "추천 정확도": 92.5,
            "사용자 만족도": 4.8,
            "평균 응답 시간": 0.5
        }
        
        col1, col2, col3 = st.columns(3)
        col1.metric("추천 정확도", f"{metrics['추천 정확도']}%", "2.5%")
        col2.metric("사용자 만족도", f"{metrics['사용자 만족도']}/5", "0.3")
        col3.metric("평균 응답 시간", f"{metrics['평균 응답 시간']}초", "-0.1초")
        
        # 성능 그래프
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = metrics['추천 정확도'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "추천 시스템 성능", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'red'},
                    {'range': [50, 80], 'color': 'yellow'},
                    {'range': [80, 100], 'color': 'green'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        st.plotly_chart(fig)

    elif selected == "향후 계획":
        st.header("📈 미래 비전 및 개발 계획")
        
        plans = [
            "개인화된 사용자 프로필 기반 추천 시스템 고도화",
            "실시간 Steam 데이터 연동 및 트렌드 분석 기능 추가",
            "모바일 애플리케이션 버전 출시",
            "다국어 지원 및 글로벌 서비스 확장",
            "VR/AR 게임 추천 기능 도입",
            "게임 커뮤니티 통합 및 소셜 추천 기능 개발"
        ]
        
        for i, plan in enumerate(plans, 1):
            st.write(f"{i}. {plan}")
        
        st.success("우리의 궁극적인 목표는 전 세계 게이머들에게 최고의 게임 경험을 제공하는 것입니다! 🌟")

    st.markdown("---")
    st.markdown("© 2025 Steam 게임 추천 시스템 | 모든 권리 보유")
