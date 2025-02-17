import streamlit as st
from streamlit_option_menu import option_menu
from ui import description, home
import requests

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 세션 상태 초기화
if 'current_app' not in st.session_state:
    st.session_state.current_app = "홈"
if 'show_chatbot' not in st.session_state:
    st.session_state['show_chatbot'] = False
if 'search_method' not in st.session_state:
    st.session_state['search_method'] = None

def run():
    st.set_page_config(
        page_title="Steam 게임 추천",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 챗봇 UI 스타일
    st.markdown(
        """
        <style>
        #chatbot-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: #3498db;
            color: white;
            font-size: 24px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            z-index: 1000;
        }
        .chat-popup {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 300px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_xyadoh9h.json"
    lottie_json = load_lottie_url(lottie_url)

    menu = [
        "홈",
        "앱 소개",
        "게임 추천"
    ]
    icons = ["house-fill", "info-circle-fill", "joystick"]

    with st.sidebar:
        selected_menu = option_menu(
            menu_title="메뉴",
            options=menu,
            icons=icons,
            menu_icon="app-indicator",
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )
        st.session_state.current_app = selected_menu

    if st.session_state.current_app == "홈":
        home.app()
    elif st.session_state.current_app == "앱 소개":
        description.app()
    elif st.session_state.current_app == "게임 추천":
        from ui import ml  # ml 모듈을 여기서 임포트
        search_method = ml.app()
        st.session_state['search_method'] = search_method

    # 챗봇 버튼 (항상 표시)
    st.markdown('<button id="chatbot-button" onclick="toggleChatbot()">💬</button>', unsafe_allow_html=True)

    # 챗봇 팝업
    if st.session_state.get('show_chatbot', False):
        with st.container():
            st.markdown('<div class="chat-popup">', unsafe_allow_html=True)
            st.write("추천 받은 게임이 궁금하시다면?")
            user_input = st.text_input("질문을 입력하세요:", key="chatbot_input")
            if user_input:
                with st.spinner("AI가 답변을 생성 중입니다... 🤖"):
                    ml.initialize_models()
                    ml.index = ml.get_index_from_huggingface()
                    query_engine = ml.index.as_query_engine()
                    response = query_engine.query(user_input)
                    st.write(response.response)
            st.markdown('</div>', unsafe_allow_html=True)

    # JavaScript for toggling chatbot
    st.markdown("""
    <script>
    function toggleChatbot() {
        const chatbotState = window.parent.getStApp().state.show_chatbot;
        window.parent.getStApp().state.show_chatbot = !chatbotState;
        window.parent.getStApp().forceRerun();
    }
    </script>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    run()
