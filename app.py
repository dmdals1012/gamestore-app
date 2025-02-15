import streamlit as st
from streamlit_option_menu import option_menu
from ui import description, home, ml



# 세션 상태 초기화
if 'current_app' not in st.session_state:
    st.session_state.current_app = "홈"

def run():
    with st.sidebar:
        app = option_menu(
            menu_title='Steam 게임 추천 🎮',
            options=['홈', '앱 소개', '게임 추천'],
            icons=['house-fill', 'info-circle-fill', 'joystick'],
            menu_icon='app-indicator',
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )
        st.session_state.current_app = app

    if st.session_state.current_app == "홈":
        home.app()
    elif st.session_state.current_app == "앱 소개":
        description.app()
    elif st.session_state.current_app == "게임 추천":
        ml.app()

if __name__ == '__main__':
    run()
