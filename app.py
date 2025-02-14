import streamlit as st
from streamlit_option_menu import option_menu
from ui import description, home, ml

st.set_page_config(page_title="Steam 게임 추천 시스템", page_icon="🎮", layout="wide")

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
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

        if app == "홈":
            home.app()
        elif app == "앱 소개":
            description.app()
        elif app == "게임 추천":
            ml.app()

if __name__ == '__main__':
    multi_app = MultiApp()
    multi_app.run()
