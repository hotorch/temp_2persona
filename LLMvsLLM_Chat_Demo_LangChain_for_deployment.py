import streamlit as st

import anthropic
import openai

import datetime
from tqdm.notebook import tqdm
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import pandas as pd

import os
import io

# from dotenv import load_dotenv


# # 환경 변수 로드
# load_dotenv()

# # API 키 설정
# openai.api_key = os.getenv("OPENAI_API_KEY")
# anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")


def get_new_ai_chains(ai1_name, ai1_system_prompt, ai2_name, ai2_system_prompt, model1_selection, model2_selection, ai1_temperature, ai2_temperature):
    if model1_selection == "gpt-3.5-turbo":
        model1 = ChatOpenAI(model_name=model1_selection, temperature=ai1_temperature, openai_api_key=st.session_state.openai_api_key)  
    else:
        model1 = ChatAnthropic(model=model1_selection, temperature=ai1_temperature, anthropic_api_key=st.session_state.anthropic_api_key)  
        
    if model2_selection == "gpt-3.5-turbo":  
        model2 = ChatOpenAI(model_name=model2_selection, temperature=ai2_temperature, openai_api_key=st.session_state.openai_api_key)
    else:
        model2 = ChatAnthropic(model=model2_selection, temperature=ai2_temperature, anthropic_api_key=st.session_state.anthropic_api_key)

    ai_1_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ai1_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),  
            ("human", "{input}"),
        ]
    )
    ai_1_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    ai_1_chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(ai_1_memory.load_memory_variables) | itemgetter("chat_history")  
        )
        | ai_1_prompt
        | model1  
    )


    ai_2_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ai2_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    ai_2_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    ai_2_chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(ai_2_memory.load_memory_variables) | itemgetter("chat_history")
        )
        | ai_2_prompt
        | model2
    )


    return ai_1_chain, ai_1_memory, ai_2_chain, ai_2_memory



def setup_page():
    st.title("🤖🆚🤖 Conversation Setup")

    st.subheader("API 키 설정")
    openai_api_key = st.text_input("OpenAI API 키", type="password")
    anthropic_api_key = st.text_input("Anthropic API 키", type="password")

    st.subheader("모델 선택")
    model1_selection = st.selectbox("모델 1", ["gpt-3.5-turbo", "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"])
    model2_selection = st.selectbox("모델 2", ["gpt-3.5-turbo", "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"])


    st.subheader("AI 1 설정")
    ai1_name = st.text_input("이름", placeholder="가상의 이름을 입력해주세요", key="ai1_name_input")
    ai1_role = st.text_input("역할", placeholder="페르소나를 입력해주세요", key="ai1_role_input")
    ai1_situation = st.text_input("상황", placeholder="대화 상황을 이야기해주세요", key="ai1_situation_input")
    ai1_goal = st.text_input("목표", placeholder="대화의 목적을 이야기해주세요", key="ai1_goal_input")
    ai1_tone = st.text_area("어조", placeholder="자연어 형태로 이야기 해주세요", key="ai1_tone_input")
    ai1_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="ai1_temperature_slider")

    st.subheader("AI 2 설정")
    ai2_name = st.text_input("이름", placeholder="가상의 이름을 입력해주세요", key="ai2_name_input")
    ai2_role = st.text_input("역할", placeholder="페르소나를 입력해주세요", key="ai2_role_input")
    ai2_situation = st.text_input("상황", placeholder="대화 상황을 이야기해주세요", key="ai2_situation_input")
    ai2_goal = st.text_input("목표", placeholder="대화의 목적을 이야기해주세요", key="ai2_goal_input")
    ai2_tone = st.text_area("어조", placeholder="자연어 형태로 이야기 해주세요", key="ai2_tone_input")
    ai2_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="ai2_temperature_slider")
    
    # AI1 시스템 프롬프트 
    ai1_system_prompt = f"""
    Your name is {ai1_name}. You are {ai1_role}.
    ## Situation
    - {ai1_situation}.
    ## Goal  
    - {ai1_goal}.
    ## Tone
    - {ai1_tone}.
    """

    # AI2 시스템 프롬프트
    ai2_system_prompt = f"""
    Your name is {ai2_name}. You are {ai2_role}. 
    ## Situation
    - {ai2_situation}.
    ## Goal
    - {ai2_goal}.
    ## Tone
    - {ai2_tone}.
    """

    st.subheader("대화 설정")
    num_turns = st.number_input("대화 턴 수", min_value=1, max_value=20, value=3, step=1)
    num_max_turns = st.number_input("전체 대화 수", min_value=1, max_value=5, value=2, step=1)

    if st.button("설정 완료"):
        st.session_state.openai_api_key = openai_api_key
        st.session_state.anthropic_api_key = anthropic_api_key
        st.session_state.model1_selection = model1_selection
        st.session_state.model2_selection = model2_selection
        st.session_state.ai1_name_value = ai1_name
        st.session_state.ai1_role_value = ai1_role
        st.session_state.ai1_situation_value = ai1_situation
        st.session_state.ai1_goal_value = ai1_goal
        st.session_state.ai1_tone_value = ai1_tone
        st.session_state.ai1_temperature_value = ai1_temperature
        st.session_state.ai2_name_value = ai2_name
        st.session_state.ai2_role_value = ai2_role
        st.session_state.ai2_situation_value = ai2_situation
        st.session_state.ai2_goal_value = ai2_goal
        st.session_state.ai2_tone_value = ai2_tone
        st.session_state.ai2_temperature_value = ai2_temperature
        st.session_state.num_turns = num_turns
        st.session_state.num_max_turns = num_max_turns
        st.session_state.ai1_system_prompt = ai1_system_prompt
        st.session_state.ai2_system_prompt = ai2_system_prompt
        
        st.session_state.page = "conversation"
        st.rerun()
    

def conversation_page():
    st.title("🤖🆚🤖 Conversation")

    conversation_container = st.container()
    conversation_list = []

    n_conversation = st.session_state.num_turns  # 대화 턴 수
    n_max_turn = st.session_state.num_max_turns  # 전체 대화 수

    for i in range(n_max_turn):
        ai_1_chain, ai_1_memory, ai_2_chain, ai_2_memory = get_new_ai_chains(
            st.session_state.ai1_name_value,
            st.session_state.ai1_system_prompt,
            st.session_state.ai2_name_value,
            st.session_state.ai2_system_prompt,
            st.session_state.model1_selection,
            st.session_state.model2_selection,
            st.session_state.ai1_temperature_value,
            st.session_state.ai2_temperature_value
        )

        ## 초기 Prompts Shooting
        ai2_initial_prompt = {
            "input": f"""
            Start the conversation with a very short greeting in a tone of {st.session_state.ai1_tone_value} appropriate for the {st.session_state.ai1_situation_value}.
            """
        }
        print(ai2_initial_prompt)
        ai_2_output = ai_2_chain.invoke(ai2_initial_prompt).content

        conversation = []
        with conversation_container:
            st.write(f" 🤖🆚🤖 Conversation {i+1} Starts 🔥🔥🔥 \n\n --- \n\n ")
            for j in range(n_conversation):
                ai_1_message_container = st.empty()
                ai_1_output = ai_1_chain.invoke({"input": ai_2_output}).content
                ai_1_memory.save_context({"input": ai_2_output}, {"output": ai_1_output})
                ai_1_message_container.write(f" * 🤖 (AI 1) {st.session_state.ai1_name_value}: {ai_1_output} \n --- \n")
                conversation.append((st.session_state.ai1_name_value, ai_1_output))

                ai_2_message_container = st.empty()
                ai_2_output = ai_2_chain.invoke({"input": ai_1_output}).content
                ai_2_memory.save_context({"input": ai_1_output}, {"output": ai_2_output})
                ai_2_message_container.write(f" * 🤖 (AI 2) {st.session_state.ai2_name_value}: {ai_2_output} \n --- \n")
                conversation.append((st.session_state.ai2_name_value, ai_2_output))

        conversation_list.append(conversation)
    
    # # 대화 저장 버튼
    # if st.button("대화 저장하기"):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    file_name = f"conversation_{timestamp}.xlsx"

    # 대화 내용을 데이터프레임으로 변환
    conversation_data = []
    for i, conversation in enumerate(conversation_list):
        for ai, message in conversation:
            conversation_data.append({"Conversation": i+1, "AI": ai, "Message": message})
    df = pd.DataFrame(conversation_data)

    # 파일로 저장
    excel_data = io.BytesIO()
    writer = pd.ExcelWriter(excel_data, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    excel_data.seek(0)

    # 파일 다운로드 버튼 
    st.download_button(
        label="다운로드",
        data=excel_data,
        file_name=file_name,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # 대화 종료 버튼
    if st.button("대화 종료"):     
        st.session_state.page = "setup"
        st.experimental_rerun()


# 메인 함수
def main():
    if "page" not in st.session_state:
        st.session_state.page = "setup"

    if st.session_state.page == "setup":
        setup_page()
    elif st.session_state.page == "conversation":
        conversation_page()

if __name__ == "__main__":
    main()