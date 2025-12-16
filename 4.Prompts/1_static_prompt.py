from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research Tool')
user_input= st.text_input('Enter your prompt')

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)


chat_model = ChatHuggingFace(llm=llm)

if st.button('Summarize'):
    result= chat_model.invoke(user_input)
    print(result)
    st.write(result.content)




