from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
chat_model = ChatHuggingFace(llm=llm)

chat_history=[
    SystemMessage(content='You are a helpful assistant'),
]

while True:
    user_input= input('You: ')
    chat_history.append(HumanMessage(user_input))
    if user_input=='exit':
        break
    result= chat_model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print('AI: ',result.content)