from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()



llm = HuggingFacePipeline.from_model_id(
    model_id="deepseek-ai/deepseek-vl2-tiny",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        temperature=0.7,

    ),
)

chat_model = ChatHuggingFace(llm=llm)
result= chat_model.invoke("What is the capital of India?")
print(result.content)