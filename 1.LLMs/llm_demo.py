from langchain_openai import OpenAI
from dotenv import load_dotenv  

load_dotenv() # Load environment variables from .env file

llm= OpenAI(model="gpt-3.5-turbo-instruct")

result= llm.invoke("What is the capital of India?") # This invoke method is used to invoke the model with the given input

print(result)