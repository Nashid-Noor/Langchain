from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = ["Hello, how are you?", "I am fine, thank you!"]

embedding = embeddings.embed_documents(documents)

print(embedding)