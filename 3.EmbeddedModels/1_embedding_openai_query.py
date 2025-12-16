from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

text = "Hello, how are you?"

embedding = embeddings.embed_query(text)

print(embedding)