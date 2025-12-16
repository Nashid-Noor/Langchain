from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents=[
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the God of Cricket , holds many batting records.",
    "Rohit Sharma is knownfor his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.",
    "Ravindra Jadeja is a versatile all-rounder known for his bowling and batting.",
    "Yuzvendra Chahal is a leg-spinner known for his googly and spin variations.",
    "Ravichandran Ashwin is a spin all-rounder known for his off-spin and batting.",
    "Hardik Pandya is a versatile all-rounder known for his batting and bowling.",
    "Bhuvneshwar Kumar is a fast bowler known for his swing and seam movement.",
    "Kuldeep Yadav is a leg-spinner known for his googly and spin variations.",
    "Mohammed Shami is a fast bowler known for his swing and seam movement.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."

]
query="Tell me about Sachin"
doc_embedding = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

cosine_similarity = cosine_similarity([query_embedding], doc_embedding)
index,scores= sorted(enumerate(cosine_similarity[0]), key=lambda x: x[1], reverse=True)[0]
print(documents[index])
print("score",scores)

