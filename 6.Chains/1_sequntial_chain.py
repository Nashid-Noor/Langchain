from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
model= ChatHuggingFace(llm=llm)
parser= StrOutputParser()

prompt1= PromptTemplate(
    template="generate a report about {topic}",
    input_variables=['topic'] 
)
prompt2= PromptTemplate(
    template="generate a 5 points summary about {text}",
    input_variables=['text']
)

chain= prompt1 | model | parser | prompt2 | model | parser
result=chain.invoke({'topic':'langchain'}) 

print(result)