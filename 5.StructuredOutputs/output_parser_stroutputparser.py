from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

model= ChatHuggingFace(llm=llm)
template1= PromptTemplate(
        template='Generate a detailed report on {topic}',
        input_variables=['topic']
)

template2= PromptTemplate(
        template='generate a 5 line summary of \n {text}',
        input_variables=['text']
)

'''

prompt1= template1.invoke({'topic':'Black Hole'})
result= model.invoke(prompt1)

prompt2= template2.invoke({'text':result.content})
result= model.invoke(prompt2)

print(result.content)

'''
parser= StrOutputParser
chain = template1 | model | parser | template2 | model | parser
result= chain.invoke({'topic': 'black hole'})

print(result)