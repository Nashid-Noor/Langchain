from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser= StrOutputParser()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
model= ChatHuggingFace(llm=llm)

prompt1= PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)
prompt2= PromptTemplate(
    template='explain this joke \n {text}',
    input_variables=['text']
)


chain= RunnableSequence(prompt1, model, parser, prompt2, model ,parser)
result= chain.invoke({'topic':'Trump'})

print(result)