from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
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

prompt1=PromptTemplate(
    template='Generate a one liner joke on the below topic \n topic : {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='give a one liner explanation about the below text \n  text : {text}',
    input_variables=['text']
)

seq_chain= RunnableSequence(prompt1, model, parser)
parallel_chain= RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation":RunnableSequence(prompt2, model,parser)
})

final_chain= RunnableSequence(seq_chain,parallel_chain)
result=final_chain.invoke({"topic":"trump"})
print(result)

