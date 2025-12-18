from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableSequence,RunnableParallel
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
    template='draft me a tweet about {topic}',
    input_variables=['topic']
)
prompt2= PromptTemplate(
    template='draft me a linkedIn post about {topic}',
    input_variables=['topic']
)

parallel_chain= RunnableParallel({
    "tweet":RunnableSequence(prompt1, model, parser),
    "post":RunnableSequence(prompt2,model,parser)
})

result= parallel_chain.invoke({"topic":"trump"})
print("Result ",result)
print("Tweet result", result["tweet"])
print("post result", result["post"])