from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def count_words(text):
    return len(text.split())

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

prompt=PromptTemplate(
    template='Generate a one liner joke on the below topic \n topic : {topic}',
    input_variables=['topic']
)

seq_chain= RunnableSequence(prompt, model, parser)
parallel_chain= RunnableParallel(
    {
        "joke":RunnablePassthrough(),
        "word_count":RunnableLambda(count_words)
    }
)

chain= RunnableSequence(seq_chain,parallel_chain)
result= chain.invoke({'topic':'trump'})

print("word count of the generated joke is {}".format(result['word_count']))