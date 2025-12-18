from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableSequence,RunnableBranch,RunnablePassthrough,RunnableLambda
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
    template='Generate few jokes on the below topic \n topic : {topic}',
    input_variables=['topic']
)

seq_chain= RunnableSequence(prompt, model, parser)
branch_chain= RunnableBranch(
    (lambda x: len(x.split())>1000000, RunnablePassthrough() ),
    RunnableLambda(count_words)
)

final_chain= seq_chain | branch_chain
print(final_chain.invoke({'topic':'trump'}))
