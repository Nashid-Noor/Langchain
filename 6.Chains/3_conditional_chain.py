from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()



class Feedback(BaseModel):
    sentiment: Literal["positive","negative"] = Field(description="Classify the sentiment as either postivie or negative")

str_parser= StrOutputParser()
pydantic_parser= PydanticOutputParser(pydantic_object=Feedback)

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
    template="classify the sentiment of the given feedback into positive or negative \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions":pydantic_parser.get_format_instructions()}
)
prompt2=PromptTemplate(
    template="Give a positive response for this \n {feedback}",
    input_variables=["feedback"]
)
prompt3=PromptTemplate(
    template="give a negative response for this \n {feedback}",
    input_variables=["feedback"]
)



classifier_chain= prompt1 | model | pydantic_parser
# print(classifier_chain.invoke({"feedback":"this is good phone"}))
branch_chain =RunnableBranch(
    (lambda x: x.sentiment =="positive", prompt2 | model | str_parser ),
    (lambda x:x.sentiment == "negative", prompt3 | model | str_parser),
    RunnableLambda(lambda x: "no feedback")
)

chain = classifier_chain | branch_chain
result= chain.invoke({"feedback":"This is a great phone"})

print(result)