from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
parser =JsonOutputParser()

template= PromptTemplate(
        template='give me name, age, and gender of a fictional person {format_instructions}',
        input_variables=['topic'],
        partial_variables={'format_instructions':parser.get_format_instructions()}
)
prompt= template.format()

result= model.invoke(prompt)
print(result.content)