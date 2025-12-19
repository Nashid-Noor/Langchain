from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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

url='https://www.flipkart.com/apple-iphone-17-pro-cosmic-orange-512-gb/p/itm999d978f08430?pid=MOBHFN6YUW9A93DC&lid=LSTMOBHFN6YUW9A93DCUFF5GV&marketplace=FLIPKART&q=apple+17+pro&store=tyy%2F4io&srno=s_1_2&otracker=AS_QueryStore_OrganicAutoSuggest_1_5_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_5_na_na_na&fm=organic&iid=bf2408b1-54c2-4ced-afbc-040f1f5fa5ab.MOBHFN6YUW9A93DC.SEARCH&ppt=hp&ppn=homepage&ssid=zb87s6tjpoaumfwg1766154377126&qH=9952c08834508a43'
loader=WebBaseLoader(url)
docs=loader.load()

prompt= PromptTemplate(
    template="what is this {link} about?",
    input_variables=['link']
)

chain= prompt | model | parser
result=chain.invoke({"link":docs[0].page_content})

print(result)
