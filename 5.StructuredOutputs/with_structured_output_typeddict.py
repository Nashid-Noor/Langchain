from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

class Review(TypedDict):
    summary: str
    sentiment:Annotated[Literal['pos','neg'],"return the sentiment of the review as negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down cons inside a list"]

chat_model = ChatHuggingFace(llm=llm)
#print(chat_model.invoke("capital of Delhi?").content)

structured_output=chat_model.with_structured_output(Review)
structured_output_result= structured_output.invoke(
"""
You are a strict JSON generator.
Return ONLY valid JSON (no markdown, no extra text).
Schema:
{
  "summary": string,
  "sentiment": "pos" | "neg",
  "pros": [string] | null,
  "cons": [string] | null
}

Review the movie "Kantha".
""")
print(structured_output_result)
