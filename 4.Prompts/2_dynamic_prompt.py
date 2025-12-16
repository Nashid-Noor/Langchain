from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt, PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Research Tool')

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
chat_model = ChatHuggingFace(llm=llm)

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical",
"Code-Oriented", "Mathematical"] )
length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation) "])


######### This template would be used in dynamic_prompt_generator.py file  #########

# template= PromptTemplate(
#     template="""
#     Please summarize the research paper titled "{paper_input}" with the following specifications:
#     Explanation Style: {style_input}
#     Explanation Length: {length_input}
#     1. Mathematical Details:
#         - Include relevant mathematical equations if present in the paper.
#         - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
#     2. Analogies:
#         - Use relatable analogies to simplify complex ideas.
#     If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
#     Ensure the summary is clear, accurate, and aligned with the provided style and length.
#     """,
#     input_variables=['paper_input','style_input','length_input']
# )


template= load_prompt('4.Prompts/2_template.json')
prompt= template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input,
})

if st.button("summarize"):
    result= chat_model.invoke(prompt)
    st.write(result.content)
