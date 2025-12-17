from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser= StrOutputParser()


llm1 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
model1= ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
model2= ChatHuggingFace(llm=llm2)


prompt1= PromptTemplate(
    template="Generate short notes on the following \n {text}",
    input_variables=["text"]
)

prompt2= PromptTemplate(
    template="generate 5 question and answers from the following \n {text}",
    input_variables={"text"}
)

prompt3= PromptTemplate(
    template="Merge the provided notes and quiz onto a single document \n notes -> {notes} quiz -> {quiz}",
    input_variables=["notes","quiz"]
)

parallel_chain= RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model1 | parser 
    }
)

sequential_chain=  prompt3 | model1 | parser 

merge_chain = parallel_chain | sequential_chain

text = """ 
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.
Still effective in cases where number of dimensions is greater than the number of samples.
Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result= merge_chain.invoke({"text":text})
print(result)