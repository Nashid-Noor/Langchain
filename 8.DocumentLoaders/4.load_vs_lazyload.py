from langchain_community.document_loaders import PyPDFLoader
import time

loader= PyPDFLoader('./test.pdf')

start= time.time()
docs=loader.load()
for doc in docs:
    print(doc.metadata)
end= time.time()
print("Duration of load function: ", end-start)

start= time.time()
docs=loader.lazy_load()
for doc in docs:
    print(doc.metadata)
end= time.time()
print("Duration of lazy load function: ", end-start)
