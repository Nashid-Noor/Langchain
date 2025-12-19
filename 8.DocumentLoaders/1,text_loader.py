from langchain_community.document_loaders import TextLoader

loader = TextLoader('./sample.txt')
docs= loader.load()

print(docs,end='\n')
print(type(docs))
print(len(docs))
print(docs[0])