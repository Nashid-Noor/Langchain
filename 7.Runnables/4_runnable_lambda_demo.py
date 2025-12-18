from langchain_core.runnables import RunnableLambda
def word_count(text):
    return len(text.split())



runnable_word_counter= RunnableLambda(word_count)
print(runnable_word_counter.invoke('hello how are you'))