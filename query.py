from model import LLMModel
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from message import SYSTEM_MESSAGE
#import questions

# Load Document
documents = SimpleDirectoryReader(
    input_files = ["./documents/samples.txt"]
).load_data()
documents = Document(text = "\n\n".join([doc.text for doc in documents]))

# Model path
model_path = "./models/mistral-7b-instruct-v0.2.Q5_K_S.gguf"

# Initialize Chatbot
chatbot = LLMModel(model_path)
#chatbot = LLMModel()

# Build Vecotor Index
vector_index = chatbot.get_build_index(documents, "./vector_store/index")

# Setup Query Engine
query_engine = chatbot.get_query_engine(vector_index)

"""
# Select Question set
version = input("Enter Question set : ")
# Check if the entered question set exists in the questions module
if hasattr(questions, version):
    ques = getattr(questions, version)
else:
    print("Question set not found.")
questions_length = len(ques)
print("number of questions: ", questions_length)

for i in range(0, questions_length):
    dict = ques[i]
    query= dict['query']
    print(query)
    merged_query = f"{SYSTEM_MESSAGE}\nQuery: {query}"
    response = query_engine.query(merged_query)
    print(response)
    print("\n")

"""

# Query the model
while True:
    query=input("Ask your Query: ")
    merged_query = f"{SYSTEM_MESSAGE}\nQuery: {query}"
    response = query_engine.query(merged_query)
    print(response)
    print("\n")
