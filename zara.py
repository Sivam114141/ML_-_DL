from langchain_ollama import ChatOllama
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.storage import InMemoryStore
from langchain_core.messages import AIMessage, HumanMessage

# Initialize the Ollama model
llm = ChatOllama(model="mistral")

# Store conversation history in memory
memory_store = InMemoryStore()
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="history", return_messages=True)

# Define a chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant."),
    ("human", "{input}"),
])

# Function to retrieve session history
def get_session_history(session_id: str):
    history = memory_store.mget([session_id])
    if history and history[0] is not None:
        return [HumanMessage(content=msg) if isinstance(msg, str) else msg for msg in history[0]]
    return []  # Return an empty list if no history exists

# Create a runnable with memory for handling history
conversation = RunnableWithMessageHistory(
    runnable=llm,
    get_session_history=get_session_history,  # Required argument
    input_key="input",
)

# Example: User input
user_input = "Hello! Remember my name is Sivam."

# Generate response
response = conversation.invoke(
    {"input": user_input}, 
    config={"configurable": {"session_id": "user123"}}
)

# Print AI response
print("AI Response:", response)
