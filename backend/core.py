from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Chat Model
model = init_chat_model(model="gemini-3.6-flash", model_provider="google_genai")

# Retriever 
@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    "retrieve relevant documnetation to help answer user queries about Langchain"
    retrieved_docs = vectorstore.as_retriever().invoke(query)

    # Serialize documents of the model 
    # Serialize: Convert Document objects into plain text the LLM can read
    serialized = "\n\n".join(
        (f"Sourse:{doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    # Return formatted text for the LLM (content) and raw docs for citations (artifact)
    return serialized, retrieved_docs

def run_llm(query:str) ->Dict[str,Any]:
    """
    run the RAG pipline to answer the quesry using the ritrieved pipline 
    Args : 
        query: the user's Question 
    Returns:
    Dictionary:{
        - Answer: the generated answer 
        - Context:the retrieved context
        }
    """
    # create the agent with the retrieval tool 

    system_prompt  = "You are a helpful AI assistant that answers questions about LangChain documentation. "
    "You have access to a tool that retrieves relevant documentation. "
    "Use the tool to find relevant information before answering questions. "
    "Always cite the sources you use in your answers. "
    "If you cannot find the answer in the retrieved documentation, say so."

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    #Build message list
    message = [{"role":"user","content":query}]

    #invoke the agent
    response = agent.invoke({"messages":message})

    #Extract the answer from the last llm message 
    final_message = response["messages"][-1]

    if isinstance(final_message.content, str):
        answer = final_message.content
    else:
        answer = "\n".join(
            block["text"] for block in final_message.content if block.get("type") == "text"
        )
    #Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:

        #check if this is a tool message with artifact
        if isinstance(message, ToolMessage) and hasattr(message,"artifact"):            

            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are The agents?")
    print(result)
    