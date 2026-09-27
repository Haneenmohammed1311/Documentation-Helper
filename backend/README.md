# Backend, core.py

This module is the actual RAG engine behind the Streamlit interface, everything `main.py` displays comes from calling `run_llm()` in this file.

## Setup section

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
Loads the same local embedding model used during ingestion. This must match exactly, since query vectors and stored vectors need to come from the same model to be meaningfully comparable.

```python
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
```
Reopens the local vector store `ingestion.py` built, no re-embedding happens here, this just connects to what's already saved on disk.

```python
model = init_chat_model(model="gemini-3.6-flash", model_provider="google_genai")
```
`init_chat_model` is LangChain's provider agnostic model loader, one function that can build a model from many different providers depending on `model_provider`.

`model_provider="google_genai"` specifically matters here, it is not optional or a style choice. LangChain has two separate Google integrations, `google_genai`, matching a plain `GOOGLE_API_KEY` from Google AI Studio, and `google_vertexai`, matching Google Cloud Platform's enterprise Vertex AI product, which needs a GCP project and service account. If `model_provider` is left out, LangChain's own automatic guessing defaults any `gemini` prefixed model name to `google_vertexai`, not `google_genai`, which fails immediately for anyone using a normal API key setup like ours. This is a well documented, genuinely common trap.

## The retrieval tool

```python
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    "retrieve relevant documentation to help answer user queries about Langchain"
    retrieved_docs = vectorstore.as_retriever().invoke(query)

    serialized = "\n\n".join(
        (f"Source:{doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
```

**`@tool(response_format="content_and_artifact")`** turns this function into something the agent can call, same `tool` decorator seen in earlier branches, but this specific parameter is new. Normally a tool returns one single value back to the model. `response_format="content_and_artifact"` tells LangChain this tool instead returns a tuple of two different things, the `content`, plain text the model actually reads and reasons over, and the `artifact`, extra structured data that gets attached to the resulting `ToolMessage` but is **not** shown to the model directly. This exists specifically so a tool can hand back rich data, here the full `Document` objects with their metadata, for your own code to use later, citations, source links, without cluttering what the model itself sees, which only needs the readable text.

**The docstring**, `"retrieve relevant documentation to help answer user queries about Langchain"`, is not a comment, it is the actual description the agent reads to decide whether and when to call this tool, exactly like every `@tool` function you've built since the search agent branch.

**`vectorstore.as_retriever().invoke(query)`** performs the actual retrieval, embeds the query, finds the closest stored chunks, returns them as a list of `Document` objects.

**The `serialized` string** is the `content` half of the return value, built by looping through each retrieved document and formatting it as `Source: <url>` followed by `Content: <text>`, joined with blank lines between each one. This readable, source-labeled text is exactly what the model sees and can quote from or cite.

**`return serialized, retrieved_docs`** returns both halves as a tuple, `serialized` becomes the `content` the model reads, `retrieved_docs`, the original list of full `Document` objects, becomes the `artifact`, preserved separately for your own code to pull citations from afterward.

## run_llm, the actual orchestration function

```python
def run_llm(query: str) -> Dict[str, Any]:
```
Takes a user's question, returns a dictionary with two keys, `answer`, the final text response, and `context`, the list of documents actually used to produce it.

```python
system_prompt = "You are a helpful AI assistant..."
```
Sets the agent's behavior and constraints, told specifically to use the retrieval tool before answering, and to cite sources, this shapes how the model decides when to call `retrieve_context` at all.

```python
agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
```
Builds the agent, same `create_agent` interface from the search agent branch, given the model, the one available tool, and the system prompt.

```python
message = [{"role": "user", "content": query}]
response = agent.invoke({"messages": message})
```
Builds the message list and invokes the agent, exactly the `{"messages": [...]}` input shape every `create_agent` call needs, confirmed as a recurring pattern across this whole course.

```python
answer = response["messages"][-1].content
```
Pulls the last message from the full conversation, the agent's final answer after any tool calls, exactly the same pattern used in the search agent branch's fix.

```python
context_docs = []
for message in response["messages"]:
    if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
        if isinstance(message.artifact, list):
            context_docs.extend(message.artifact)
```
This loop walks through every message in the full conversation looking specifically for `ToolMessage` instances, messages representing a tool's result, and checks whether each one has an `artifact` attached. Recall from `retrieve_context` above, `artifact` is exactly where the raw `Document` objects were stashed. This loop is what pulls those original documents back out, so the calling code, `main.py`'s Streamlit interface, can display real source citations to the user, not just the plain answer text.

```python
return {"answer": answer, "context": context_docs}
```
The final structured result, ready for `main.py` to display, `answer` for the chat bubble text, `context` for a sources or citations section.

## Known fixes applied in this version

`model_provider` corrected from `google_vertexai` to `google_genai`, matching our actual `GOOGLE_API_KEY` setup rather than requiring Google Cloud Platform credentials we don't have.

`response["message"]` corrected to `response["messages"]` in two places, matching the actual key `create_agent` returns, plural, matching the `{"messages": [...]}` input shape.