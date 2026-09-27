\# Backend, core.py



This module is the actual RAG engine behind the Streamlit interface, everything `main.py` displays comes from calling `run\_llm()` in this file.



\## Setup section



```python

embeddings = HuggingFaceEmbeddings(model\_name="sentence-transformers/all-MiniLM-L6-v2")

```

Loads the same local embedding model used during ingestion. This must match exactly, since query vectors and stored vectors need to come from the same model to be meaningfully comparable.



```python

vectorstore = Chroma(persist\_directory="./chroma\_db", embedding\_function=embeddings)

```

Reopens the local vector store `ingestion.py` built, no re-embedding happens here, this just connects to what's already saved on disk.



```python

model = init\_chat\_model(model="gemini-3.6-flash", model\_provider="google\_genai")

```

`init\_chat\_model` is LangChain's provider agnostic model loader, one function that can build a model from many different providers depending on `model\_provider`.



`model\_provider="google\_genai"` specifically matters here, it is not optional or a style choice. LangChain has two separate Google integrations, `google\_genai`, matching a plain `GOOGLE\_API\_KEY` from Google AI Studio, and `google\_vertexai`, matching Google Cloud Platform's enterprise Vertex AI product, which needs a GCP project and service account. If `model\_provider` is left out, LangChain's own automatic guessing defaults any `gemini` prefixed model name to `google\_vertexai`, not `google\_genai`, which fails immediately for anyone using a normal API key setup like ours. This is a well documented, genuinely common trap.



\## The retrieval tool



```python

@tool(response\_format="content\_and\_artifact")

def retrieve\_context(query: str):

&#x20;   "retrieve relevant documentation to help answer user queries about Langchain"

&#x20;   retrieved\_docs = vectorstore.as\_retriever().invoke(query)



&#x20;   serialized = "\\n\\n".join(

&#x20;       (f"Source:{doc.metadata.get('source', 'Unknown')}\\n\\nContent: {doc.page\_content}")

&#x20;       for doc in retrieved\_docs

&#x20;   )

&#x20;   return serialized, retrieved\_docs

```



\*\*`@tool(response\_format="content\_and\_artifact")`\*\* turns this function into something the agent can call, same `tool` decorator seen in earlier branches, but this specific parameter is new. Normally a tool returns one single value back to the model. `response\_format="content\_and\_artifact"` tells LangChain this tool instead returns a tuple of two different things, the `content`, plain text the model actually reads and reasons over, and the `artifact`, extra structured data that gets attached to the resulting `ToolMessage` but is \*\*not\*\* shown to the model directly. This exists specifically so a tool can hand back rich data, here the full `Document` objects with their metadata, for your own code to use later, citations, source links, without cluttering what the model itself sees, which only needs the readable text.



\*\*The docstring\*\*, `"retrieve relevant documentation to help answer user queries about Langchain"`, is not a comment, it is the actual description the agent reads to decide whether and when to call this tool, exactly like every `@tool` function you've built since the search agent branch.



\*\*`vectorstore.as\_retriever().invoke(query)`\*\* performs the actual retrieval, embeds the query, finds the closest stored chunks, returns them as a list of `Document` objects.



\*\*The `serialized` string\*\* is the `content` half of the return value, built by looping through each retrieved document and formatting it as `Source: <url>` followed by `Content: <text>`, joined with blank lines between each one. This readable, source-labeled text is exactly what the model sees and can quote from or cite.



\*\*`return serialized, retrieved\_docs`\*\* returns both halves as a tuple, `serialized` becomes the `content` the model reads, `retrieved\_docs`, the original list of full `Document` objects, becomes the `artifact`, preserved separately for your own code to pull citations from afterward.



\## run\_llm, the actual orchestration function



```python

def run\_llm(query: str) -> Dict\[str, Any]:

```

Takes a user's question, returns a dictionary with two keys, `answer`, the final text response, and `context`, the list of documents actually used to produce it.



```python

system\_prompt = "You are a helpful AI assistant..."

```

Sets the agent's behavior and constraints, told specifically to use the retrieval tool before answering, and to cite sources, this shapes how the model decides when to call `retrieve\_context` at all.



```python

agent = create\_agent(model, tools=\[retrieve\_context], system\_prompt=system\_prompt)

```

Builds the agent, same `create\_agent` interface from the search agent branch, given the model, the one available tool, and the system prompt.



```python

message = \[{"role": "user", "content": query}]

response = agent.invoke({"messages": message})

```

Builds the message list and invokes the agent, exactly the `{"messages": \[...]}` input shape every `create\_agent` call needs, confirmed as a recurring pattern across this whole course.



```python

answer = response\["messages"]\[-1].content

```

Pulls the last message from the full conversation, the agent's final answer after any tool calls, exactly the same pattern used in the search agent branch's fix.



```python

context\_docs = \[]

for message in response\["messages"]:

&#x20;   if isinstance(message, ToolMessage) and hasattr(message, "artifact"):

&#x20;       if isinstance(message.artifact, list):

&#x20;           context\_docs.extend(message.artifact)

```

This loop walks through every message in the full conversation looking specifically for `ToolMessage` instances, messages representing a tool's result, and checks whether each one has an `artifact` attached. Recall from `retrieve\_context` above, `artifact` is exactly where the raw `Document` objects were stashed. This loop is what pulls those original documents back out, so the calling code, `main.py`'s Streamlit interface, can display real source citations to the user, not just the plain answer text.



```python

return {"answer": answer, "context": context\_docs}

```

The final structured result, ready for `main.py` to display, `answer` for the chat bubble text, `context` for a sources or citations section.



\## Known fixes applied in this version



`model\_provider` corrected from `google\_vertexai` to `google\_genai`, matching our actual `GOOGLE\_API\_KEY` setup rather than requiring Google Cloud Platform credentials we don't have.



`response\["message"]` corrected to `response\["messages"]` in two places, matching the actual key `create\_agent` returns, plural, matching the `{"messages": \[...]}` input shape.

