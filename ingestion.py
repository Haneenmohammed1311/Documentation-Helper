import asyncio
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

load_dotenv()

# Configure SSL context to use certifi certificates, avoids SSL
# verification errors on some Windows setups.
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 500):
    """Process documents in batches asynchronously."""
    print("\n" + "*" * 37)
    print("VECTOR STORAGE PHASE")
    print("*" * 37)
    print(f"Preparing to add {len(documents)} documents to vector store")

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    print(f"Split into {len(batches)} batches of {batch_size} documents each")

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            print(f"  Batch {batch_num}/{len(batches)} added successfully ({len(batch)} documents)")
            return True
        except Exception as e:
            print(f"  Batch {batch_num} FAILED - {e}")
            return False

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        print(f"All batches processed successfully! ({successful}/{len(batches)})")
    else:
        print(f"WARNING: only {successful}/{len(batches)} batches succeeded")


async def main():
    """Main async function to orchestrate the entire process."""
    print("*" * 37)
    print("DOCUMENTATION INGESTION PIPELINE")
    print("*" * 37)

    print("\nStarting to crawl the documentation site...")

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        print(f"  Crawled: {tavily_crawl_result_item['url']}")
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    print("\n" + "*" * 37)
    print("DOCUMENT CHUNKING PHASE")
    print("*" * 37)
    print(f"Processing {len(all_docs)} documents with chunk size 4000, overlap 200")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    print(f"Created {len(splitted_docs)} chunks from {len(all_docs)} documents")

    await index_documents_async(splitted_docs, batch_size=500)

    print("\n" + "*" * 37)
    print("PIPELINE COMPLETE")
    print("*" * 37)
    print(f"Documents extracted : {len(all_docs)}")
    print(f"Chunks created       : {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())