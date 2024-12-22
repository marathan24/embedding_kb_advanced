# test_embedding_kb.py
import logging
import json
import random
import PyPDF2
import psycopg2
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from naptha_sdk.client.node import Node
from test_embedding_kb.schemas import InputSchema
from test_embedding_kb.embedder import MemoryEmbedder, RecursiveTextSplitter

logger = logging.getLogger(__name__)

file_path = Path(__file__).resolve()
init_data_path = file_path.parent / "data"

class TestEmbeddingKB:
    def __init__(self, kb_run: Dict[str, Any]):
        self.kb_run = kb_run
        self.kb_deployment = kb_run.kb_deployment
        self.kb_node_url = self.kb_deployment.kb_node_url
        self.kb_config = self.kb_deployment.kb_config

        # Database connection parameters
        self.db_params = {
            'dbname': 'naptha',
            'user': 'naptha',
            'password': 'napthapassword',
            'host': 'localhost',
            'port': '3002'
        }

        if isinstance(self.kb_run.inputs, dict):
            self.input_schema = InputSchema(**self.kb_run.inputs)
        else:
            self.input_schema = InputSchema.model_validate(self.kb_run.inputs)

        self.mode = self.input_schema.mode
        self.query = self.input_schema.query
        
        # Initialize embedder and splitter
        self.embedder = MemoryEmbedder(model=self.kb_config['embedder']['model'])
        self.splitter = RecursiveTextSplitter(
            chunk_size=self.kb_config['embedder']['chunk_size'],
            chunk_overlap=self.kb_config['embedder']['chunk_overlap'],
            separators=self.kb_config['embedder']['separators']
        )

    def read_pdf(self) -> str:
        """Read a PDF file and return its text content"""
        # glob all pdfs in the data folder from pathlib
        pdf_files = list(init_data_path.glob("*.pdf"))
        pdfs = []
        for pdf_file in pdf_files:
            text = ""
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            pdfs.append(text)
        return pdfs

    async def process_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Process text into chunks and create embeddings"""
        chunks = self.splitter.split_text(text)
        embeddings = self.embedder.embed_batch(chunks)
        
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                "id": random.randint(1, 1000000),
                "text": chunk,
                "embedding": embedding,
                "metadata": metadata or {},
            }
            documents.append(doc)
        return documents

    async def run(self, *args, **kwargs):
        if self.mode == "init":
            return await self.init()
        elif self.mode == "query":
            return await self.run_query()
        elif self.mode == "add_data":
            return await self.add_data()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
    async def init(self, *args, **kwargs):
        table_name = self.kb_config['table_name']
        schema = self.kb_config['schema']
        node_client = Node(self.kb_node_url)

        # Create the table using psycopg2
        logger.info(f"Creating table {table_name}")
        await node_client.create_table(table_name, schema)

        # Read PDF and process it
        logger.info("Processing PDFs")
        texts = self.read_pdf()
        all_documents = []
        for i, text in enumerate(texts):
            documents = await self.process_text(text, {"source": f"pdf_{i}"})
            all_documents.extend(documents)

        # Add documents to the table
        logger.info("Adding documents to table")
        for doc in tqdm(all_documents):
            await node_client.add_row(table_name, doc)

        return {"status": "success", "message": f"Successfully populated {table_name} table with {len(all_documents)} chunks"}
    
    async def add_data(self, *args, **kwargs):
        node_client = Node(self.kb_node_url)
        table_name = self.kb_config['table_name']

        data = json.loads(self.input_schema.data)

        # make sure documents are either a list of dicts or a single dict
        if not isinstance(data, list):
            data = [data]

        for doc in data:
            # embed the text if not already embedded
            if 'embedding' not in doc:
                embedding = self.embedder.embed_text(doc['text'])
                doc['embedding'] = embedding

            # if id is not present, generate a random one
            if 'id' not in doc:
                doc['id'] = random.randint(1, 1000000)

            await node_client.add_row(table_name, doc)

        return {"status": "success", "message": f"Successfully added {len(self.input_schema.data)} chunks to table {table_name}"}

    async def run_query(self, *args, **kwargs):
        table_name = self.kb_config['table_name']

        # Create embedding for query
        query_embedding = self.embedder.embed_text(self.query)

        # Query the table using vector similarity with psycopg2
        node_client = Node(self.kb_node_url)

        logger.info(f"Querying table {table_name} with query: {self.query}")
        results = await node_client.vector_search(
            table_name=table_name,
            vector_column="embedding",
            query_vector=query_embedding,
            top_k=5,
            include_similarity=True
        )
        return results

# run.py
async def run(kb_run: Dict[str, Any], *args, **kwargs):
    """
    Run the Memory Embedding Knowledge Base deployment
    Args:
        kb_run: Knowledge base run configuration containing deployment details
    """
    test_kb = TestEmbeddingKB(kb_run)
    return await test_kb.run()


if __name__ == "__main__":
    import asyncio
    import os
    import json
    from dotenv import load_dotenv
    from naptha_sdk.schemas import KBDeployment, KBRun
    
    load_dotenv()
    
    with open(file_path.parent / "configs" / "kb_deployments.json", "r") as file:
        kb_deployments = json.load(file)

    kb_deployment = KBDeployment(**kb_deployments[0])

    # # Example KB run configuration
    # kb_run = KBRun(
    #     consumer_id="test_embedding_kb",
    #     kb_deployment=kb_deployment,
    #     inputs={
    #         "mode": "init",
    #     }
    # )

    # Ensure OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)

    # # Run init
    # print("Starting init...")
    # result = asyncio.run(run(kb_run))
    # print("Init complete!")
    # print("Result:", result)

    # Run query
    kb_run = KBRun(
        consumer_id="test_embedding_kb",
        kb_deployment=kb_deployment,
        inputs={
            "mode": "query",
            "query": "what is attention?"
        }
    )
    print("Starting query...")
    result = asyncio.run(run(kb_run))
    print("Query complete!")
    print("Result:", result)
