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
                "url": metadata.get("url", ""),
                "title": metadata.get("title", f"chunk_{i}")
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
        # node_client = Node(self.kb_node_url)
        table_name = self.kb_config['table_name']
        schema = self.kb_config['schema']

        # Create the table using psycopg2
        logger.info(f"Creating table {table_name}")
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()

        # Drop table if exists
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Create table with vector extension
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY,
            text TEXT,
            embedding vector({self.kb_config['embedder']['embedding_dim']}),
            metadata jsonb,
            url TEXT,
            title TEXT
        );
        """
        cur.execute(create_table_sql)
        conn.commit()

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
            cur.execute(
                f"""
                INSERT INTO {table_name} (id, text, embedding, metadata, url, title)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    doc['id'],
                    doc['text'],
                    doc['embedding'],
                    json.dumps(doc['metadata']),
                    doc['url'],
                    doc['title']
                )
            )

        conn.commit()
        cur.close()
        conn.close()

        return {"status": "success", "message": f"Successfully populated {table_name} table with {len(all_documents)} chunks"}
    
    async def add_data(self, *args, **kwargs):
        # node_client = Node(self.kb_node_url)
        table_name = self.kb_config['table_name']

        # Process the input data
        data = json.loads(self.input_schema.data)
        documents = []
        
        logger.info(f"Processing {len(data)} documents")
        for item in data:
            text = item.pop('text')  # Remove text from metadata
            processed_docs = await self.process_text(text, item)
            documents.extend(processed_docs)

        # Add documents to the table using psycopg2
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()

        logger.info(f"Adding {len(documents)} chunks to table {table_name}")
        for doc in tqdm(documents):
            cur.execute(
                f"""
                INSERT INTO {table_name} (id, text, embedding, metadata, url, title)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    doc['id'],
                    doc['text'],
                    doc['embedding'],
                    json.dumps(doc['metadata']),
                    doc['url'],
                    doc['title']
                )
            )

        conn.commit()
        cur.close()
        conn.close()

        return {"status": "success", "message": f"Successfully added {len(documents)} chunks to table {table_name}"}

    async def run_query(self, *args, **kwargs):
        table_name = self.kb_config['table_name']

        # Create embedding for query
        query_embedding = self.embedder.embed_text(self.query)

        # Query the table using vector similarity with psycopg2
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()

        logger.info(f"Querying table {table_name} with query: {self.query}")
        cur.execute(
            f"""
            SELECT text, embedding <-> %s::vector AS distance
            FROM {table_name}
            ORDER BY embedding <-> %s::vector
            LIMIT 5;
            """,
            (query_embedding, query_embedding)
        )
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        return {
            "status": "success",
            "results": [{"text": row[0], "distance": row[1]} for row in results]
        }


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
