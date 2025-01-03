import logging
import json
import random
import PyPDF2
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from naptha_sdk.client.node import Node
from naptha_sdk.schemas import KBDeployment, KBRunInput
from naptha_sdk.storage.storage_provider import StorageProvider
from naptha_sdk.storage.schemas import CreateTableRequest, CreateRowRequest, ReadStorageRequest, DatabaseReadOptions
from embedding_kb.schemas import InputSchema, EmbedderConfig
from embedding_kb.embedder import MemoryEmbedder, RecursiveTextSplitter

logger = logging.getLogger(__name__)

class EmbeddingKB:
    def __init__(self, deployment: KBDeployment):
        self.deployment = deployment
        self.config = self.deployment.config
        self.storage_provider = StorageProvider(self.deployment.node)
        self.table_name = self.config["path"]

        embedder_config = EmbedderConfig(**self.config["embedder"])

        # Initialize embedder and splitter
        self.embedder = MemoryEmbedder(model=embedder_config.model)
        self.splitter = RecursiveTextSplitter(
            chunk_size=embedder_config.chunk_size,
            chunk_overlap=embedder_config.chunk_overlap,
            separators=embedder_config.separators
        )
        
    # TODO: Remove this. In future, the create function should be called by create_module in the same way that run is called by run_module
    async def init(self, *args, **kwargs):
        await create(self.deployment)
        return {"status": "success", "message": f"Successfully populated {self.table_name} table"}

    def _read_pdf(self, pdf_file_paths) -> str:
        """Read a PDF file and return its text content"""
        # glob all pdfs in the data folder from pathlib

        pdfs = []
        for pdf_file in pdf_file_paths:
            text = ""
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            pdfs.append(text)
        return pdfs

    async def _process_text(self,text: str, metadata: Dict = None) -> List[Dict]:
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

    async def add_data(self, input_data: Dict[str, Any], *args, **kwargs):

        file_path = Path(__file__).resolve()
        pdf_file_path = file_path.parent / input_data["path"]

        # Read PDF and process it
        logger.info("Processing PDFs")
        texts = self._read_pdf([pdf_file_path])

        all_documents = []
        for i, text in enumerate(texts):
            documents = await self._process_text(text, {"source": f"pdf_{i}"})
            all_documents.extend(documents)

        # Add documents to the table
        logger.info("Adding documents to table")
        for doc in tqdm(all_documents):

            create_row_request = CreateRowRequest(
                storage_type=self.config['storage_type'],
                path=self.table_name,
                data=doc
            )

            # Add a row
            create_row_result = await self.storage_provider.create(create_row_request)

        logger.info(f"Add row result: {create_row_result}")

        return {"status": "success", "message": f"Successfully added {input_data["path"]} to table {self.table_name}"}

    async def run_query(self, input_data: Dict[str, Any], *args, **kwargs):
        logger.info(f"Querying table {self.table_name} with query: {input_data['query']}")

        # Create embedding for query
        query_embedding = self.embedder.embed_text(input_data['query'])

        db_read_options = DatabaseReadOptions(
            query_vector=query_embedding,
            vector_col="embedding",
            top_k=5,
            include_similarity=True
        )

        read_storage_request = ReadStorageRequest(
            storage_type=self.config['storage_type'],
            path=self.table_name,
            options=db_read_options
        )

        read_result = await self.storage_provider.read(read_storage_request)
        logger.info(f"Query results: {read_result}")

        return read_result[0]["text"].replace("\n", " ").strip()


# TODO: Make it so that the create function is called when the kb/create endpoint is called
async def create(deployment: KBDeployment):
    file_path = Path(__file__).resolve()
    init_data_path = file_path.parent / "data"

    storage_provider = StorageProvider(deployment.node)
    storage_type = deployment.config['storage_type']
    table_name = deployment.config['path']
    schema = deployment.config['schema']

    embedding_kb = EmbeddingKB(deployment)

    logger.info(f"Creating {storage_type} at {table_name} with schema {schema}")

    create_table_request = CreateTableRequest(
        storage_type=storage_type,
        path=table_name,
        schema=schema
    )

    create_table_result = await storage_provider.create(create_table_request)

    pdf_file_paths = list(init_data_path.glob("*.pdf"))

    # Read PDF and process it
    logger.info("Processing PDFs")
    texts = embedding_kb._read_pdf(pdf_file_paths)
    all_documents = []
    for i, text in enumerate(texts):
        documents = await embedding_kb._process_text(text, {"source": f"pdf_{i}"})
        all_documents.extend(documents)

    # Add documents to the table
    logger.info("Adding documents to table")
    for doc in tqdm(all_documents):

        create_row_request = CreateRowRequest(
            storage_type=storage_type,
            path=table_name,
            data=doc
        )

        # Add a row
        create_row_result = await storage_provider.create(create_row_request)

        logger.info(f"Add row result: {create_row_result}")

    return {"status": "success", "message": f"Successfully populated {table_name} table with {len(all_documents)} chunks"}

async def run(module_run: KBRunInput, *args, **kwargs):
    """
    Run a method ofthe Memory Embedding Knowledge Base deployment
    Args:
        module_run: Module run configuration containing deployment details
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    embedding_kb = EmbeddingKB(deployment)

    method = getattr(embedding_kb, module_run.inputs.function_name, None)

    if not method:
        raise ValueError(f"Invalid function name: {module_run.inputs.function_name}")

    return await method(module_run.inputs.function_input_data)


if __name__ == "__main__":
    import asyncio
    import os
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    from naptha_sdk.schemas import KBRunInput

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("kb", "embedding_kb/configs/deployment.json", node_url = os.getenv("NODE_URL")))


    inputs_dict = {
        "init": InputSchema(
            function_name="init",
            function_input_data=None,
        ),
        "run_query": InputSchema(
            function_name="run_query",
            function_input_data={"query": "what is attention?"},
        ),
        "add_data": InputSchema(
            function_name="add_data",
            function_input_data={"path": "data/aiayn.pdf"},
        ),
    }

    module_run = KBRunInput(
        inputs=inputs_dict["add_data"],
        deployment=deployment,
        consumer_id=naptha.user.id,
    )

    result = asyncio.run(run(module_run))
    print("Result:", result)
