# /embedding_kb/run.py

import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, List

import PyPDF2
import asyncio
from tqdm import tqdm

from naptha_sdk.schemas import KBDeployment, KBRunInput
from naptha_sdk.storage.storage_provider import StorageProvider
from naptha_sdk.storage.schemas import (
    CreateStorageRequest,
    ReadStorageRequest,
    StorageType,
    DatabaseReadOptions,
)
from naptha_sdk.user import sign_consumer_id

from embedding_kb.schemas import InputSchema
from embedding_kb.embedder import MemoryEmbedder, SemChunkTextSplitter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class EmbeddingKB:
    def __init__(self, deployment: KBDeployment):
        self.deployment = deployment
        self.config = self.deployment.config
        self.storage_provider = StorageProvider(self.deployment.node)
        self.table_name = self.config.storage_config.path
        self.storage_type = self.config.storage_config.storage_type

        llm_config = self.config.llm_config

        # Initialize MemoryEmbedder with new chunking parameters
        self.embedder = MemoryEmbedder(
            model=llm_config.model,
            chunk_size=llm_config.options.chunk_size,
            chunk_overlap=llm_config.options.chunk_overlap
        )

        # Initialize SemChunkTextSplitter
        self.splitter = SemChunkTextSplitter(
            chunk_size=llm_config.options.chunk_size,
            chunk_overlap=llm_config.options.chunk_overlap
        )

    async def init(self, *args, **kwargs):
        """Initialize the EmbeddingKB by creating the database table and populating it with initial data."""
        response = await create(self.deployment)
        return response

    def _read_pdf(self, pdf_file) -> List[str]:
        """Read and extract text from a list of PDF files."""
        pdfs = []
        
        
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
            pdfs.append(text)
        return pdfs

    async def _process_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split the text into chunks and generate embeddings for each chunk."""
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
            print("Added : ", {i})
        return documents

    async def add_data(self, input_data: Dict[str, Any], *args, **kwargs):
        """Add data from a PDF file to the embedding knowledge base."""
        file_path = Path(__file__).resolve()
        pdf_file_path = file_path.parent / input_data["path"]

        logger.info(f"Processing PDF: {pdf_file_path}")
        texts = self._read_pdf([pdf_file_path])

        all_documents = []
        for i, text in enumerate(texts):
            documents = await self._process_text(text, {"source": f"pdf_{i}"})
            all_documents.extend(documents)

        logger.info("Adding documents to the database table")
        for doc in tqdm(all_documents, desc="Adding Documents"):
            create_request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={
                    "data": doc
                }
            )
            await self.storage_provider.execute(create_request)

        logger.info(f"Successfully added {input_data['path']} to table {self.table_name}")
        return {
            "status": "success",
            "message": f"Successfully added {input_data['path']} to table {self.table_name}"
        }

    async def run_query(self, input_data: Dict[str, Any], *args, **kwargs):
        """Run a query against the embedding knowledge base and retrieve the most relevant text."""
        logger.info(f"Running query on table {self.table_name}: {input_data['query']}")

        query_embedding = self.embedder.embed_text(input_data['query'])

        db_read_options = DatabaseReadOptions(
            query_vector=query_embedding,
            vector_col=self.config.storage_config.options["vector_col"],
            top_k=self.config.storage_config.options["top_k"],
            include_similarity=self.config.storage_config.options["include_similarity"]
        )

        read_request = ReadStorageRequest(
            storage_type=StorageType.DATABASE,
            path=self.table_name,
            options=db_read_options.model_dump()
        )

        read_result = await self.storage_provider.execute(read_request)
        results = read_result.data if hasattr(read_result, 'data') else read_result

        if isinstance(results, list) and results:
            top_result = results[0]
            logger.info(f"Top result: {top_result['text']}")
            return top_result["text"].replace("\n", " ").strip()

        logger.warning("No results found for the query")
        return ""


async def create(deployment: KBDeployment):
    """Create the database table and populate it with initial PDF data."""
    file_path = Path(__file__).resolve()
    init_data_path = file_path.parent / "data"

    storage_provider = StorageProvider(deployment.node)
    table_name = deployment.config.storage_config.path
    schema = deployment.config.storage_config.storage_schema

    embedding_kb = EmbeddingKB(deployment)

    logger.info(f"Creating database table '{table_name}' with schema: {schema}")

    create_table_request = CreateStorageRequest(
        storage_type=StorageType.DATABASE,
        path=table_name,
        data={},
        options={"schema": schema, "create_table": True}
    )

    await storage_provider.execute(create_table_request)

    pdf_file_paths = "embedding_kb/data/aiayn.pdf"
    if not pdf_file_paths:
        logger.warning("No PDF files found to process during initialization.")

    logger.info("Processing PDFs for initialization")
    texts = embedding_kb._read_pdf(pdf_file_paths)
    all_documents = []
    for i, text in enumerate(texts):
        documents = await embedding_kb._process_text(text, {"source": f"pdf_{i}"})
        all_documents.extend(documents)

    logger.info("Adding documents to the database table")
    for doc in tqdm(all_documents, desc="Adding Initial Documents"):
        create_row_request = CreateStorageRequest(
            storage_type=StorageType.DATABASE,
            path=table_name,
            data={
                'data': doc
            }
        )
        await storage_provider.execute(create_row_request)

    logger.info(f"Successfully populated '{table_name}' table with {len(all_documents)} chunks")
    return {
        "status": "success",
        "message": f"Successfully populated '{table_name}' table with {len(all_documents)} chunks"
    }


async def run(module_run: Dict, *args, **kwargs):
    """Entry point for running module functions based on the provided input."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    module_run = KBRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)

    logger.info(f"Module run: {module_run}")

    embedding_kb = EmbeddingKB(module_run.deployment)

    method = getattr(embedding_kb, module_run.inputs.func_name, None)

    if not method:
        raise ValueError(f"Invalid function name: {module_run.inputs.func_name}")

    return await method(module_run.inputs.func_input_data)


if __name__ == "__main__":
    import asyncio
    import os
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    # Initialize logging for the main execution
    main_logger = logging.getLogger("Main")
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(handler)

    async def main():
        naptha = Naptha()

        deployment = await setup_module_deployment(
            "kb",
            "embedding_kb/configs/deployment.json",
            node_url=os.getenv("NODE_URL")
        )

        # Define inputs for different functions
        inputs_dict = {
            "init": {
                "func_name": "init",
                "func_input_data": None,
            },
            "run_query": {
                "func_name": "run_query",
                "func_input_data": {"query": "what is attention?"},
            },
            "add_data": {
                "func_name": "add_data",
                "func_input_data": {"path": "data/aiayn.pdf"},
            },
        }

        # Example: Initialize the database
        init_run = {
            "inputs": inputs_dict["init"],
            "deployment": deployment,
            "consumer_id": naptha.user.id,
            "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
        }

        init_result = await run(init_run)
        main_logger.info("Initialization Result: %s", init_result)

        # Example: Add data from a PDF
        add_data_run = {
            "inputs": inputs_dict["add_data"],
            "deployment": deployment,
            "consumer_id": naptha.user.id,
            "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
        }

        add_data_result = await run(add_data_run)
        main_logger.info("Add Data Result: %s", add_data_result)

        # Example: Run a query
        # query_run = {
        #     "inputs": inputs_dict["run_query"],
        #     "deployment": deployment,
        #     "consumer_id": naptha.user.id,
        #     "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
        # }

        # query_result = await run(query_run)
        # main_logger.info("Query Result: %s", query_result)

    # Execute the main function
    asyncio.run(main())
