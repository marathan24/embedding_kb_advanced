import logging
import os
import random
import PyPDF2
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from naptha_sdk.schemas import KBDeployment, KBRunInput
from naptha_sdk.storage.storage_provider import StorageProvider
from naptha_sdk.storage.schemas import (
    CreateStorageRequest, ReadStorageRequest, StorageType,
    DatabaseReadOptions
)
from naptha_sdk.user import sign_consumer_id
from embedding_kb.schemas import InputSchema
from embedding_kb.embedder import MemoryEmbedder, RecursiveTextSplitter

logger = logging.getLogger(__name__)

class EmbeddingKB:
    def __init__(self, deployment: KBDeployment):
        self.deployment = deployment
        self.config = self.deployment.config
        self.storage_provider = StorageProvider(self.deployment.node)
        self.table_name = self.config.storage_config.path
        self.storage_type = self.config.storage_config.storage_type

        embedder_config = self.config.llm_config

        self.embedder = MemoryEmbedder(model=embedder_config.model)
        self.splitter = RecursiveTextSplitter(
            chunk_size=embedder_config.options.chunk_size,
            chunk_overlap=embedder_config.options.chunk_overlap,
            separators=embedder_config.options.separators
        )

    async def init(self, *args, **kwargs):
        await create(self.deployment)
        return {"status": "success", "message": f"Successfully populated {self.table_name} table"}

    def _read_pdf(self, pdf_file_paths) -> str:
        pdfs = []
        for pdf_file in pdf_file_paths:
            text = ""
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            pdfs.append(text)
        return pdfs

    async def _process_text(self, text: str, metadata: Dict = None) -> List[Dict]:
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

        logger.info("Processing PDFs")
        texts = self._read_pdf([pdf_file_path])

        all_documents = []
        for i, text in enumerate(texts):
            documents = await self._process_text(text, {"source": f"pdf_{i}"})
            all_documents.extend(documents)

        logger.info("Adding documents to table")
        for doc in tqdm(all_documents):
            create_request = CreateStorageRequest(
                storage_type=StorageType.DATABASE,
                path=self.table_name,
                data={
                    "data": doc
                }
            )
            await self.storage_provider.execute(create_request)

        return {"status": "success", "message": f"Successfully added {input_data['path']} to table {self.table_name}"}

    async def run_query(self, input_data: Dict[str, Any], *args, **kwargs):
        logger.info(f"Querying table {self.table_name} with query")

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
            return results[0]["text"].replace("\n", " ").strip()
        return ""

async def create(deployment: KBDeployment):
    file_path = Path(__file__).resolve()
    init_data_path = file_path.parent / "data"

    storage_provider = StorageProvider(deployment.node)
    table_name = deployment.config.storage_config.path
    schema = deployment.config.storage_config.storage_schema

    embedding_kb = EmbeddingKB(deployment)

    logger.info(f"Creating database table {table_name} with schema {schema}")

    create_table_request = CreateStorageRequest(
        storage_type=StorageType.DATABASE,
        path=table_name,
        data={},
        options={"schema": schema, "create_table": True}
    )

    await storage_provider.execute(create_table_request)

    pdf_file_paths = list(init_data_path.glob("*.pdf"))

    logger.info("Processing PDFs")
    texts = embedding_kb._read_pdf(pdf_file_paths)
    all_documents = []
    for i, text in enumerate(texts):
        documents = await embedding_kb._process_text(text, {"source": f"pdf_{i}"})
        all_documents.extend(documents)

    logger.info("Adding documents to table")
    for doc in tqdm(all_documents):
        create_row_request = CreateStorageRequest(
            storage_type=StorageType.DATABASE,
            path=table_name,
            data={
                'data': doc
            }
        )
        await storage_provider.execute(create_row_request)

    return {"status": "success", "message": f"Successfully populated {table_name} table with {len(all_documents)} chunks"}

async def run(module_run: Dict, *args, **kwargs):
    
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

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("kb", "embedding_kb/configs/deployment.json", node_url = os.getenv("NODE_URL")))


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

    module_run = {
        "inputs": inputs_dict["add_data"],
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    result = asyncio.run(run(module_run))
    print("Result:", result)
