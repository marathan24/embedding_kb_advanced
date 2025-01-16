# Embedding Knowledge Base Modules

This is a simple knowledge base module to demo how to use embeddings. 

### Create a New Embedding Knowledge Base on a Node

```bash
naptha create kb:embedding_kb 
```

### Initialize the content in the Knowledge Base

```bash
naptha run kb:embedding_kb -p "function_name='init'"
```

### Add to the Knowledge Base

```bash
naptha run kb:embedding_kb -p '{
    "function_name": "add_data",
    "function_input_data": {"path": "data/aiayn.pdf"}
}'
```

### Query the Knowledge Base Module

```bash
naptha run kb:embedding_kb -p '{
    "function_name": "run_query",
    "function_input_data": {
        "query": "what is attention?"
    }
}'
```
