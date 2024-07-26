import torch
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
import os
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex, ServiceContext, load_index_from_storage, StorageContext
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

class LLMModel:
    def __init__(self, model_path):
        # Initialize LLM Model
        self.llm = LlamaCPP(
                model_path=model_path,
                temperature=0.1,
                max_new_tokens=256,
                context_window=4096,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": -1},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )
    
    # Build Vector Index using the loaded document
    def get_build_index(self, documents, save_dir="./vector_store/index"):
        embed_model="local:BAAI/bge-small-en-v1.5"
        sentence_window_size=3
        node_parser = SentenceWindowNodeParser(
                window_size = sentence_window_size,
                window_metadata_key = "window",
                original_text_metadata_key = "original_text"
        )
        sentence_context = ServiceContext.from_defaults(
                llm = self.llm,
                embed_model= embed_model,
                node_parser = node_parser,
        )

        if not os.path.exists(save_dir):
            # create and load the index
            index = VectorStoreIndex.from_documents(
                [documents], service_context=sentence_context
            )
            index.storage_context.persist(persist_dir=save_dir)

        else:
            # load the existing index
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=save_dir),
                service_context=sentence_context,
            )

        return index

    # Function to build query engine
    def get_query_engine(self, sentence_index):
        similarity_top_k=6
        rerank_top_n=2
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )
        engine = sentence_index.as_query_engine(
            similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
        )
        memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        chat_mode="context",
        memory=memory
        return engine
