# # -*- coding: utf-8 -*-
# """Retrieval.ipynb

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/1pAnIX5ydWrtlNQ19CMJWFcZhyiwAu0Vs
# """

# import os
# import pickle
# import time
# from typing import Iterable, List, Optional, Tuple

# import numpy as np

# from transformers.tokenization_utils import PreTrainedTokenizer
# from transformers.tokenization_utils_base import BatchEncoding
# from transformers.utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
# from configuration_rag import RagConfig
# from tokenization_rag import RagTokenizer


# if is_datasets_available():
#     from datasets import Dataset, load_dataset, load_from_disk

# if is_faiss_available():
#     import faiss


# logger = logging.get_logger(__name__)

# class Index:

#     def init_index(self):
#         """
#         将索引数据加载到内存中。
#         """
#         raise NotImplementedError

#     def is_initialized(self):
#         """
#         如果索引已初始化，返回 True，否则返回 False
#         """
#         raise NotImplementedError

#     def get_doc_dicts(self, doc_ids: np.ndarray):
#         """
#         根据文档索引 doc_ids，返回对应文档的标题和内容的字典列表
#         doc_ids：(batch_size, n_docs)，表示批量查询中每个查询的文档索引
#         Returns：[{title: ..., text: ...}, ...]。
#         """
#         raise NotImplementedError

#     def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5):
#         """
#         基于查询向量（question_hidden_states）检索与之最匹配的文档
#         question_hidden_states：(batch_size, vector_size)
#         n_docs：int，表示每个查询需要检索的文档数量
#         Returns
#         检索出的文档索引: (batch_size, n_docs)。
#         检索文档的向量表示: (batch_size, vector_size)
#         """
#         raise NotImplementedError

# class HFIndexBase(Index):

#     def __init__(self, vector_size, dataset, index_initialized=False):
#         self.vector_size = vector_size
#         self.dataset = dataset # Dataset from huggingface
#         self._index_initialized = index_initialized # 索引是否初始化
#         self._check_dataset_format(with_index=index_initialized) # 检查数据集的格式是否符合要求
#         dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32") # 将embedding格式变为numpy

#     def _check_dataset_format(self, with_index: bool):
#         if not isinstance(self.dataset, Dataset): # 检查 dataset 是否是 datasets.Dataset 对象
#             raise TypeError(f"Expected a datasets.Dataset object, but received {type(self.dataset)}")


#         if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0: # 数据集是否包含 title、text 和 embeddings 三列
#             raise ValueError(
#                 "Dataset must contain title (str), text (str) and embeddings (arrays of dimension vector_size), "
#                 f"but got columns {self.dataset.column_names}"
#             )

#         if with_index and "embeddings" not in self.dataset.list_indexes(): # 如果 with_index=True，验证 embeddings 列是否已添加 FAISS 索引。
#             raise ValueError(
#                 "The FAISS index for 'embeddings' is missing."
#             )

#     def init_index(self):
#         raise NotImplementedError()

#     def is_initialized(self):
#         return self._index_initialized

#     def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]: # 根据文档 ID 检索对应文档的字典表示
#         num_docs = doc_ids.shape[0]
#         result = []

#         for i in range(num_docs):
#           doc_id = doc_ids[i].tolist()
#           result.append(self.dataset[doc_id])

#         return result

#     def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
#         _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs) # 在 embeddings 列上查找每个查询的前 n_docs 个最相关文档

#         docs = []
#         for indices in ids:
#             valid_indices = [i for i in indices if i >= 0]  # 过滤有效索引
#             docs.append(self.dataset[valid_indices])

#         vectors = [doc["embeddings"] for doc in docs] # 根据返回的索引ids，从数据集中提取对应的文档

#         for i in range(len(vectors)):
#             if len(vectors[i]) < n_docs:
#                 padding = np.zeros((n_docs - len(vectors[i]), self.vector_size))
#                 vectors[i] = np.vstack([vectors[i], padding]) # 嵌入补齐

#         return np.array(ids), np.array(vectors)
#         # shapes (batch_size, n_docs) and (batch_size, n_docs, d) 每个查询检索到的文档数量，文档嵌入向量的维度（Embedding Dimension）


# class CustomHFIndex(HFIndexBase):

#     def __init__(self, vector_size: int, dataset, index_path=None):
#         super().__init__(vector_size, dataset, index_initialized=index_path is None)
#         self.index_path = index_path

#     @classmethod
#     def load_from_disk(cls, vector_size, dataset_path, index_path):
#         logger.info(f"Loading passages from {dataset_path}")
#         if dataset_path is None or index_path is None:
#             raise ValueError(
#                 "Please provide `dataset_path` and `index_path` after calling `dataset.save_to_disk(dataset_path)` "
#                 "and `dataset.get_index('embeddings').save(index_path)`."
#             )
#         dataset = load_from_disk(dataset_path)
#         return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)

#     def init_index(self):
#         if not self.is_initialized():
#             logger.info(f"Loading index from {self.index_path}")
#             self.dataset.load_faiss_index("embeddings", file=self.index_path)
#             self._index_initialized = True


# class RagRetriever:

#     def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
#         self._init_retrieval = init_retrieval
#         requires_backends(self, ["datasets", "faiss"])
#         super().__init__()
#         self.index = index or self._build_index(config)
#         self.generator_tokenizer = generator_tokenizer
#         self.question_encoder_tokenizer = question_encoder_tokenizer

#         self.n_docs = config.n_docs
#         self.batch_size = config.retrieval_batch_size

#         self.config = config
#         if self._init_retrieval:
#             self.init_retrieval()

#         self.ctx_encoder_tokenizer = None
#         self.return_tokenized_docs = False

#     @staticmethod
#     def _build_index(config):
#         return CustomHFIndex.load_from_disk(
#             vector_size=config.retrieval_vector_size,
#             dataset_path=config.passages_path,
#             index_path=config.index_path,
#         )

#     @classmethod
#     def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
#         requires_backends(cls, ["datasets", "faiss"]) # 检查是否安装了依赖库datasets和faiss
#         config = kwargs.pop("config", None) or RagConfig.from_pretrained(retriever_name_or_path, **kwargs) #从kwargs中获取config
#         rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config) #加载预训练的分词器。RagTokenizer：分为问题编码器和生成器分词器
#         question_encoder_tokenizer = rag_tokenizer.question_encoder # 用于编码问题输入
#         generator_tokenizer = rag_tokenizer.generator # 用于生成器模型
#         if indexed_dataset is not None:
#             config.index_name = "custom"
#             index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
#         else: #调用类方法 _build_index 来自动生成索引
#             index = cls._build_index(config)
#         return cls(# 返回模型配置、分词器和索引
#             config,
#             question_encoder_tokenizer=question_encoder_tokenizer,
#             generator_tokenizer=generator_tokenizer,
#             index=index,
#         )

#     def save_pretrained(self, save_directory): # 保存当前索引和配置到磁盘
#         if isinstance(self.index, CustomHFIndex):
#             if self.config.index_path is None:
#                 index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
#                 self.index.dataset.get_index("embeddings").save(index_path)
#                 self.config.index_path = index_path
#             if self.config.passages_path is None:
#                 passages_path = os.path.join(save_directory, "hf_dataset")
#                 # datasets don't support save_to_disk with indexes right now
#                 faiss_index = self.index.dataset._indexes.pop("embeddings")
#                 self.index.dataset.save_to_disk(passages_path)
#                 self.index.dataset._indexes["embeddings"] = faiss_index
#                 self.config.passages_path = passages_path

#         self.config.save_pretrained(save_directory)
#         rag_tokenizer = RagTokenizer(   # 保存当前使用的分词器到指定路径
#             question_encoder=self.question_encoder_tokenizer,
#             generator=self.generator_tokenizer,
#         )
#         rag_tokenizer.save_pretrained(save_directory)

#     def init_retrieval(self): # 初始化索引器

#         logger.info("initializing retrieval") # 用日志记录当前步骤
#         self.index.init_index() # 将索引加载到内存中

#     def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
#         # 后处理检索到的文档，使其适配生成模型

#         def cat_input_and_doc(doc_title, doc_text, input_string, prefix):  # 合并文档标题、内容和输入字符串，形成生成模型的输入格式
#             # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
#             # TODO(piktus): better handling of truncation
#             if doc_title.startswith('"'):
#                 doc_title = doc_title[1:]
#             if doc_title.endswith('"'):
#                 doc_title = doc_title[:-1]
#             if prefix is None:
#                 prefix = ""
#             out = (prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace(
#                 "  ", " "
#             )
#             return out

#         rag_input_strings = [   # 遍历文档集合，将每个文档的标题和内容与查询字符串拼接成输入
#             cat_input_and_doc(
#                 docs[i]["title"][j],
#                 docs[i]["text"][j],
#                 input_strings[i],
#                 prefix,
#             )
#             for i in range(len(docs))
#             for j in range(n_docs)
#         ]

#         contextualized_inputs = self.generator_tokenizer.batch_encode_plus( # 使用生成器分词器编码输入
#             rag_input_strings,
#             max_length=self.config.max_combined_length,
#             return_tensors=return_tensors,
#             padding="max_length",
#             truncation=True,
#         )

#         return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]  # 编码后的输入ID和注意力掩码

#     def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]: #该函数将输入的 t 分成多个小块，每个小块的大小为 chunk_size。
#         return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

#     def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
#         question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size) # 将输入的 question_hidden_states 按批量大小 self.batch_size 分成小批量
#         ids_batched = [] # 存储检索结果的文档 ID
#         vectors_batched = [] #存储检索结果的文档向量
#         for question_hidden_states in question_hidden_states_batched: # 遍历每个小批量
#             # start_time = time.time()
#             ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs) # 调用索引（self.index.get_top_docs）根据当前批量的查询向量 question_hidden_states 检索 n_docs 个相关文档。
#             # logger.debug(
#             #     f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
#             # )
#             ids_batched.extend(ids) # 将当前批量的检索结果加入到总的结果列表中
#             vectors_batched.extend(vectors)
#         return (   # 将最终检索的文档 ID 和向量转换为 Numpy 数组格式并返回
#             np.array(ids_batched),
#             np.array(vectors_batched),
#         )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

#     def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]: # 使用 _main_retrieve 获取检索结果

#         doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
#         return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids) # 详细文档信息

#     def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer):
#         # used in end2end retriever training
#         self.ctx_encoder_tokenizer = ctx_encoder_tokenizer # 接收一个预训练的分词器（Tokenizer），用于文本编码
#         self.return_tokenized_docs = True

#     def __call__(
#         self,
#         question_input_ids: List[List[int]],  # 输入查询的 Token ID
#         question_hidden_states: np.ndarray,   # 用于检索的隐藏状态向量
#         prefix=None,  # 用于生成器 Tokenizer 的前缀字符串
#         n_docs=None,   # 指定要检索的文档数量
#         return_tensors=None,
#     ) -> BatchEncoding:

#         n_docs = n_docs if n_docs is not None else self.n_docs
#         prefix = prefix if prefix is not None else self.config.generator.prefix
#         retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)  # 检索相关文档

#         input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)  # 解码输入的 Token ID，变为文本字符串
#         context_input_ids, context_attention_mask = self.postprocess_docs(
#             docs, input_strings, prefix, n_docs, return_tensors=return_tensors
#         )  # 使用 postprocess_docs 处理检索到的文档数据

#         if self.return_tokenized_docs:  # 如果需要对检索到的文档进行Tokenizer编码处理
#             retrieved_doc_text = []
#             retrieved_doc_title = []

#             for b_idx in range(len(docs)):
#                 for doc_idx in range(n_docs):
#                     retrieved_doc_text.append(docs[b_idx]["text"][doc_idx])
#                     retrieved_doc_title.append(docs[b_idx]["title"][doc_idx])

#             tokenized_docs = self.ctx_encoder_tokenizer(
#                 retrieved_doc_title,
#                 retrieved_doc_text,
#                 truncation=True,
#                 padding="longest",
#                 return_tensors=return_tensors,
#             )

#             return BatchEncoding(
#                 {
#                     "context_input_ids": context_input_ids,
#                     "context_attention_mask": context_attention_mask,
#                     "retrieved_doc_embeds": retrieved_doc_embeds,
#                     "doc_ids": doc_ids,
#                     "tokenized_doc_ids": tokenized_docs["input_ids"],
#                     "tokenized_doc_attention_mask": tokenized_docs["attention_mask"],
#                 },
#                 tensor_type=return_tensors,
#             )

#         else:
#             return BatchEncoding(
#                 {
#                     "context_input_ids": context_input_ids,
#                     "context_attention_mask": context_attention_mask,
#                     "retrieved_doc_embeds": retrieved_doc_embeds,
#                     "doc_ids": doc_ids,
#                 },
#                 tensor_type=return_tensors,
#             )

import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
from configuration_rag import RagConfig
from tokenization_rag import RagTokenizer

if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss

logger = logging.get_logger(__name__)

class Index:

    def init_index(self):
        """
        Load the index data into memory.
        """
        raise NotImplementedError

    def is_initialized(self):
        """
        Return True if the index is initialized, otherwise False.
        """
        raise NotImplementedError

    def get_doc_dicts(self, doc_ids: np.ndarray):
        """
        Given document indices (doc_ids), return a list of dictionaries containing titles and texts of the documents.
        doc_ids: (batch_size, n_docs), representing the indices of documents for each query in the batch.
        Returns: [{title: ..., text: ...}, ...].
        """
        raise NotImplementedError

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5):
        """
        Retrieve the most relevant documents based on the query vectors (question_hidden_states).
        question_hidden_states: (batch_size, vector_size)
        n_docs: int, the number of documents to retrieve for each query.
        Returns:
        Retrieved document indices: (batch_size, n_docs).
        Document vector representations: (batch_size, n_docs, vector_size).
        """
        raise NotImplementedError

class HFIndexBase(Index):

    def __init__(self, vector_size, dataset, index_initialized=False):
        self.vector_size = vector_size
        self.dataset = dataset  # Dataset from huggingface
        self._index_initialized = index_initialized  # Whether the index is initialized
        self._check_dataset_format(with_index=index_initialized)  # Validate dataset format
        dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")  # Convert embeddings to numpy format

    def _check_dataset_format(self, with_index: bool):
        if not isinstance(self.dataset, Dataset):  # Check if dataset is of type datasets.Dataset
            raise TypeError(f"Expected a datasets.Dataset object, but received {type(self.dataset)}")

        if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0:  # Check if dataset contains required columns
            raise ValueError(
                "Dataset must contain title (str), text (str) and embeddings (arrays of dimension vector_size), "
                f"but got columns {self.dataset.column_names}"
            )

        if with_index and "embeddings" not in self.dataset.list_indexes():  # Validate if embeddings column has a FAISS index if with_index=True
            raise ValueError(
                "The FAISS index for 'embeddings' is missing."
            )

    def init_index(self):
        raise NotImplementedError()

    def is_initialized(self):
        return self._index_initialized

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:  # Retrieve the document representations based on IDs
        num_docs = doc_ids.shape[0]
        result = []

        for i in range(num_docs):
          doc_id = doc_ids[i].tolist()
          result.append(self.dataset[doc_id])

        return result

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)  # Search for top n_docs most relevant documents

        docs = []
        for indices in ids:
            valid_indices = [i for i in indices if i >= 0]  # Filter valid indices
            docs.append(self.dataset[valid_indices])

        vectors = [doc["embeddings"] for doc in docs]  # Extract document embeddings based on returned indices

        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                padding = np.zeros((n_docs - len(vectors[i]), self.vector_size))
                vectors[i] = np.vstack([vectors[i], padding])  # Pad embeddings to match n_docs

        return np.array(ids), np.array(vectors)
        # shapes (batch_size, n_docs) and (batch_size, n_docs, d) - Number of retrieved documents and their vector dimensions


class CustomHFIndex(HFIndexBase):

    def __init__(self, vector_size: int, dataset, index_path=None):
        super().__init__(vector_size, dataset, index_initialized=index_path is None)
        self.index_path = index_path

    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        logger.info(f"Loading passages from {dataset_path}")
        if dataset_path is None or index_path is None:
            raise ValueError(
                "Please provide `dataset_path` and `index_path` after calling `dataset.save_to_disk(dataset_path)` "
                "and `dataset.get_index('embeddings').save(index_path)`."
            )
        dataset = load_from_disk(dataset_path)
        return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)

    def init_index(self):
        if not self.is_initialized():
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
            self._index_initialized = True


class RagRetriever:

    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
        self._init_retrieval = init_retrieval
        requires_backends(self, ["datasets", "faiss"])
        super().__init__()
        self.index = index or self._build_index(config)
        self.generator_tokenizer = generator_tokenizer
        self.question_encoder_tokenizer = question_encoder_tokenizer

        self.n_docs = config.n_docs
        self.batch_size = config.retrieval_batch_size

        self.config = config
        if self._init_retrieval:
            self.init_retrieval()

        self.ctx_encoder_tokenizer = None
        self.return_tokenized_docs = False

    @staticmethod
    def _build_index(config):
        return CustomHFIndex.load_from_disk(
            vector_size=config.retrieval_vector_size,
            dataset_path=config.passages_path,
            index_path=config.index_path,
        )

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        requires_backends(cls, ["datasets", "faiss"])  # Check if required libraries are installed
        config = kwargs.pop("config", None) or RagConfig.from_pretrained(retriever_name_or_path, **kwargs)  # Retrieve config from kwargs
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)  # Load pretrained tokenizer (question encoder + generator)
        question_encoder_tokenizer = rag_tokenizer.question_encoder  # Tokenizer for question input
        generator_tokenizer = rag_tokenizer.generator  # Tokenizer for generator model
        if indexed_dataset is not None:
            config.index_name = "custom"
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        else:  # Automatically build index using class method
            index = cls._build_index(config)
        return cls(  # Return model configuration, tokenizers, and index
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
        )

    def save_pretrained(self, save_directory):  # Save current index and configuration to disk
        if isinstance(self.index, CustomHFIndex):
            if self.config.index_path is None:
                index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
                self.index.dataset.get_index("embeddings").save(index_path)
                self.config.index_path = index_path
            if self.config.passages_path is None:
                passages_path = os.path.join(save_directory, "hf_dataset")
                # Datasets don't support save_to_disk with indexes right now
                faiss_index = self.index.dataset._indexes.pop("embeddings")
                self.index.dataset.save_to_disk(passages_path)
                self.index.dataset._indexes["embeddings"] = faiss_index
                self.config.passages_path = passages_path

        self.config.save_pretrained(save_directory)
        rag_tokenizer = RagTokenizer(   # Save the current tokenizer to the specified directory
            question_encoder=self.question_encoder_tokenizer,
            generator=self.generator_tokenizer,
        )
        rag_tokenizer.save_pretrained(save_directory)

    def init_retrieval(self):  # Initialize the retriever

        logger.info("initializing retrieval")  # Log the current step
        self.index.init_index()  # Load the index into memory

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        # Postprocess retrieved documents to make them suitable for the generator model

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):  # Concatenate document title, content, and input string into the generator input format
            # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
            # TODO(piktus): better handling of truncation
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ""
            out = (prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace(
                "  ", " "
            )
            return out

        rag_input_strings = [   # Combine each document's title, content, and query string into inputs
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))
            for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(  # Use the generator tokenizer to encode inputs
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]  # Encoded input IDs and attention masks

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:  # Split the input tensor t into smaller chunks of size chunk_size
        return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)  # Divide question_hidden_states into smaller batches
        ids_batched = []  # Store document IDs from retrieval
        vectors_batched = []  # Store document vectors from retrieval
        for question_hidden_states in question_hidden_states_batched:  # Process each batch
            # start_time = time.time()
            ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)  # Retrieve top n_docs for each query
            # logger.debug(
            #     f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
            # )
            ids_batched.extend(ids)  # Append current batch retrieval results to the list
            vectors_batched.extend(vectors)
        return (   # Convert retrieved document IDs and vectors to Numpy arrays and return
            np.array(ids_batched),
            np.array(vectors_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:  # Use _main_retrieve to get retrieval results

        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)  # Detailed document information

    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer):
        # Used in end-to-end retriever training
        self.ctx_encoder_tokenizer = ctx_encoder_tokenizer  # Set a pretrained tokenizer for text encoding
        self.return_tokenized_docs = True

    def __call__(
        self,
        question_input_ids: List[List[int]],  # Input query token IDs
        question_hidden_states: np.ndarray,   # Hidden states for retrieval
        prefix=None,  # Prefix string for the generator tokenizer
        n_docs=None,   # Number of documents to retrieve
        return_tensors=None,
    ) -> BatchEncoding:

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)  # Retrieve relevant documents

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)  # Decode input token IDs into text strings
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )  # Process retrieved documents

        if self.return_tokenized_docs:  # If tokenized document processing is required
            retrieved_doc_text = []
            retrieved_doc_title = []

            for b_idx in range(len(docs)):
                for doc_idx in range(n_docs):
                    retrieved_doc_text.append(docs[b_idx]["text"][doc_idx])
                    retrieved_doc_title.append(docs[b_idx]["title"][doc_idx])

            tokenized_docs = self.ctx_encoder_tokenizer(
                retrieved_doc_title,
                retrieved_doc_text,
                truncation=True,
                padding="longest",
                return_tensors=return_tensors,
            )

            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                    "tokenized_doc_ids": tokenized_docs["input_ids"],
                    "tokenized_doc_attention_mask": tokenized_docs["attention_mask"],
                },
                tensor_type=return_tensors,
            )

        else:
            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                },
                tensor_type=return_tensors,
            )
