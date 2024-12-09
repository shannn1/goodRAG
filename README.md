## Enhanced Knowledge Retrieval and Response Generation for Contextual Question-Answering Using RAG
**Background:** Open-domain Question Answering (ODQA) has emerged as a critical task in natural language processing, where the goal is to provide accurate and relevant answers to user queries from a vast pool of documents. In this project, we build a Retrieval-Augmented Generation (RAG-Sequence) model from scratch, handling the dynamic and diverse nature of open-domain questions. 

**Objective:** Improve ODQA accuracy by leveraging the synergy between retrieval-based methods and generative transformers.

**Dataset:** [Natural Qestion](https://huggingface.co/datasets/lighteval/natural_questions_clean)

**Methods:** 
- **T5-small:** Scaled-down version of the T5 (Text-to-Text Transfer Transformer), while maintaining the core capabilities.
- **Fine-tuned T5 small:** Fine-tuning was performed on only the last two decoder layers of the T5-small model. The training configuration included a learning rate of 2e-5, a batch size of 16, weight decay of 0.01, and a total of 3 epochs.
- **RAG:** RAG-Sequence integrates retrieval-based methods with generative transformers, enabling dynamic augmentation of model knowledge during inference.
![Overview of RAG approach](https://drive.google.com/uc?id=1JOn2dZU5vX5r93fjlERZZn1rsVqUv-E4)
The system uses a Query Encoder to embed the input query and retrieves the top‑k relevant contexts from a vector database using DPR. These retrieved contexts, combined with the original query, are then fed into the pre-trained BART generator to produce the final response.
