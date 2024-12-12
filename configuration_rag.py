from transformers.configuration_utils import PretrainedConfig
from transformers.utils import add_start_docstrings

class RagConfig(PretrainedConfig):
    model_type = "rag"
    is_composition = True

    def __init__(
        self,
        vocab_size=None,
        is_encoder_decoder=True,
        prefix=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        title_sep=" / ",
        doc_sep=" // ",
        n_docs=5,
        max_combined_length=300,
        retrieval_vector_size=768,
        retrieval_batch_size=8,
        dataset="wiki_dpr",
        dataset_split="train",
        index_name="compressed",
        index_path=None,
        passages_path=None,
        use_dummy_dataset=False,
        reduce_loss=False,
        label_smoothing=0.0,
        do_deduplication=True,
        exclude_bos_score=False,
        do_marginalize=False,
        output_retrieved=False,
        use_cache=True,
        forced_eos_token_id=None,
        dataset_revision=None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            prefix=prefix,
            vocab_size=vocab_size,
            **kwargs,
        )
        if "question_encoder" not in kwargs or "generator" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because "
                f"both `question_encoder` and `generator` sub-configurations were not passed, only {kwargs}"
            )
        question_encoder_config = kwargs.pop("question_encoder")
        question_encoder_model_type = question_encoder_config.pop("model_type")
        decoder_config = kwargs.pop("generator")
        decoder_model_type = decoder_config.pop("model_type")

        from configuration_auto import AutoConfig

        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)

        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score
        self.do_marginalize = do_marginalize

        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length

        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name

        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset
        self.dataset_revision = dataset_revision

        self.output_retrieved = output_retrieved

        self.do_deduplication = do_deduplication

        self.use_cache = use_cache

        if self.forced_eos_token_id is None:
            self.forced_eos_token_id = getattr(self.generator, "forced_eos_token_id", None)

    def from_question_encoder_generator_configs(
        cls, question_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        return cls(question_encoder=question_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)
