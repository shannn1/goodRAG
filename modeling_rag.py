
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever


logger = logging.get_logger(__name__)


@dataclass
class RetrievAugLMMarginOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None #(1,) 语言模型的损失
    logits: torch.FloatTensor = None # (batch_size, sequence_length, config.vocab_size) 语言模型头部的预测分数
    doc_scores: torch.FloatTensor = None # (batch_size, config.n_docs) 检索的文档嵌入与问题编码器最后一层隐藏状态之间的分数
    past_key_values: Optional[List[torch.FloatTensor]] = None # (2, batch_size, num_heads, sequence_length, embed_size_per_head) 解码器的预计算隐藏状态
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None # (batch_size, config.n_docs, hidden_size) 由检索器找到的文档嵌入，用于计算 doc_scores
    retrieved_doc_ids: Optional[torch.LongTensor] = None # (batch_size, config.n_docs) 检索到的文档索引
    context_input_ids: Optional[torch.LongTensor] = None # 输入 ID 注意力掩码, 检索文档与问题编码器输入中处理得来的数据
    context_attention_mask: Optional[torch.LongTensor] = None # 注意力掩码, 检索文档与问题编码器输入中处理得来的数据
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None # 问题编码器的最后一层隐藏状态
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None # 问题编码器每一层的中间结果
    question_enc_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None # 问题编码器的注意力分数
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None # 生成器编码器的最后一层隐藏状态生成器编码器的中间结果
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None # 生成器编码器的注意力分数
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None # 生成器编码器的注意力分数
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None # 生成器解码器的中间结果
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None # 生成器解码器的注意力分数
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None # 生成器解码器的交叉注意力分数


@dataclass
class RetrievAugLMOutput(ModelOutput):
    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class RagPreTrainedModel(PreTrainedModel):
    config_class = RagConfig # 配置类为 RagConfig，用于加载模型相关的参数
    base_model_prefix = "rag" # 定义模型的前缀，用于标识模型相关文件或变量
    _supports_flash_attn_2 = True  # 标记该模型是否支持高效注意力机制
    _supports_sdpa = True

    @classmethod
    def from_pretrained(cls, *args, **kwargs): # 从预训练模型权重加载实例
        kwargs["_fast_init"] = False  # 复合模型（如 RAG）暂不支持快速初始化。
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: RagRetriever = None,
        **kwargs,
    ) -> PreTrainedModel:
        # 从 kwargs 提取与 question_encoder 和 generator 相关的配置
        kwargs_question_encoder = {
            argument[len("question_encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("question_encoder_")
        }

        kwargs_generator = {
            argument[len("generator_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("generator_")
        }
        # 删除提取的参数，防止冲突
        for key in kwargs_question_encoder.keys():
            del kwargs["question_encoder_" + key]
        for key in kwargs_generator.keys():
            del kwargs["generator_" + key]

        # 加载 question_encoder 和 generator
        question_encoder = kwargs_question_encoder.pop("model", None)

        # 检查是否提供了 question_encoder 和 generator 的预训练模型路径，使用 AutoModel 和 AutoModelForSeq2SeqLM 动态加载指定的模型
        if question_encoder is None:
            assert question_encoder_pretrained_model_name_or_path is not None, (
                "If `model` is not defined as an argument, a `question_encoder_pretrained_model_name_or_path` has to"
                " be defined"
            )
            from ..auto.modeling_auto import AutoModel

            if "config" not in kwargs_question_encoder:
                from ..auto.configuration_auto import AutoConfig

                question_encoder_config, kwargs_question_encoder = AutoConfig.from_pretrained(
                    question_encoder_pretrained_model_name_or_path,
                    **kwargs_question_encoder,
                    return_unused_kwargs=True,
                )
                kwargs_question_encoder["config"] = question_encoder_config

            question_encoder = AutoModel.from_pretrained(
                question_encoder_pretrained_model_name_or_path, **kwargs_question_encoder
            )

        generator = kwargs_generator.pop("model", None)
        if generator is None:
            assert generator_pretrained_model_name_or_path is not None, (
                "If `generator_model` is not defined as an argument, a `generator_pretrained_model_name_or_path` has"
                " to be defined"
            )
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            if "config" not in kwargs_generator:
                from ..auto.configuration_auto import AutoConfig

                generator_config, kwargs_generator = AutoConfig.from_pretrained(
                    generator_pretrained_model_name_or_path, **kwargs_generator, return_unused_kwargs=True
                )

                kwargs_generator["config"] = generator_config

            generator = AutoModelForSeq2SeqLM.from_pretrained(
                generator_pretrained_model_name_or_path, **kwargs_generator
            )

        # 若未提供配置对象，则基于 question_encoder 和 generator 的配置创建新的 RagConfig
        config = kwargs.get("config", None)
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        return cls(question_encoder=question_encoder, generator=generator, config=config, retriever=retriever)

class RagModel(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None, # 模型的配置对象（如 RagConfig），包含所有关键的超参数
        question_encoder: Optional[PreTrainedModel] = None, # 用于编码问题（输入查询）的编码器模型
        generator: Optional[PreTrainedModel] = None, # 用于生成自然语言输出的生成器模型
        retriever: Optional[RagRetriever] = None,  # or maybe just use a `set_retriever(...)` method
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an question_encoder and a generator has to be provided."

        if config is None: # 如果未提供 config，通过问题编码器和生成器的配置自动生成一个
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super().__init__(config)
        if question_encoder is None: # 默认使用 AutoModel 加载
            from ..auto.modeling_auto import AutoModel

            question_encoder = AutoModel.from_config(config.question_encoder)

        if generator is None: # 默认使用 AutoModelForSeq2SeqLM 加载
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            generator = AutoModelForSeq2SeqLM.from_config(config.generator)

        self.retriever = retriever # 必须是 RagRetriever 的实例，负责检索相关上下文
        if self.retriever is not None:
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

        self.ctx_encoder = None  # 用于选择是否训练上下文编码器
        self.context_encoder_training = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, # 表示输入问题（如查询）的标记化序列和对应的注意力掩码
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 预先计算好的编码器输出，减少重复计算
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        doc_scores: Optional[torch.FloatTensor] = None, # 每个检索文档的相关性分数
        context_input_ids: Optional[torch.LongTensor] = None, # 检索到的文档的标记化结果及其掩码
        context_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        n_docs: Optional[int] = None, # 检索文档的数量
    ) -> Union[Tuple[torch.Tensor], RetrievAugLMOutput]:
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved

        # whether retriever has to be used 
        has_to_retrieve = (
            self.retriever is not None
            and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
            and encoder_outputs is None
        )
        # encoder_outputs are pre-computed during RAG-token generation
        # 如果未提供 encoder_outputs 且需要检索文档，先用 question_encoder 对问题进行编码
        # 调用 retriever 对问题编码向量进行检索
        if encoder_outputs is None:
            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )
                question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder

                retriever_outputs = self.retriever(
                    input_ids,
                    question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    return_tensors="pt",
                )
                if self.context_encoder_training:
                    (
                        context_input_ids,
                        context_attention_mask,
                        retrieved_doc_embeds,
                        retrived_doc_input_ids,
                        retrived_doc_attention_mask,
                        retrieved_doc_ids,
                    ) = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["tokenized_doc_ids"],
                        retriever_outputs["tokenized_doc_attention_mask"],
                        retriever_outputs["doc_ids"],
                    )
                    # 返回检索文档的标记化数据（context_input_ids 等）及嵌入
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)

                    retrived_doc_input_ids = retrived_doc_input_ids.to(input_ids)
                    retrived_doc_attention_mask = retrived_doc_attention_mask.to(input_ids)
                    retrieved_doc_embeds = self.ctx_encoder(
                        retrived_doc_input_ids, attention_mask=retrived_doc_attention_mask, return_dict=True
                    ).pooler_output
                    retrieved_doc_embeds = retrieved_doc_embeds.view(
                        -1, n_docs, question_encoder_last_hidden_state.shape[1]
                    )  # reshaping

                    # compute doc_scores involving ctx_encoder 使用 torch.bmm 计算问题编码与文档嵌入的相关性分数
                    doc_scores = torch.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)

                else:
                    context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["doc_ids"],
                    )

                    # set to correct device
                    retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)

                    # compute doc_scores
                    doc_scores = torch.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)
            else:
                assert context_input_ids is not None, (
                    "Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can"
                    " set a retriever using the `set_retriever(...)` function."
                )
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

        assert (
            doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert (doc_scores.shape[1] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {context_input_ids.shape[0]}."
        )

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        # 使用上下文文档及问题的编码结果作为输入，通过 generator 生成最终输出
        gen_outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            # don't output retrieved docs
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return RetrievAugLMOutput( # 包含生成结果、检索信息（如文档得分、嵌入）和中间层的注意力信息等
            logits=gen_outputs.logits,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values, 
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state,
            generator_enc_hidden_states=gen_outputs.encoder_hidden_states,
            generator_enc_attentions=gen_outputs.encoder_attentions,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
            generator_cross_attentions=gen_outputs.cross_attentions,
        )


class RagSequenceForGeneration(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        super().__init__(config)

        # instantiate model
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        exclude_bos_score: Optional[bool] = None,
        reduce_loss: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        n_docs: Optional[int] = None,
        **kwargs,  # needs kwargs for generation
    ) -> RetrievAugLMMarginOutput:
        
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        exclude_bos_score = exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        if labels is not None:
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                decoder_input_ids,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                exclude_bos_score=exclude_bos_score,
                n_docs=n_docs,
            )

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=outputs.logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None,
        do_deduplication: Optional[bool] = None,  # defaults to True
        num_return_sequences: Optional[int] = None,  # defaults to 1
        num_beams: Optional[int] = None,  # defaults to 1
        n_docs: Optional[int] = None,
        **model_kwargs,
    ) -> torch.LongTensor:

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        assert (
            input_ids is not None or context_input_ids is not None
        ), " At least one of input_ids or context_input_ids must be given"

        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            context_input_ids = self.retriever(
                input_ids,
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="pt",
            )["context_input_ids"]

            # set to correct device
            context_input_ids = context_input_ids.to(input_ids)

        hypos = []
        model_kwargs["num_beams"] = num_beams
        model_kwargs["num_return_sequences"] = num_beams
        model_kwargs["attention_mask"] = None

        batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs

        for index in range(batch_size):
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)

            output_sequences = self.generator.generate(
                generator_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len
            if do_deduplication:
                # do_deduplication, max_output_len
                output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))

            num_candidates = output_sequences.shape[
                0
            ]  # after deduplication, this number can be less than n_docs*n_beam

            # then, run model forwards to get nll scores:
            if input_ids is not None:
                new_input_ids = input_ids[index : index + 1].repeat(num_candidates, 1)
                outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
            else:  # input_ids is None, need context_input_ids/mask and doc_scores
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

                individual_input_ids = generator_input_ids.repeat(
                    num_candidates, 1
                )  # (num_candidates*n_docs, max_len)

                individual_attention_mask = context_attention_mask[index * n_docs : (index + 1) * n_docs]
                individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)

                individual_doc_scores = doc_scores[index : (index + 1), :]  # doc_scores.shape = [batch, n_docs]
                individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)  # [num_candidates, n_docs]

                outputs = self(
                    context_input_ids=individual_input_ids,
                    context_attention_mask=individual_attention_mask,
                    doc_scores=individual_doc_scores,
                    labels=output_sequences,
                    exclude_bos_score=True,
                )

            top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

            # add hypothesis
            hypos.append(output_sequences[top_cand_inds])

        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)

    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
    ):
        # shift tokens left
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # bos_token_id is None for T5
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # seq_logits dim = (batch*n_docs, tgt_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x #vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        # calculate loss
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # sum over tokens, exclude bos while scoring
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        output = (
            tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
        )
        ind = 0
        for t in tensors:
            output[ind : ind + t.shape[0], : t.shape[1]] = t
            ind += t.shape[0]
        return output

