import os
import warnings
from typing import List, Optional

from ...tokenization_utils_base import BatchEncoding  # Import BatchEncoding to encapsulate tokenized results
from ...utils import logging  # Import logging utilities from Transformers
from .configuration_rag import RagConfig  # Import RagConfig class to handle configuration for the RAG model

logger = logging.get_logger(__name__)  # Initialize a logger for the current module

class RagTokenizer:
    def __init__(self, question_encoder, generator):
        self.question_encoder = question_encoder  # Tokenizer for processing input questions
        self.generator = generator  # Tokenizer for processing output (answers)
        self.current_tokenizer = self.question_encoder  # Set the default tokenizer to question_encoder

    def save_pretrained(self, save_directory):
        """
        Save the tokenizers for both the question encoder and generator to the specified directory.
        """
        if os.path.isfile(save_directory):  # Ensure the path provided is a directory, not a file
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)  # Create the directory if it does not exist

        # Define the save paths for question_encoder and generator tokenizers
        question_encoder_path = os.path.join(save_directory, "question_encoder_tokenizer")
        generator_path = os.path.join(save_directory, "generator_tokenizer")

        # Save the respective tokenizers to their paths
        self.question_encoder.save_pretrained(question_encoder_path)
        self.generator.save_pretrained(generator_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load tokenizers for question_encoder and generator from a pretrained model or path.
        """
        from ..auto.tokenization_auto import AutoTokenizer  # Dynamically import AutoTokenizer for automatic selection

        config = kwargs.pop("config", None)  # Extract the config from kwargs if provided

        if config is None:  # If no config is provided, load it using RagConfig
            config = RagConfig.from_pretrained(pretrained_model_name_or_path)

        # Load question_encoder tokenizer based on the config and subfolder
        question_encoder = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.question_encoder, subfolder="question_encoder_tokenizer"
        )
        # Load generator tokenizer based on the config and subfolder
        generator = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.generator, subfolder="generator_tokenizer"
        )
        # Return a new RagTokenizer instance with loaded tokenizers
        return cls(question_encoder=question_encoder, generator=generator)

    def __call__(self, *args, **kwargs):
        """
        Use the current tokenizer to tokenize the input. Supports dynamic switching between tokenizers.
        """
        return self.current_tokenizer(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        Decode a batch of tokenized sequences back into text using the generator tokenizer.
        """
        return self.generator.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        Decode a single tokenized sequence back into text using the generator tokenizer.
        """
        return self.generator.decode(*args, **kwargs)

    def _switch_to_input_mode(self):
        """
        Switch the current tokenizer to question_encoder for processing input questions.
        """
        self.current_tokenizer = self.question_encoder

    def _switch_to_target_mode(self):
        """
        Switch the current tokenizer to generator for processing target sequences (answers).
        """
        self.current_tokenizer = self.generator

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepare a batch of inputs and optional targets for seq2seq tasks.

        Args:
            src_texts: List of source input texts (e.g., questions or prompts).
            tgt_texts: Optional list of target texts (e.g., answers or responses).
            max_length: Optional maximum length for source tokenization.
            max_target_length: Optional maximum length for target tokenization.
            padding: Padding strategy ('longest', 'max_length', or 'do_not_pad').
            return_tensors: Format of the returned tensors ('pt', 'tf', or 'np').
            truncation: Whether to truncate sequences that exceed max_length.
            **kwargs: Additional arguments passed to the tokenizer.
        
        Returns:
            A BatchEncoding object containing tokenized inputs and optional labels.
        """
        # Determine default max lengths if not provided
        max_length = max_length or self.current_tokenizer.model_max_length
        max_target_length = max_target_length or self.current_tokenizer.model_max_length

        # Tokenize source texts
        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )

        # If no target texts are provided, return only the tokenized inputs
        if tgt_texts is None:
            return model_inputs

        # Tokenize target texts with the generator tokenizer
        with self.generator.as_target_tokenizer():
            labels = self.generator(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                max_length=max_target_length,
                padding=padding,
                truncation=truncation,
                **kwargs,
            )

        # Add the tokenized target sequences as labels
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
