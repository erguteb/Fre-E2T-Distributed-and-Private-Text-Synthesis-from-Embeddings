import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
from typing import List, Union


class EmbeddingService:

    def __init__(
        self,
        model_name: str = None,
        normalize_embeddings=False,
        batch_size=64,
        device="cuda:0",
    ):
        self.model_name = model_name
        if self.model_name == "gtr-t5-base-no-norm":
            transformer = models.Transformer("sentence-transformers/gtr-t5-base")
            pooling = models.Pooling(
                transformer.get_word_embedding_dimension(), pooling_mode="mean"
            )
            self.model = SentenceTransformer(modules=[transformer, pooling])
            self.model.to(device)
        else:
            self.model = SentenceTransformer(self.model_name)

        self.batch_size = batch_size
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        self.normalize_embeddings = normalize_embeddings
        self.device = device

    def embed_text(
        self,
        text: Union[str, List[str]],
        convert_to_tensor=False,
    ) -> np.ndarray:
        return self.model.encode(
            text,
            batch_size=self.batch_size,
            show_progress_bar=True,
            device=self.device,
        )

    def get_dimension(self) -> int:
        return self.vector_dimension
