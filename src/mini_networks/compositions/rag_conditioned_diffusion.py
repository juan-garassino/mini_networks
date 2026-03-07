"""RAG-conditioned diffusion: retrieve/generate prompt then guide diffusion."""
from __future__ import annotations

from mini_networks.core.logging.logger import Logger
from mini_networks.compositions.rag_guided_generation import (
    RAGGuidedGeneration,
    RAGGuidedGenerationConfig,
)
from mini_networks.compositions.clip_guided_diffusion import (
    CLIPGuidedDiffusion,
    CLIPGuidedDiffusionConfig,
)


class RAGConditionedDiffusionConfig(CLIPGuidedDiffusionConfig):
    model_name: str = "rag_conditioned_diffusion"
    prompt_seed: str = "To be or not to be"


class RAGConditionedDiffusion:
    def __init__(self):
        self.rag = RAGGuidedGeneration()
        self.diff = CLIPGuidedDiffusion()

    def train(self, config: RAGConditionedDiffusionConfig, logger: Logger) -> None:
        rag_cfg = RAGGuidedGenerationConfig(
            fast_demo=config.fast_demo,
            data_root=config.data_root,
            device=config.device,
        )
        self.rag.train(rag_cfg, logger)
        self.diff.train(config, logger)

    def sample(self, config: RAGConditionedDiffusionConfig) -> tuple:
        prompt = self.rag.generate(
            RAGGuidedGenerationConfig(fast_demo=True, data_root=config.data_root, device=config.device),
            config.prompt_seed,
            max_new_tokens=16,
        )
        images = self.diff.text_to_image(prompt, config)
        return images, prompt
