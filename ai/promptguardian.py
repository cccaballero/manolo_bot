import torch
from transformers import pipeline

from config import Config


class PromptGuardian:
    def __init__(self, config: Config):
        self.config = config

        self.classifier = pipeline(
            "text-classification",
            model=self.config.prompt_guardian_model,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def classify(self, text: str) -> str:
        result = self.classifier(text)
        return result[0].get("label")
