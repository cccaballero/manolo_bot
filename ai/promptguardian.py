import logging
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from config import Config


class PromptGuardian:
    """Guardian for prompt injection detection. Based on https://huggingface.co/ProtectAI/deberta-v3-small-prompt-injection-v2"""

    safe_labels = ["SAFE"]
    unsafe_labels = ["INJECTION"]

    def __init__(self, config: Config):
        self.config = config

        tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-small-prompt-injection-v2")
        model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-small-prompt-injection-v2")

        self.classifier = pipeline(
            "text-classification", model=model, tokenizer=tokenizer, truncation=True, max_length=512
        )

    def get_safe_labels(self):
        return self.safe_labels

    def get_unsafe_labels(self):
        return self.unsafe_labels

    def classify(self, text: str) -> str:
        start_time = time.time()
        result = self.classifier(text)
        end_time = time.time()
        logging.debug(f"Guardian inference time: {end_time - start_time} seconds")
        return result[0].get("label")
