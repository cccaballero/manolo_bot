import logging
import time

from transformers import pipeline

from config import Config


class PromptGuardian:
    """Guardian for prompt jailbreak detection. Based on https://huggingface.co/meta-llama/Prompt-Guard-86M"""

    safe_labels = ["BENIGN", "INJECTION"]
    unsafe_labels = ["JAILBREAK"]

    def __init__(self, config: Config):
        self.config = config

        self.classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M", device="cpu")

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
