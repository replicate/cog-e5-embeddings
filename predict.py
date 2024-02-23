import os
import time
import subprocess
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from cog import BasePredictor, Input


MODEL_URL = "https://weights.replicate.delivery/default/Intfloat/e5-mistral-7b-instruct-cache.tar"


def download_weights(url, dest):
    start = time.time()
    subprocess.check_call(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    instruction = f"Instruct: {task_description}\nQuery: {query}"
    return instruction


class Predictor(BasePredictor):
    def setup(self) -> None:
        if not os.path.exists("e5-mistral-7b-instruct-cache"):
            download_weights(MODEL_URL, "./e5-mistral-7b-instruct-cache")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("e5-mistral-7b-instruct-cache")
        self.model = AutoModel.from_pretrained(
            "e5-mistral-7b-instruct-cache",
            torch_dtype=torch.float16,
            device_map=self.device,
        )

    def predict(
        self,
        task: str = Input(description="The task description.", default=None),
        query: str = Input(description="The query to be used.", default=None),
        document: str = Input(description="The document to be used."),
        normalize: bool = Input(
            description="Whether to output the normalized embeddings or not.",
            default=False,
        ),
    ) -> Dict:
        max_length = 4096

        if task != None and query != None:
            queries = [get_detailed_instruct(task, query)]
            documents = [document]

            input_texts = queries + documents

            # tokenize the input texts
            batch_dict = self.tokenizer(
                input_texts,
                max_length=max_length - 1,
                return_attention_mask=False,
                padding=False,
                truncation=True,
            )

            batch_dict["input_ids"] = [
                input_ids + [self.tokenizer.eos_token_id]
                for input_ids in batch_dict["input_ids"]
            ]
            batch_dict = self.tokenizer.pad(
                batch_dict,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            # move inputs to gpu
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

            output_embeddings = [
                embedding for embedding in embeddings.cpu().detach().numpy()
            ]

            # normalize embeddings
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            if normalize:
                output_embeddings.extend(
                    [
                        embedding
                        for embedding in normalized_embeddings.cpu().detach().numpy()
                    ]
                )
            scores = (normalized_embeddings[:1] @ normalized_embeddings[1:].T) * 100

            output_dict = {
                "embeddings": output_embeddings,
                "score": scores.tolist()[0][0],
            }

            return output_dict
        else:
            input_texts = [document]

            # tokenize the input texts
            batch_dict = self.tokenizer(
                input_texts,
                max_length=max_length - 1,
                return_attention_mask=False,
                padding=False,
                truncation=True,
            )

            batch_dict["input_ids"] = [
                input_ids + [self.tokenizer.eos_token_id]
                for input_ids in batch_dict["input_ids"]
            ]
            batch_dict = self.tokenizer.pad(
                batch_dict,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            # move inputs to gpu
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

            output_embeddings = [
                embedding for embedding in embeddings.cpu().detach().numpy()
            ]

            # normalize embeddings
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            if normalize:
                output_embeddings.extend(
                    [
                        embedding
                        for embedding in normalized_embeddings.cpu().detach().numpy()
                    ]
                )

            output_dict = {"embeddings": output_embeddings, "score": None}
            return output_dict
