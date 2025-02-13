import torch
from base_models import BayesRewardModel
from transformers import (
    AutoModel, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sal.models.reward_models import PRM
from sal.config import Config


class BayesPRM(PRM):
    def __init__(
            self,
            model: BayesRewardModel,
            config: Config,
        ):
        super().__init__()
        self.model = model

    def load_model_and_tokenizer(self, **model_kwargs) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        emb_model_path = "Alibaba-NLP/gte-modernbert-base"
        emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_path)
        emb_model = AutoModel.from_pretrained(
            emb_model_path,
            device_map="cuda:1",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        return emb_model, emb_tokenizer

    def score(
        self, 
        questions: list[str], 
        outputs: list[list[str]], 
        batch_size: int=8,
    ):
        conversations = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                conversations.append(conversation)

        # Generate PRM predictions
        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                acquisitions = self.model.pred_acq(inputs_batch)
            output_scores.append(acquisitions)

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)
        return reshaped_output_scores