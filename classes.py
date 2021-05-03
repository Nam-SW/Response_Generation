from typing import List

from transformers import GPT2TokenizerFast, TFAutoModelForCausalLM

from utils import filtering


class CashMannager:
    def __init__(self, tokenizer):
        self.bos = tokenizer.bos_token
        self.eos = tokenizer.eos_token
        self.sep = tokenizer.sep_token

        self._message_logs = dict()
        self._latest_request_time_dict = dict()

    def _get_log_form(self):
        return {
            "messages": [],
            "last_author_id": None,
            "last_message_id": None,
        }

    def get_messages(self, message) -> List[str]:
        channel_id = message.channel.id
        return self._message_logs.get(channel_id, self._get_log_form())

    def add_message(self, message):
        # text = filtering(message.content)
        text = message.content
        author_id = message.author.id
        message_id = message.id
        channel_id = message.channel.id

        message_dict = self.get_messages(message)

        if author_id == message_dict["last_author_id"]:
            message_dict["messages"][-1] += f" {text}"

        else:
            message_dict["messages"].append(text)
            message_dict["last_author_id"] = author_id

        message_dict["last_message_id"] = message_id

        self._message_logs[channel_id] = message_dict

    def is_last_one(self, message) -> bool:
        channel_id = message.channel.id
        return message.id == self.get_messages(channel_id)["last_message_id"]

    def clear_cash(self, message):
        channel_id = message.channel.id
        del self._message_logs[channel_id]


class Predictor:
    def __init__(
        self,
        tokenizer_name: str,
        model_name: str,
        utterance_window: int = 4,
    ):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
