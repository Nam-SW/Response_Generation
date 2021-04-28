from typing import List

from utils import filtering


class CashMannager:
    def __init__(self, tokenizer):
        self.bos = tokenizer.bos_token
        self.eos = tokenizer.eos_token
        self.sep = tokenizer.sep_token

        self._massage_logs = dict()
        self._latest_request_time_dict = dict()

    def get_messages(self, channel_id) -> List[str]:
        return self._massage_logs.get(
            channel_id, {"messages": [], "last_massage_id": None}
        )

    def add_message(self, message):
        text = filtering(message.content)
        author_id = message.author.id
        channel_id = message.channel.id

        message_dict = self.get_messages(channel_id)

        if author_id == message_dict["last_massage_id"]:
            pass

    def check_last_one(self, massage) -> bool:
        pass

    def clear_cash(self, channel_id):
        pass
