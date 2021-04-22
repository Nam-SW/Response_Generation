from typing import List


class MassageCash:
    def __init__(self):
        self._massage_queue = dict()
        self._latest_request_time_dict = dict()
        
    def get_messages(self, channel_id) -> List[str]:
        pass

    def add_message(self, channel_id, author, message):
        pass

    def check_time(self, channel_id) -> bool:
        pass

    def clear_cash(self, channel_id):
        pass
