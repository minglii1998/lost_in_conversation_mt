from llms import generate_json
from utils import extract_conversation
from tasks import get_task
import json, random

class UserAgent:
    def __init__(self, task, model="gpt-4o"):
        self.model = model
        self.task = task
        self.task_name = task.get_task_name()
        with open("prompts/user_agent.txt", "r") as f:
            self.prompt_response = f.read()

    def generate_response(self, conversation, sample, temperature=1.0):
        num_user_msgs = sum(1 for msg in conversation if msg["role"] == "user")

        if self.task_name in ["translation", "summary", "data2text"]:
            return self.task.populate_lazy_prompt(sample, num_user_msgs)

        lazy_sample = sample["lazy_3step"] if "lazy_3step" in sample else sample # we should get rid of this irregularity
        if "lazy" in lazy_sample:
            lazy_sample = lazy_sample["lazy"]

        if num_user_msgs == 0:
            shard_id = -1
            if "shards" in lazy_sample:
                first_shard = lazy_sample["shards"][0]
                shard_id = first_shard["shard_id"]
                initial_query = first_shard["shard"]
            else: # remove once everything has "shards"
                initial_query = lazy_sample["initial_query"]

            return initial_query, shard_id, 0.0

        else: # Turn 2 ... n
            hint_revealed_ids = [msg["content"]["hint_id"] for msg in conversation if msg["role"] == "log" and msg["content"]["type"] == "hint_revealed"]
            if "shards" in sample:
                # new way
                shards = lazy_sample["shards"][1:] # filter out the first shard, which is the initial query
                shard_ids = [shard["shard_id"] for shard in shards]
                shard_ids_revealed = [shard_id for shard_id in shard_ids if shard_id in hint_revealed_ids]
                shard_ids_not_revealed = [shard_id for shard_id in shard_ids if shard_id not in hint_revealed_ids]

                shard_texts_revealed = [shard for shard in shards if shard["shard_id"] in shard_ids_revealed]
                shard_texts_not_revealed = [shard for shard in shards if shard["shard_id"] in shard_ids_not_revealed]
                shards_revealed_str = json.dumps(shard_texts_revealed)
                shards_not_revealed_str = json.dumps(shard_texts_not_revealed)

            else:
                hint_not_revealed_ids = [hint["hint_id"] for hint in lazy_sample["hints"] if hint["hint_id"] not in hint_revealed_ids]

                shards_revealed_str = json.dumps([hint for hint in lazy_sample["hints"] if hint["hint_id"] in hint_revealed_ids])
                shards_not_revealed_str = json.dumps([hint for hint in lazy_sample["hints"] if hint["hint_id"] in hint_not_revealed_ids])

            user_agent_prompt_populated = self.prompt_response.replace("[[CONVERSATION_SO_FAR]]", extract_conversation(conversation, to_str=True, skip_system=True)).replace("[[HINTS_REVEALED]]", shards_revealed_str).replace("[[HINTS_NOT_REVEALED]]", shards_not_revealed_str)
            response_obj = generate_json([{"role": "user", "content": user_agent_prompt_populated}], model=self.model, step="lazy-simulator-user-agent-response", timeout=100, return_metadata=True, temperature=temperature)
            response = response_obj["message"]

            return response["response"], response["hint_id"], response_obj["total_usd"]
