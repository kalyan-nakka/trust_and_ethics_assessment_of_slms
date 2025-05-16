import os
import json

from tempfile import TemporaryDirectory
from chat import Chat


class MessageConstructor(object):
    def __init__(self, task_desc = None):
        self.task_desc = task_desc

    def get_message(self, text):
        messages = [{"role": "user", "content": text}]
        return messages


def main(args):
    all_responses = []

    # Read the json lines from the dataset file
    dataset = []
    with open(args.do_not_answer.data_dir + args.do_not_answer.dataset_file) as f:
        for line in f.readlines():
            dataset.append(json.loads(line))

    with TemporaryDirectory(dir="/home/kalyan/cache") as dirname:
        msg_constructor = MessageConstructor()
        language_model_client = Chat.from_helm(args, cache=dirname)

        for data_record in dataset:
            data_save_line = {
                "model": args.model_config.model,
                "prompt": data_record.get("question", None),
                "risk_area": data_record.get("risk_area", None),
                "types_of_harm": data_record.get("types_of_harm", None),
                "specific_harms": data_record.get("specific_harm", None),
            }

            prompt = data_record.get("question", "prompt")
            response = language_model_client.do_generation(dataset=[prompt],
                                                           message_constructor=msg_constructor,
                                                           n=args.do_not_answer.n,
                                                           t=args.do_not_answer.t,
                                                           max_tokens=args.do_not_answer.max_tokens,
                                                           dry_run=args.dry_run)

            reply = response[1][0][4]['choices'][0]['message']['content']
            data_save_line["response"] = reply

            all_responses.append(data_save_line)

        # Save the responses from LLMs
        os.makedirs(os.path.dirname(args.do_not_answer.out_file), exist_ok=True)
        with open(args.do_not_answer.out_file, "w") as f:
            for x in all_responses:
                f.write(json.dumps(x) + "\n")
