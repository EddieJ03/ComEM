import re
from pathlib import Path

import pandas as pd
from diskcache import Cache
from jinja2 import Template
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from utils import APICostCalculator, openai_chat_complete


class Selecting:
    template = Template(
        """Select all records from the following candidates that refer to the same real-world entity as the given record. Answer with each corresponding record number surrounded by "[]" or "[0]" if there are none.

Given entity record:
{{ anchor }}

Candidate records:{% for candidate in candidates %}
[{{ loop.index }}] {{ candidate }}{% endfor %}
"""
    )

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        template: Template = template,
        use_cache: bool = True,
    ):
        self.model = model_name
        self.template = template

        self.api_cost_decorator = APICostCalculator(model_name=model_name)

        if use_cache:
            cache = Cache(f"results/diskcache/selecting_{model_name}")
            self.chat_complete = self.api_cost_decorator(
                cache.memoize(name="chat_complete")(openai_chat_complete)
            )
        else:
            self.chat_complete = self.api_cost_decorator(openai_chat_complete)


    def __call__(self, instance) -> list[bool]:
        response = self.chat_complete(
            messages=[
                {
                    "role": "user",
                    "content": self.template.render(
                        anchor=instance["anchor"],
                        candidates=instance["candidates"],
                    ),
                }
            ],
            model=self.model,
            seed=42,
            temperature=0.0,
            logprobs=self.model.startswith("gpt"),
            top_logprobs=3 if self.model.startswith("gpt") else None,
        )

        matches = re.findall(r"\[(\d+)\]", response.choices[0].message.content.strip())
        preds = [False] * len(instance["candidates"])

        if matches:
            for match in matches:
                idx = int(match)
                if 1 <= idx <= len(instance["candidates"]):
                    preds[idx - 1] = True

        return preds

    @property
    def cost(self):
        return self.api_cost_decorator.cost

    @cost.setter
    def cost(self, value: int):
        self.api_cost_decorator.cost = value

    @property
    def prompt_tokens(self):
        return self.api_cost_decorator.prompt_tokens

    @prompt_tokens.setter
    def prompt_tokens(self, value: int):
        self.api_cost_decorator.prompt_tokens = value

    @property
    def completion_tokens(self):
        return self.api_cost_decorator.completion_tokens

    @completion_tokens.setter
    def completion_tokens(self, value: int):
        self.api_cost_decorator.completion_tokens = value


if __name__ == "__main__":
    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
    selector = Selecting()
    for file in dataset_files:
        dataset = file.stem
        print(f"[bold magenta]{dataset}[/bold magenta]")
        df = pd.read_csv(file)

        groupby = list(
            df.groupby("id_left")[["record_left", "record_right", "label"]]
            .apply(lambda x: x.to_dict("list"))
            .to_dict()
            .items()
        )
        instances = [
            {
                "anchor": v["record_left"][0],
                "candidates": v["record_right"],
                "labels": v["label"],
            }
            for _, v in groupby
        ]

        preds_lst = thread_map(
            selector,
            instances,
            max_workers=16,
        )
        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4))
        print(confusion_matrix(labels[: len(preds)], preds))
        print(f"Cost: {selector.cost:.2f}")

        results[dataset] = classification_report(
            labels[: len(preds)], preds, output_dict=True
        )["True"]
        results[dataset].pop("support")
        for k, v in results[dataset].items():
            results[dataset][k] = v * 100

    results["mean"] = {
        "precision": sum(v["precision"] for v in results.values()) / len(results),
        "recall": sum(v["recall"] for v in results.values()) / len(results),
        "f1-score": sum(v["f1-score"] for v in results.values()) / len(results),
    }
    df = pd.DataFrame.from_dict(results, orient="index")
    print(df)
    print(df.to_csv(float_format="%.2f", index=False))
    print(f"{selector.cost:.2f}")
