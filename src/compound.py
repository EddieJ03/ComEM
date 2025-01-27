from pathlib import Path
from typing import Literal

import pandas as pd
from rich import print
from sklearn.metrics import classification_report, confusion_matrix

from comparing_sq import ComparingSQ
from matching_sq import MatchingSQ
from selecting import Selecting

RANKING_STRATEGY = "matching"

class ComEM:
    ranking_strategy: Literal["matching", "comparing"] = "matching"

    def __init__(
        self,
        ranking_model_name: str = "flan-t5-xl",
        selecting_model_name: str = "gpt-4o-mini",
        ranking_strategy: Literal["matching", "comparing"] = ranking_strategy,
    ):
        self.ranking_model_name = ranking_model_name
        self.selecting_model_name = selecting_model_name
        if self.ranking_strategy == "matching":
            self.ranker = MatchingSQ(model_name=ranking_model_name)
        elif self.ranking_strategy == "comparing":
            self.ranker = ComparingSQ(model_name=ranking_model_name)
        self.selector = Selecting(model_name=selecting_model_name)

    def __call__(self, instance, threshold: float = 0.5,  topK: int = 1) -> list[bool]:
        if self.ranking_strategy == "matching":
            indexes = self.ranker.pointwise_rank(instance)
        elif self.ranking_strategy == "comparing":
            indexes = self.ranker.pairwise_rank(instance, topK=topK)

        indexes_k = [idx for _, idx in indexes[:topK]]

        indexes_threshold = [idx for score, idx in indexes if score >= threshold]

        preds = [False] * len(instance["candidates"])
        
        indexes = indexes_threshold if len(indexes_threshold) > len(indexes_k) else indexes_k

        instance_k = {
            "anchor": instance["anchor"],
            "candidates": [instance["candidates"][idx] for idx in indexes],
        }
        preds_k = self.selector(instance_k)
        for i, pred in enumerate(preds_k):
            preds[indexes[i]] = pred

        assert len(preds) == len(instance["candidates"]), "Prediction length mismatch"

        return preds

    @property
    def cost(self):
        return self.selector.cost
    
    @property
    def prompt_tokens(self):
        return self.selector.prompt_tokens
    
    @property
    def completion_tokens(self):
        return self.selector.completion_tokens

    @cost.setter
    def cost(self, value: int):
        self.selector.cost = value


if __name__ == "__main__":
    results = {}
    dataset_files = sorted(Path("data/llm4em").glob("*.csv"))
    compound = ComEM()
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

        preds_lst = [compound(it, threshold=0.5, topK=4) for it in instances]

        preds = [pred for preds in preds_lst for pred in preds]
        labels = [label for it in instances for label in it["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4))
        print(confusion_matrix(labels[: len(preds)], preds))

        report = classification_report(labels[: len(preds)], preds, output_dict=True)
        results[dataset] = report['1']
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
    
    print(f"Cost: {compound.cost:.2f}")
    print(f"Completion Tokens: {compound.completion_tokens:.2f}")
    print(f"Prompt Tokens: {compound.prompt_tokens:.2f}")
