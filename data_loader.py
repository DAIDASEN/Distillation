
from datasets import load_dataset

def load_deepscaler(split: str = "train"):
    """Load DeepScaleR dataset"""
    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split=split)
    return ds

def iter_deepscaler_batches(ds, batch_size: int):
    """
    Iterator for DeepScaleR batches
    Yields: (questions: list[str], answers: list[str])
    """
    n = len(ds)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = ds.select(range(start, end))
        questions = [ex["problem"] for ex in batch]
        answers = [ex["answer"] for ex in batch] # Clean answers
        yield questions, answers
