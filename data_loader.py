from datasets import load_dataset

def load_data():
    dataset = load_dataset("bentrevett/multi30k")

    train_data = dataset["train"][:1000]

    return train_data