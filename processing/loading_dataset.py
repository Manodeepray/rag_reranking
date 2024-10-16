from datasets import load_dataset


def load_docs():
    corpus = load_dataset(path= "BeIR/nq", name = "corpus" ,data_dir = 'datasets' )


    queries = load_dataset(path = "BeIR/nq", name = "queries" , data_dir = 'datasets')
    # Load Natural Questions dataset from HuggingFace
    # dataset = load_dataset("natural_questions", split='train[:1%]')  # For example, take 1% of the training data

    # Check sample data from the dataset
    # This prints one example from the dataset
    return corpus , queries