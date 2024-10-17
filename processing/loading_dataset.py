from datasets import load_dataset


def load_docs():
    corpus = load_dataset(path= "BeIR/nq", name = "corpus" )


    queries = load_dataset(path = "BeIR/nq", name = "queries")
    #new_corpus = []
    #new_query = []
    new_corpus = corpus['corpus'][:10000]
    new_query = queries['queries'][:10000]
    # Load Natural Questions dataset from HuggingFace
    # dataset = load_dataset("natural_questions", split='train[:1%]')  # For example, take 1% of the training data

    # Check sample data from the dataset
    # This prints one example from the dataset
    return new_corpus , new_query

if __name__ == "__main__":
    
    corpus , queries = load_docs()
    #print(len(corpus['_id']),len(corpus['title']),len(corpus['text']))
    #print(queries)