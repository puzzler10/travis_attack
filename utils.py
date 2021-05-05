def show_random_elements(dataset, num_examples=10):
    """Print some elements in a nice format so you can take a look at them. Use for a dataset from the `datasets` package.  """
    import datasets
    import random
    import pandas as pd
    from IPython.display import display, HTML

    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
