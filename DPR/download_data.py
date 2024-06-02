import os
import wget
import pandas as pd
import json
import argparse

def download(DATA_PATH):
    """download training data"""
    url = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz'
    wget.download(url, out=DATA_PATH)

def save_dataset(ddta,path):
    with open(path,'w') as json_file:
        json.dump(ddta,json_file)

def sample_data(DATA_PATH):
    "sample data"
    df = pd.read_json(DATA_PATH)
    df = df.drop(columns=['dataset','answers'])
    df['positive_ctxs'].apply(lambda x:[j.pop(i) for j in x for i in ['score', 'title_score', 'passage_id']])
    df['negative_ctxs'].apply(lambda x:[j.pop(i) for j in x for i in ['score', 'title_score', 'passage_id']])
    df['hard_negative_ctxs'].apply(lambda x:[j.pop(i) for j in x for i in ['score', 'title_score', 'passage_id']])

    df_sample = df.sample(30_000)
    sample_df = list(df_sample.T.to_dict().values())

    save_dataset(sample_df,DATA_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    download(args.data_dir)
    sample_data(args.data_dir)

if __name__ == "__main__":
    main()