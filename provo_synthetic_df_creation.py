from preprocessing.dataframe import DataFrameCreator


def main():
    with open("data/provo.txt", "r") as f:
        texts = f.readlines()

    for model in ["bert-base-uncased", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        print(model)
        df_creator = DataFrameCreator(model)
        data = df_creator.create_dataframe_from_texts(texts)
        data.to_csv(f"data/provo_{model}.csv", index=False)
        del df_creator


if __name__ == "__main__":
    main()
