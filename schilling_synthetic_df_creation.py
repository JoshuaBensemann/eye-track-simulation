import os

from preprocessing.dataframe import DataFrameCreator


def main():
    data_dir = "data/"

    with open("data/schilling et al.txt", "r") as f:
        texts = f.readlines()

    with open("models.txt", "r") as f:
        models = [model.replace("\n", "") for model in f.readlines()]
        print(models)

    for model in models:
        filename = f"{data_dir}schilling_{model}.csv"
        if os.path.isfile(filename):
            continue

        print(model)
        df_creator = DataFrameCreator(model)
        data = df_creator.create_dataframe_from_texts(texts)
        data.to_csv(filename, index=False)
        del df_creator


if __name__ == "__main__":
    main()
