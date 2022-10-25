import os
from preprocessing.dataframe import DataFrameCreator


def main():
    output_dir = "data/csv/"
    input_dir = "data/text/"

    with open("texts.txt", "r") as f:
        text_names = [text.replace("\n", "") for text in f.readlines()]

    with open("models.txt", "r") as f:
        models = [model.replace("\n", "") for model in f.readlines()]
        print(models)

    for text in text_names:
        with open(f"{input_dir}{text}.txt", "r") as f:
            text_data = f.readlines()

        for model in models:
            filename = f"{output_dir}{text}_{model}.csv"
            if os.path.isfile(filename):
                continue

            print(model)
            df_creator = DataFrameCreator(model)
            data = df_creator.create_dataframe_from_texts(text_data)
            data.to_csv(filename, index=False)
            del df_creator


if __name__ == "__main__":
    main()
