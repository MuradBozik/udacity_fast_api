"""Script to train machine learning model.
Author  : Murad Bozik
Data    : 11DEC1712 2022
"""
import argparse
import logging
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_model
import pandas as pd
import hydra
import os
from pathlib import Path


root_folder = Path(os.path.abspath(__file__)).parent.parent
logging.basicConfig(filename=f'{root_folder}/output.log',
                    level=logging.INFO, format="%(asctime)-15s %(message)s",
                    filemode='w')
logger = logging.getLogger()


def go(args):
    # Load the data.
    data = pd.read_csv(args.data_path)
    logger.info("Data loaded!")

    # Optional enhancement, use K-fold cross validat ion instead of a
    # train-test split.
    train, test = train_test_split(
        data, test_size=args.test_size, random_state=args.random_seed)
    logger.info(
        f"Training data has {len(train)} and test data has {len(test)} rows")

    X_train, y_train, encoder, lb = process_data(
        train, label="salary", training=True
    )
    logger.info("Training data processed")

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    logger.info("Test data processed")

    with hydra.initialize(version_base=None, config_path="./ml", job_name="train_model"):
        cfg = hydra.compose(
            config_name="logreg_config",
            overrides=[f"random_state={args.random_seed}"])

    # Train and save a model.
    model = train_model(X_train=X_train, y_train=y_train, config=cfg)
    logger.info("Model training completed, Evaluation started on test set")

    pred_test = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, pred_test)
    logger.info("Overall performance on test set:\n" +
                f"precision: {precision}, recall:{recall}, f_beta: {fbeta}")

    logger.info("Looking into slice performances for categorical features")
    perform_on_slice(model, encoder, lb, test)
    logger.info("Slice performances saved in slice_output.txt")
    # Saving the model, encoder and label binarizer
    save_model(model, encoder, lb, name=args.output_model_name)


def perform_on_slice(model, encoder, lb, test):
    lines = []
    cat_features = test.select_dtypes("object").columns.to_list()
    for on in cat_features:
        lines.append(f"Slice {on}:\n")
        values = test.loc[:, on].unique()
        for value in values:
            test_slice = test.loc[test[on] == value]
            X_slice, y_slice, _, _ = process_data(test_slice,
                                                  categorical_features=cat_features,
                                                  label="salary", training=False,
                                                  encoder=encoder, lb=lb)
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            line = f"{value}:" + "\n" + \
                f"precision: {precision}, recall:{recall}, f_beta: {fbeta}" + "\n"
            lines.append(line)
        lines.append("\n")

    with open(f"{root_folder}/slice_output.txt", "w") as fp:
        fp.writelines(lines)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training and logging random forest model")

    parser.add_argument(
        "--data_path",
        default=f"{root_folder}/data/census_clean.csv",
        type=str,
        help="Path to the dataset. It will be split into train and test",
        required=False,
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Size of the test split. Fraction of the dataset, or number of items",
        required=False,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )
    parser.add_argument(
        "--output_model_name",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)
