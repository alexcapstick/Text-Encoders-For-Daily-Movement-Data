import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers import losses
from sentence_transformers import evaluation

from args import Args
import data_processing


args = Args()


print("Loading data...")
start_time = time.time()


if not Path(args.processed_file_dir + 'activity.parquet').exists():
    data_processing.load_and_process_raw_data_files(args).to_parquet(
        args.processed_file_dir + 'activity.parquet'
    )

if args.preprocess:
    dataset = data_processing.aggregate_locations(
        pd.read_parquet(args.processed_file_dir + 'activity.parquet'), 
        args
    )
    dataset.to_parquet(args.processed_file_dir + "processed_activity.parquet")
else:
    dataset = pd.read_parquet(args.processed_file_dir + "processed_activity.parquet")

sentences = data_processing.process_day_str_from_locations(
    dataset, args
)

print(f"Data loaded in {time.time()-start_time:.2f} seconds.")

print("Loading model...")
start_time = time.time()
model = SentenceTransformer(args.model_name)
print(f"Model loaded in {time.time()-start_time:.2f} seconds.")

sentences = (
    sentences
    [[args.id_col, args.datetime_col, args.location_str_col]]
)


class RandomTriplets(torch.utils.data.IterableDataset):
    def __init__(self, dataset, args, n_pairs=10000):

        dataset = (   
            # keep all ids that have at least 2 rows
            dataset
            .groupby(args.id_col)
            .filter(lambda x: len(x) > 1)
            .sort_values(args.datetime_col)
            # make sure that all dates for each id have at least one other date within 30 days
            .groupby(args.id_col, group_keys=False)
            .apply(
                lambda x: x.loc[
                    (
                        x[args.datetime_col].diff().abs().dt.days 
                        < args.fine_tune_required_days_threshold
                    ) | (
                        x[args.datetime_col].diff(-1).abs().dt.days 
                        < args.fine_tune_required_days_threshold
                    )
                ]
            )
            .set_index([args.id_col, args.datetime_col])
            .sort_index()
        )

        self.dataset = dataset
        self.args = args
        self.n_pairs = n_pairs
        self.ids = dataset.index.get_level_values(0).unique().tolist()

    def generate_random_pairs(self, dataset, N):

        def random_sample_from_dates_within_threshold(sample_dates, other_dates, args):
            dates_diff = np.abs(
                sample_dates.reshape(-1, 1) 
                - other_dates.reshape(1, -1)
            )
            dates_diff_threshold = dates_diff < np.timedelta64(args.fine_tune_required_days_threshold, 'D')

            random_idx = np.argmax(
                dates_diff_threshold * self.args.rng.integers(0, 100, size=dates_diff_threshold.shape),
                axis=1
            )

            return pd.DataFrame(
                {
                    "date": sample_dates,
                    "date_positive": other_dates[random_idx],
                }
            )

        def random_different_id_and_date(sample_id, other_ids, other_dates, n_examples):
            
            other_bool = other_ids != sample_id
            other_ids = other_ids[other_bool]
            other_dates = other_dates[other_bool]
            
            random_idx = self.args.rng.choice(len(other_ids), size=n_examples, replace=True) 

            return pd.DataFrame(
                {
                    "pateint_id_negative": other_ids[random_idx],
                    "date_negative": other_dates[random_idx],
                }
            )
        
        other_ids = dataset.index.get_level_values(0)
        other_dates = dataset.index.get_level_values(1)

        random_sample = (
            dataset
            .sample(n=N, replace=True, random_state=self.args.rng)
            .reset_index()
            .groupby(self.args.id_col)
            .apply(
                lambda df: pd.concat(
                    [

                        random_sample_from_dates_within_threshold(
                            df['start_date'].values, 
                            dataset.loc[df.name].index.values,
                            args=self.args
                        ),

                        random_different_id_and_date(
                            df.name,
                            other_ids,
                            other_dates,
                            n_examples=len(df)
                        )

                    ],
                    axis=1
                )
            )
            .droplevel(-1)
            .reset_index()
            .itertuples()
        )
        return random_sample

    def get_evaluation(self, n_pairs=1000):
        random_sample = self.generate_random_pairs(self.dataset, N=n_pairs)
        anchors = []
        positives = []
        negatives = []
        for instance in random_sample:
            anchors.append(self.dataset.loc[instance.patient_id, instance.date][self.args.location_str_col])
            positives.append(self.dataset.loc[instance.patient_id, instance.date_positive][self.args.location_str_col])
            negatives.append(self.dataset.loc[instance.pateint_id_negative, instance.date_negative][self.args.location_str_col])
        return anchors, positives, negatives

    def __iter__(self):
        random_sample = self.generate_random_pairs(self.dataset, N=self.n_pairs)
        for instance in random_sample:
            yield InputExample(
                texts=[
                    self.dataset.loc[instance.patient_id, instance.date][self.args.location_str_col],
                    self.dataset.loc[instance.patient_id, instance.date_positive][self.args.location_str_col],
                    self.dataset.loc[instance.pateint_id_negative, instance.date_negative][self.args.location_str_col],
                ],
                label=1
            ) 

    def __len__(self):
        return self.n_pairs


train_ds = RandomTriplets(sentences, args=args, n_pairs=100000)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, )
train_loss = losses.TripletLoss(model=model)

num_epochs = 10
warmup_steps = 1000

print("Fine-tuning model...")
start_time = time.time()
model.fit(
    train_objectives=[(train_dl, train_loss)],
    epochs=num_epochs,
    evaluator=evaluation.TripletEvaluator(
        *train_ds.get_evaluation(n_pairs=10000),
        name="fine_tune_triplet_eval",
    ),
    warmup_steps=warmup_steps,
    output_path=args.fine_tune_location,
    checkpoint_path=args.fine_tune_location
) 
print(f"Model fine-tuned in {time.time()-start_time:.2f} seconds.")