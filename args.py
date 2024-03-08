import numpy as np
from dataclasses import dataclass

@dataclass
class Args:
    raw_file_dirs: tuple = (
        "./data/raw_data_1/",
        "./data/raw_data_2/"
    )
    label_file_dir: str = "./data/processed_data/"
    processed_file_dir: str = "./data/processed_data/"
    preprocess: bool = False
    max_sequence_length: int = 128
    id_col: str = "patient_id"
    datetime_col: str = "start_date"
    location_col: str = "location_name"
    aggregation_freq: str = "1D"
    inner_aggregation_freq: str = "20min" # "auto" or a pd.Grouper freq
    location_list_col: str = "location_list"
    location_str_col: str = "location_str"
    no_location_str: str = "Nowhere"
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    fine_tune_location: str = "./fine_tuned_model"
    batch_size: int = 256
    seed: int = 42
    train_fraction: float = 0.8
    fine_tune_required_days_threshold: int = 30

    def __post_init__(self):

        # setting rng
        self.rng = np.random.default_rng(self.seed)
        
        # setting inner_aggregation_freq if auto
        if self.inner_aggregation_freq == "auto":
            minutes_in_day = 60*24
            self.inner_aggregation_freq = f"{int(np.ceil(minutes_in_day/self.max_sequence_length))}min"