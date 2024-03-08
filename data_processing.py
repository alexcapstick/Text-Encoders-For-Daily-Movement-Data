import pandas as pd
import numpy as np
import datetime
import os


def load_and_process_raw_data_files(args):
    """
    Loads and processes the raw data files from Minder.
    
    """

    file_paths = []
    for file_dir in args.raw_file_dirs:
        file_paths.extend([os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith(".parquet")])

    activity_file_paths = [f for f in file_paths if "activity" in f]
    sleep_file_paths = [f for f in file_paths if "sleep" in f]

    allowed_locations = [
        "bedroom1", 
        "hallway", 
        "bathroom1", 
        "kitchen", 
        "lounge",
        "living room", 
        "WC1",
        "bed_in",
        "bed_out",
    ]

    location_map = {
        'bedroom1': "Bedroom", 
        'hallway': "Hallway", 
        'bathroom1': "Bathroom", 
        'kitchen': "Kitchen", 
        'lounge': "Lounge",
        'living room': "Lounge", 
        'WC1': "Bathroom",
        'bed_in': "Sleep",
        'bed_out': "Wake",
    }

    activity_df = pd.concat([
        pd.read_parquet(
            f, 
            filters=[
                (args.datetime_col, ">=", datetime.datetime.fromisoformat("2021-07-01T00Z")),
                (args.datetime_col, "<", datetime.datetime.fromisoformat("2024-01-31T00Z")),
            ],
            columns=[args.id_col, args.datetime_col, args.location_col]
        ) 
        for f in activity_file_paths
    ])
    sleep_df = pd.concat([
        pd.read_parquet(
            f, 
            filters=[
                (args.datetime_col, ">=", datetime.datetime.fromisoformat("2021-07-01T00Z")),
                (args.datetime_col, "<", datetime.datetime.fromisoformat("2024-01-31T00Z")),
            ],
            columns=[args.id_col, args.datetime_col, "value"]
        ) 
        for f in sleep_file_paths
    ]).rename(columns={"value": args.location_col})

    activity_df = (
        pd.concat([activity_df, sleep_df])
        .assign(
            **{
                args.datetime_col : lambda df: pd.to_datetime(df[args.datetime_col]).dt.tz_localize(None)   
            }
        )
        .sort_values([args.id_col, args.datetime_col])
        .loc[lambda x: x[args.location_col].isin(allowed_locations)]
        .replace({args.location_col: location_map})
    )

    return activity_df



def aggregate_locations(df, args):
    """
    Aggregates location data to the specified frequency and fills in 
    missing time periods with the no location string.
    
    """

    return (
        df
        # groupby patient an day
        .groupby(
            [
                args.id_col, 
                pd.Grouper(
                    key=args.datetime_col, 
                    freq=args.inner_aggregation_freq, 
                    origin="start_day"
                )
            ]
        )
        # get the most common location for each inner time period
        [args.location_col]
        .agg(lambda x: x.mode().sample(1, random_state=args.rng))
        .to_frame(args.location_col)
        .reset_index()
        # replace empty time periods with no location str
        .groupby(args.id_col, group_keys=True)
        .apply(
            lambda df: (
                df.set_index(args.datetime_col)
                .reindex(
                    pd.date_range(
                        pd.to_datetime(df[args.datetime_col].min().date()),
                        pd.to_datetime(df[args.datetime_col].max().date()+pd.Timedelta(days=1)),
                        freq=args.inner_aggregation_freq,
                        inclusive="left",
                    )
                )
                .rename_axis(args.datetime_col)[[args.location_col]]
            )
        )
        .reset_index()
        .fillna({args.location_col: args.no_location_str})
    )


def process_day_str_from_locations(df, args):
    """
    Turns location data into a string for each day.
    If no location exists for a day, the no location string is used.
    
    """
    sentences = (
        df
        # groupby patient and day
        .groupby(
            [args.id_col, pd.Grouper(key=args.datetime_col, freq=args.aggregation_freq)]
        )
        # get sequence of location names
        [args.location_col]
        .apply(lambda x: x.tolist())
        .reset_index()
        # ensuring that days with no location visits are included
        .groupby(args.id_col, group_keys=True)
        .apply(
            lambda df: (
                df.set_index(args.datetime_col)
                .reindex(
                    pd.date_range(
                        df[args.datetime_col].min(),
                        df[args.datetime_col].max(),
                        freq=args.aggregation_freq,
                    )
                )
                .rename_axis(args.datetime_col)[[args.location_col]]
            )
        )
        .reset_index()
        # where there are no location visits, fill with ['No Location']
        .assign(
            **{
                args.location_list_col: lambda df: (
                    df[args.location_col]
                    # fill missing values with 'No Location'
                    .fillna(args.no_location_str)
                    # convert all values to list
                    .apply(lambda x: [x] if type(x) == str else x)
                )
            }
        )
        # convert location sequences to strings
        .assign(
            **{
                args.location_str_col: lambda x: x[args.location_list_col].str.join(
                    ", "
                )
            }
        )
    )

    return sentences
