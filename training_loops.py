import multiprocessing
import pandas as pd
import swift_2
from preprocessing.dataframe import swift_2_process_df
from multiprocessing import Pool


def run_trial(text_id, trial_id, df):
    trial, dwell_time = swift_2.run(trial_id, df)
    print(f"Text {text_id} Trial {trial_id} complete")

    return trial, dwell_time


def main(data, text, output_dir, trials=200):
    num_cpus = min(multiprocessing.cpu_count(), 50)

    for sentence_id, sentence in enumerate(text):
        with Pool(processes=num_cpus) as pool:
            trial_results = []
            dwell_time_results = []

            start = sentence[0]
            stop = sentence[1]
            current_data = data.iloc[start:stop].copy()
            current_data = swift_2_process_df(current_data)

            items = [(sentence_id, trial_id, current_data) for trial_id in range(trials)]
            results = pool.starmap(run_trial, items)

            for i, (trial, dwell_time) in enumerate(results):
                if i == 0:
                    trial_results.append(trial)

                dwell_time_results.append(dwell_time)

            full_trial_results = trial_results[0] #pd.concat(trial_results)
            full_dwell_time_results = pd.concat(dwell_time_results, axis=1)

            full_trial_results.to_csv(f"{output_dir}Sentence_{sentence_id}.csv", index=False)
            full_dwell_time_results.transpose().to_csv(f"{output_dir}DT_Sentence_{sentence_id}.csv", index=False)

            del full_trial_results, full_dwell_time_results, trial_results, dwell_time_results
