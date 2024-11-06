import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


def main():
    # DATASET_NAME = 'moons'
    # DATASET_NAME = 'arrow'
    DATASET_NAME = 'phone'

    FOLDER_NAME = '../data/'
    if DATASET_NAME == 'moons':
        FOLDER_NAME += 'moons.csv'
    else:
        FOLDER_NAME += 'encoded/' + DATASET_NAME + '.csv'

    real_data = pd.read_csv(FOLDER_NAME)
    real_data = real_data.iloc[:, :-1]
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    if not os.path.exists('../data/metadata/generated/' + DATASET_NAME + '.json'):
        metadata.save_to_json(filepath= '../data/metadata/generated/' + DATASET_NAME + '.json')

    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(real_data)
    history = synthesizer.get_loss_values()
    synthetic_data = synthesizer.sample(num_rows=10000)

    synthesizer.save(filepath='../models/generative/ctgan_' + DATASET_NAME + '.pkl')
    history.to_csv('../models/generative/history/ctgan_' + DATASET_NAME + '.csv', header=True, index=False)
    synthetic_data.to_csv('../data/generated/CTGAN/' + DATASET_NAME + '.csv', header=True, index=False)

if __name__ == "__main__":
    main()