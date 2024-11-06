import pandas as pd
import numpy as np

def save_csv(data, path):
    '''
    Save the data to a csv file
    input: data - list
           path - string
    output: None
    '''
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def main():
    df = pd.read_excel("../data/original/arrow_data.xlsx", sheet_name=[0, 1])
    # df = pd.concat([df[0], df[1]]).reset_index(drop=True)

    df[0] = df[0].drop(df[0].columns[0], axis=1)
    df[1] = df[1].drop(df[1].columns[0], axis=1)

    # change all non numeric values to 0 in first column
    df[0].iloc[:, 0] = pd.to_numeric(df[0].iloc[:, 0], errors='coerce').fillna(0)
    df[1].iloc[:, 0] = pd.to_numeric(df[1].iloc[:, 0], errors='coerce').fillna(0)

    # iterate through columns and change non numeric values to numeric
    for col in [9, 10]:
        df[0].iloc[:, col] = df[0].iloc[:, col].str.extract(r'(\d+)')
        df[1].iloc[:, col] = df[1].iloc[:, col].str.extract(r'(\d+)')

    # fill all missing values with -1
    df[0] = df[0].fillna(-1)
    df[1] = df[1].fillna(-1)

    # change dtype of columns
    df[0]["Stock"] = df[0]["Stock"].astype('float64')
    df[0]["Maximum regulator current (MA)"] = df[0]["Maximum regulator current (MA)"].astype('float64')
    df[0]["Maximum Zener impedance (OHM)"] = df[0]["Maximum Zener impedance (OHM)"].astype('float64')
    df[1]["Stock"] = df[1]["Stock"].astype('float64')
    df[1]["Maximum regulator current (MA)"] = df[1]["Maximum regulator current (MA)"].astype('float64')
    df[1]["Maximum Zener impedance (OHM)"] = df[1]["Maximum Zener impedance (OHM)"].astype('float64')

    df[0]["Fabricant"] = df[0]["Fabricant"].astype('category')
    df[0]["Type"] = df[0]["Type"].astype('category')
    df[0]["Configuration"] = df[0]["Configuration"].astype('category')
    df[0]["Packaging"] = df[0]["Packaging"].astype('category')
    df[0]["SVHC"] = df[0]["SVHC"].astype('category')
    df[1]["Fabricant"] = df[1]["Fabricant"].astype('category')
    df[1]["Type"] = df[1]["Type"].astype('category')
    df[1]["Configuration"] = df[1]["Configuration"].astype('category')
    df[1]["Packaging"] = df[1]["Packaging"].astype('category')
    df[1]["SVHC"] = df[1]["SVHC"].astype('category')

    cat_columns = df[0].select_dtypes(['category']).columns
    df[0][cat_columns] = df[0][cat_columns].apply(lambda x: x.cat.codes)
    df[1][cat_columns] = df[1][cat_columns].apply(lambda x: x.cat.codes)

    data_obso = df[0]
    data_act = df[1]
    y_obso = [0] * data_obso.shape[0]
    y_act = [1] * data_act.shape[0]
    data_obso["label"] = y_obso
    data_act["label"] = y_act
    data = pd.concat([data_obso, data_act]).reset_index(drop=True)
    columns = data.columns
    data = data.to_numpy()

    np.random.seed(0)
    np.random.shuffle(data)

    data_ = []
    data_.append(columns)
    for row in data:
        data_.append(row)
    save_csv(data_, "../data/arrow.csv")


if __name__ == '__main__':
    main()