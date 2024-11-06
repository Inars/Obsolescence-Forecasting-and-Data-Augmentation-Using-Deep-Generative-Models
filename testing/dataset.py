import pandas as pd

def main():
    df_arrow = pd.read_csv('../data/arrow.csv')
    df_phone = pd.read_csv('../data/phone.csv')
    print(df_arrow.describe())
    print(df_phone.describe())
    print(df_arrow['label'].value_counts())
    print(df_phone['label'].value_counts())
    print(df_arrow['label'].value_counts(normalize=True))
    print(df_phone['label'].value_counts(normalize=True))

if __name__ == '__main__':
    main()