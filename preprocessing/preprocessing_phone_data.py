import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series

#function to save the data to a csv file
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

#function to replace new column with the old one of the same name
def add_new_col_to_df(df, col, s):
    del df[col]
    new_df = df.join(s)
    
    return new_df

#function to split '|' separated values into multiple rows
def split_column_data_to_multiple_rows(df, col):
    s = df[col].str.split('|').apply(Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = col
    
    new_df = add_new_col_to_df(df, col, s)
    
    return new_df

def main():
    phones = pd.read_csv("../data/original/phone_data.csv", on_bad_lines='skip')
    cols = ['brand','model', 'GPRS', 'EDGE', 'status', 'dimentions', 'SIM', 'display_type',
            'display_resolution', 'display_size', 'OS', 'CPU', 'Chipset', 'GPU', 'memory_card',
            'internal_memory', 'RAM', 'primary_camera', 'secondary_camera', 'WLAN', 'bluetooth',
            'GPS', 'sensors', 'battery', 'colors', 'approx_price_EUR']
    data = phones[cols]

    data_copy = data.copy()
    data_df_cols = ['brand','model', 'primary_camera', 'secondary_camera', 'WLAN', 'bluetooth', 'sensors',
                'colors', 'approx_price_EUR']
    data_df = data_copy[data_df_cols]
    
    #Split and replace with new column values for Primary Camera Column
    new_primary_camera_df = split_column_data_to_multiple_rows(data_df, 'primary_camera')
    new_primary_camera_df.drop_duplicates(keep=False,inplace=True) 
    #Split and replace with new column values for Secondary Camera Column
    new_secondary_camera_df = split_column_data_to_multiple_rows(new_primary_camera_df, 'secondary_camera')
    new_secondary_camera_df.drop_duplicates(keep=False,inplace=True) 
    #Split and replace with new column values for WLAN Column
    new_WLAN_df = split_column_data_to_multiple_rows(new_secondary_camera_df, 'WLAN')
    new_WLAN_df.drop_duplicates(keep=False,inplace=True) 
    #Split and replace with new column values for Bluetooth Column
    new_bluetooth_df = split_column_data_to_multiple_rows(new_WLAN_df, 'bluetooth')
    new_bluetooth_df.drop_duplicates(keep=False,inplace=True) 
    #Split and replace with new column values for Sensors Column
    new_sensors_df = split_column_data_to_multiple_rows(new_bluetooth_df, 'sensors')
    new_sensors_df.drop_duplicates(keep=False,inplace=True) 
    #Split and replace with new column values for Colors Column
    new_colors_df = split_column_data_to_multiple_rows(new_sensors_df, 'colors')
    new_colors_df.drop_duplicates(keep=False,inplace=True) 
    #Bringig down internal memory column to the new dataframe for processing and plit and replace with new column values for Internal Memory Column
    new_df = new_colors_df.join(data_copy['internal_memory'])
    new_internal_memory_df = split_column_data_to_multiple_rows(new_df, 'internal_memory')
    new_internal_memory_df.drop_duplicates(keep=False,inplace=True) 
    #Now bringing down status from original dataframe and joining to the updated dataframe and splitting Status Column into multiple status related columns
    status_df = new_internal_memory_df.join(data_copy['status'])
    split_data = status_df["status"].str.split(" ")
    sdata = split_data.to_list()
    names = ["release_status", "released", "release_year", 'release_day', 'release_month/quarter', 'release_hour', 'release_min']
    new_split_df = pd.DataFrame(sdata, columns=names)
    new_status_df = new_split_df.drop(['released', 'release_day', 'release_hour', 'release_min'], axis=1)
    #The new dataframe contains newly created status related columns replacing the status column
    new_data_df = status_df.join(new_status_df)
    new_data_df = new_data_df.drop(['status'], axis=1)
    new_data_df.drop_duplicates(keep=False,inplace=True) 
    #Adding battery column to new dataframe for further processing and spliting battery column into multiple columns replacing the battery column
    battery_df = new_data_df.join(data_copy['battery'])
    split_battery_data = battery_df['battery'].str.split(" ")
    battery_data = split_battery_data.to_list()
    battery_col_names = ['removable/non-removable', 'battery_type', 'battery_current', 'battery_unit', 'colname_battery', 'col6',
            'col7', 'col8', 'col9']
    battery_split_df = pd.DataFrame(battery_data, columns=battery_col_names)
    battery_split_df = battery_split_df.drop([ 'colname_battery','col6', 'col7', 'col8', 'col9'], axis=1)
    new_battery_split_df = battery_split_df.replace(to_replace ="battery", value ="NaN") 
    #adding newly created battery related columns to the dataframe
    new_battery_data_df = new_data_df.join(new_battery_split_df)
    new_battery_data_df.drop_duplicates(keep=False,inplace=True) 
    #adding remaining features to the dataframe
    full_df = new_battery_data_df.join(data_copy[['display_resolution', 'display_size', 'GPRS', 'EDGE',
                                                'dimentions', 'SIM', 'OS', 'CPU', 'Chipset', 'GPU',
                                                'memory_card', 'RAM', 'GPS']])
    #Stripping off unwanted information from display columns
    full_df['display_resolution'] = full_df['display_resolution'].str.split('(').str[0]
    full_df['display_size'] = full_df['display_size'].str.split('(').str[0]
    full_df['dimentions'] = full_df['dimentions'].str.split('(').str[0]
    full_df.drop_duplicates(keep=False,inplace=True)
    #OS data contains "|" separated values thus splitting into multiples rows. 
    final_X_df = split_column_data_to_multiple_rows(full_df, 'OS')
    final_X_df.drop_duplicates(keep=False,inplace=True) 

    #Imputing missing values
    final_X_df['primary_camera'] = final_X_df['primary_camera'].fillna("No")
    final_X_df['secondary_camera'] = final_X_df['secondary_camera'].fillna("No")
    final_X_df['colors'] = final_X_df['colors'].fillna("No")
    final_X_df['bluetooth'] = final_X_df['bluetooth'].fillna("No")
    final_X_df['sensors'] = final_X_df['sensors'].fillna("No")
    final_X_df['release_year'] = final_X_df['release_year'].fillna("2010")
    final_X_df['release_year'] = final_X_df['release_year'].replace('Exp.', '2010') 
    final_X_df['release_year'] = pd.to_numeric(final_X_df['release_year'], errors='coerce')
    final_X_df['release_month/quarter'] = final_X_df['release_month/quarter'].fillna("No")
    final_X_df['removable/non-removable'] = final_X_df['removable/non-removable'].fillna("Non-removable")
    final_X_df['removable/non-removable'] = final_X_df['removable/non-removable'].replace("Li-Ion", 'Non-removable') 
    final_X_df['battery_type'] = final_X_df['battery_type'].fillna("Li-Ion")
    final_X_df['battery_type'] = final_X_df['battery_type'].replace('NaN.', 'Li-Ion')
    final_X_df['battery_current'] = final_X_df['battery_current'].fillna(0) 
    final_X_df['battery_current'] = final_X_df['battery_current'].replace("(BST-20)", 0)
    final_X_df['battery_current'] = final_X_df['battery_current'].replace("mAh", 0)
    final_X_df['battery_current'] = final_X_df['battery_current'].replace("", 0)
    final_X_df['battery_current'] = final_X_df['battery_current'].replace("NaN", 0)
    final_X_df['battery_current'] = pd.to_numeric(final_X_df['battery_current'], errors='coerce')
    final_X_df['battery_current'] = final_X_df['battery_current'].replace(0, final_X_df['battery_current'].mean())
    final_X_df = final_X_df.drop(['battery_unit'], axis=1)
    final_X_df['display_resolution'] = final_X_df['display_resolution'].fillna("No")
    final_X_df['display_size'] = final_X_df['display_size'].fillna("No")
    final_X_df['GPRS'] = final_X_df['GPRS'].fillna("No")
    final_X_df['EDGE'] = final_X_df['EDGE'].fillna("No")
    final_X_df['dimentions'] = final_X_df['dimentions'].fillna("No")
    final_X_df['SIM'] = final_X_df['SIM'].fillna("No")
    final_X_df['CPU'] = final_X_df['CPU'].fillna("No")
    final_X_df['Chipset'] = final_X_df['Chipset'].fillna("No")
    final_X_df['GPU'] = final_X_df['GPU'].fillna("No")
    final_X_df['memory_card'] = final_X_df['memory_card'].fillna("No")
    final_X_df['RAM'] = final_X_df['RAM'].fillna("No")
    final_X_df['GPS'] = final_X_df['GPS'].fillna("No")
    final_X_df['OS'] = final_X_df['OS'].fillna("No") 
    final_X_df['internal_memory'] = final_X_df['internal_memory'].fillna("No")
    final_X_df['approx_price_EUR'] = final_X_df['approx_price_EUR'].fillna(final_X_df['approx_price_EUR'].mean())
    final_X_df['release_status'] = final_X_df['release_status'].replace('Available.', 'Available')
    final_X_df['release_status'] = final_X_df['release_status'].replace('Cancelled', 'Discontinued')
    final_X_df['release_status'] = final_X_df['release_status'].fillna("Discontinued")

    # quantify all categorical columns
    final_X_df['brand'] = pd.Categorical(final_X_df['brand'])
    final_X_df['model'] = pd.Categorical(final_X_df['model'])
    final_X_df['primary_camera'] = pd.Categorical(final_X_df['primary_camera'])
    final_X_df['secondary_camera'] = pd.Categorical(final_X_df['secondary_camera'])
    final_X_df['WLAN'] = pd.Categorical(final_X_df['WLAN'])
    final_X_df['bluetooth'] = pd.Categorical(final_X_df['bluetooth'])
    final_X_df['sensors'] = pd.Categorical(final_X_df['sensors'])
    final_X_df['colors'] = pd.Categorical(final_X_df['colors'])
    final_X_df['display_resolution'] = pd.Categorical(final_X_df['display_resolution'])
    final_X_df['display_size'] = pd.Categorical(final_X_df['display_size'])
    final_X_df['GPRS'] = pd.Categorical(final_X_df['GPRS'])
    final_X_df['EDGE'] = pd.Categorical(final_X_df['EDGE'])
    final_X_df['dimentions'] = pd.Categorical(final_X_df['dimentions'])
    final_X_df['SIM'] = pd.Categorical(final_X_df['SIM'])
    final_X_df['OS'] = pd.Categorical(final_X_df['OS'])
    final_X_df['CPU'] = pd.Categorical(final_X_df['CPU'])
    final_X_df['Chipset'] = pd.Categorical(final_X_df['Chipset'])
    final_X_df['GPU'] = pd.Categorical(final_X_df['GPU'])
    final_X_df['memory_card'] = pd.Categorical(final_X_df['memory_card'])
    final_X_df['RAM'] = pd.Categorical(final_X_df['RAM'])
    final_X_df['GPS'] = pd.Categorical(final_X_df['GPS'])
    final_X_df['removable/non-removable'] = pd.Categorical(final_X_df['removable/non-removable'])
    final_X_df['battery_type'] = pd.Categorical(final_X_df['battery_type'])
    final_X_df['battery_current'] = pd.Categorical(final_X_df['battery_current'])
    final_X_df['internal_memory'] = pd.Categorical(final_X_df['internal_memory'])
    final_X_df['release_month/quarter'] = pd.Categorical(final_X_df['release_month/quarter'])
    final_X_df['release_status'] = pd.Categorical(final_X_df['release_status'])

    #convert all categorical columns to numerical
    final_X_df['brand'] = final_X_df['brand'].cat.codes
    final_X_df['model'] = final_X_df['model'].cat.codes
    final_X_df['primary_camera'] = final_X_df['primary_camera'].cat.codes
    final_X_df['secondary_camera'] = final_X_df['secondary_camera'].cat.codes
    final_X_df['WLAN'] = final_X_df['WLAN'].cat.codes
    final_X_df['bluetooth'] = final_X_df['bluetooth'].cat.codes
    final_X_df['sensors'] = final_X_df['sensors'].cat.codes
    final_X_df['colors'] = final_X_df['colors'].cat.codes
    final_X_df['display_resolution'] = final_X_df['display_resolution'].cat.codes
    final_X_df['display_size'] = final_X_df['display_size'].cat.codes
    final_X_df['GPRS'] = final_X_df['GPRS'].cat.codes
    final_X_df['EDGE'] = final_X_df['EDGE'].cat.codes
    final_X_df['dimentions'] = final_X_df['dimentions'].cat.codes
    final_X_df['SIM'] = final_X_df['SIM'].cat.codes
    final_X_df['OS'] = final_X_df['OS'].cat.codes
    final_X_df['CPU'] = final_X_df['CPU'].cat.codes
    final_X_df['Chipset'] = final_X_df['Chipset'].cat.codes
    final_X_df['GPU'] = final_X_df['GPU'].cat.codes
    final_X_df['memory_card'] = final_X_df['memory_card'].cat.codes
    final_X_df['RAM'] = final_X_df['RAM'].cat.codes
    final_X_df['GPS'] = final_X_df['GPS'].cat.codes
    final_X_df['removable/non-removable'] = final_X_df['removable/non-removable'].cat.codes
    final_X_df['battery_type'] = final_X_df['battery_type'].cat.codes
    final_X_df['battery_current'] = final_X_df['battery_current'].cat.codes
    final_X_df['internal_memory'] = final_X_df['internal_memory'].cat.codes
    final_X_df['release_month/quarter'] = final_X_df['release_month/quarter'].cat.codes
    final_X_df['release_status'] = final_X_df['release_status'].cat.codes

    #place release status column at the end
    cols = final_X_df.columns.tolist()
    cols.remove('release_status')
    cols.append('release_status')
    final_X_df = final_X_df[cols]

    #rename release_status column to label
    final_X_df.rename(columns={'release_status': 'label'}, inplace=True)

    #save the data to a csv file
    columns = final_X_df.columns
    data = final_X_df.to_numpy()
    data_ = []
    data_.append(columns)
    for row in data:
        data_.append(row)
    save_csv(data_, "../data/phone.csv")


if __name__ == "__main__":
    main()