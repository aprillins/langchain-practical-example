import pandas as pd

def convert_volume(value):
    if 'M' in value:
        return float(value.replace('M', '').replace(',', '')) * 1_000_000
    elif 'K' in value:
        return float(value.replace('K', '').replace(',', '')) * 1_000
    return float(value.replace(',', ''))

def remove_commas_from_numbers(input_file, output_file):
    df = pd.read_csv(input_file)

    # Replace commas in numeric columns and convert to appropriate data types
    numeric_columns = ['Price', 'Open', 'High', 'Low']
    
    for column in numeric_columns:
        df[column] = df[column].str.replace(',', '').astype(float)

    # Convert 'Vol.' and handle 'M' and 'K'
    df['Vol.'] = df['Vol.'].apply(convert_volume)

    df.to_csv(output_file, index=False)

filename = 'data_historis_jpfa_2012-2024.csv'
input_csv = '../data/' + filename  # Replace with your file
output_csv = 'output/' + filename
remove_commas_from_numbers(input_csv, output_csv)