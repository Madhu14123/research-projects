import pandas as pd
import numpy as np
import os

# Define the path to the folder containing the files
folder_path = 'D:/energy_price_predn/energy_data1'

# Initialize an empty list to store the results
all_results = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # Only process CSV files
        # Full file path
        file_path = os.path.join(folder_path, filename)

        print(f"Processing file: {filename}")  # Print file being processed
        
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Step 1: Filter data for the specific PointOfConnection 'ALB0331'
        df = df[df["PointOfConnection"] == "ALB0331"]

        # Skip files where no rows match the filter
        if df.empty:
            print(f"Skipping file {filename} as it contains no data for ALB0331.")
            continue

        # Convert 'TradingDate' to datetime
        df['TradingDate'] = pd.to_datetime(df['TradingDate'], errors='coerce')

        # Skip files with parsing issues
        if df['TradingDate'].isnull().any():
            print(f"Skipping file {filename} due to date parsing issues.")
            continue

        df['Year'] = df['TradingDate'].dt.year
        df['Month'] = df['TradingDate'].dt.month
        df['Day'] = df['TradingDate'].dt.day

        # Step 2: Calculate Avg$PerMWHr, PrevAvg$PerMWHr, and PrevMed$PerMWHr
        df['Avg$PerMWHr'] = df.groupby(['Year', 'Month', 'Day', 'TradingPeriod'])['DollarsPerMegawattHour'].transform('mean')
        df['PrevAvg$PerMWHr'] = df.groupby(['Year', 'Month', 'Day', 'TradingPeriod'])['Avg$PerMWHr'].shift(1)
        df['PrevMed$PerMWHr'] = df.groupby(['Year', 'Month', 'Day', 'TradingPeriod'])['DollarsPerMegawattHour'].shift(1)

        # Step 3: Calculate 'Med$PerMWHr' from the current day's 'DollarsPerMegawattHour'
        df['Med$PerMWHr'] = df.groupby(['Year', 'Month', 'Day', 'TradingPeriod'])['DollarsPerMegawattHour'].transform('mean')

        # Step 4: Calculate 'SinPeriod' and 'CosPeriod' for `TradingPeriod`
        df['SinPeriod'] = np.sin(2 * np.pi * df['TradingPeriod'] / 24)
        df['CosPeriod'] = np.cos(2 * np.pi * df['TradingPeriod'] / 24)

        # Step 5: Calculate 'SinDate' and 'CosDate' for cyclical representation of the date
        df['SinDate'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['CosDate'] = np.cos(2 * np.pi * df['Month'] / 12)

        # Step 6: Aggregate data by 'Year', 'Month', 'Day', and 'TradingPeriod'
        trading_period_df = df.groupby(['Year', 'Month', 'Day', 'TradingPeriod'], as_index=False).agg({
            'IsProxyPriceFlag': 'first',  # Assuming this doesn't change within the same trading period
            'Med$PerMWHr': 'mean',  # Take the mean of Med$PerMWHr for each trading period
            'Avg$PerMWHr': 'mean',  # Take the mean of Avg$PerMWHr for each trading period
            'PrevAvg$PerMWHr': 'mean',  # Take the mean of PrevAvg$PerMWHr for each trading period
            'PrevMed$PerMWHr': 'mean',  # Take the mean of PrevMed$PerMWHr for each trading period
            'SinPeriod': 'mean',  # Take the mean of SinPeriod
            'CosPeriod': 'mean',  # Take the mean of CosPeriod
            'SinDate': 'mean',  # Take the mean of SinDate
            'CosDate': 'mean',  # Take the mean of CosDate
        })

        # Append the result of the current file to the list
        all_results.append(trading_period_df)

# Combine all results into a single DataFrame
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    # Save the combined results to a new CSV file
    final_df.to_csv('transformed_energy_data_ALB0331.csv', index=False)
    print("All files processed and combined successfully.")
    print(final_df)
else:
    print("No matching data found for ALB0331 across all files.")
