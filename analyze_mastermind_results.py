import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def define_strategy(strategy):
    if strategy == 0:
        return "Random Guess Strategy"
    elif strategy == 1:
        return "Simple Heuristic Strategy"
    elif strategy == 2:
        return "Knuth's Algorithm"
    else:
        return "Unknown Strategy"

def gather_csv_files(directory):
    return glob.glob(os.path.join(directory, 'mastermind_ai_results_*.csv'))

def validate_csv_structure(df):
    required_columns = ['Strategy', 'Game Number', 'Attempts', 'Success', 'Guesses', 'Feedbacks', 'Secret Code']
    return all(column in df.columns for column in required_columns)

def load_and_combine_csv_files(file_list):
    combined_df = pd.DataFrame()
    for file in file_list:
        df = pd.read_csv(file)
        if validate_csv_structure(df):
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Invalid CSV structure in file: {file}")
    return combined_df

def apply_strategy_names(df):
    df['Strategy Name'] = df['Strategy'].apply(define_strategy)
    return df

def analyze_and_visualize(df):
    df = apply_strategy_names(df)
    grouped = df.groupby('Strategy Name').agg({
        'Attempts': 'mean',
        'Success': 'mean'
    }).reset_index()
    grouped.columns = ['Strategy', 'Average Attempts', 'Success Rate']
    grouped['Success Rate'] *= 100
    grouped = grouped.sort_values(by='Average Attempts')

    print(grouped)

    plt.figure(figsize=(10, 5))
    plt.bar(grouped['Strategy'], grouped['Average Attempts'], color='blue')
    plt.xlabel('Strategy')
    plt.ylabel('Average Attempts')
    plt.title('Average Attempts by Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(grouped['Strategy'], grouped['Success Rate'], color='green')
    plt.xlabel('Strategy')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    directory = './results/'
    csv_files = gather_csv_files(directory)
    combined_df = load_and_combine_csv_files(csv_files)
    if not combined_df.empty:
        analyze_and_visualize(combined_df)
    else:
        print("No valid CSV files found.")