import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from mastermind import define_strategy  # Import the define_strategy function

# Select Strategies to compare
strategies = [0, 1, 2, 3, 4, 5, 6, 7]  # An array of numbers representing the strategies to analyze this run

def gather_csv_files(directory):
    """Gather all CSV files matching the pattern in the specified directory."""
    return glob.glob(os.path.join(directory, 'mastermind_ai_results_*.csv'))

def validate_csv_structure(df):
    """Validate the structure of the CSV file."""
    required_columns = ['Strategy', 'Game Number', 'Attempts', 'Success', 'Guesses', 'Feedbacks', 'Secret Code']
    return all(column in df.columns for column in required_columns)

def load_and_combine_csv_files(file_list):
    """Load and combine all valid CSV files into a single DataFrame."""
    combined_df = pd.DataFrame()
    for file in file_list:
        df = pd.read_csv(file)
        if validate_csv_structure(df):
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Invalid CSV structure in file: {file}")
    return combined_df

def apply_strategy_names(df):
    """Apply user-friendly strategy names to the DataFrame."""
    df['Strategy Name'] = df['Strategy'].apply(define_strategy)
    return df

def analyze_and_visualize(df):
    """Analyze and visualize the results."""
    # Apply strategy names
    df = apply_strategy_names(df)

    # Filter the DataFrame to include only the specified strategies
    df = df[df['Strategy'].isin(strategies)]

    # Group by strategy name and calculate the average attempts, success rate, and total games won
    grouped = df.groupby('Strategy Name').agg({
        'Attempts': 'mean',
        'Success': 'mean',
        'Game Number': 'count'
    }).reset_index()

    # Calculate the total number of games won
    grouped['Total Games Won'] = df[df['Success'] == True].groupby('Strategy Name').size().reindex(grouped['Strategy Name'], fill_value=0).values

    # Rename columns for clarity
    grouped.columns = ['Strategy', 'Average Attempts', 'Success Rate', 'Total Games', 'Total Games Won']
    grouped['Success Rate'] *= 100

    # Sort by the most effective strategy (e.g., highest success rate)
    grouped = grouped.sort_values(by='Success Rate', ascending=False)

    # Display the aggregated data
    print(grouped)

    # Plot average attempts by strategy
    plt.figure(figsize=(10, 5))
    plt.bar(grouped['Strategy'], grouped['Average Attempts'], color='blue')
    plt.xlabel('Strategy')
    plt.ylabel('Average Attempts')
    plt.title('Average Attempts by Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot success rate by strategy
    plt.figure(figsize=(10, 5))
    plt.bar(grouped['Strategy'], grouped['Success Rate'], color='green')
    plt.xlabel('Strategy')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plot total games won by strategy
    plt.figure(figsize=(10, 5))
    plt.bar(grouped['Strategy'], grouped['Total Games Won'], color='purple')
    plt.xlabel('Strategy')
    plt.ylabel('Total Games Won')
    plt.title('Total Games Won by Strategy')
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