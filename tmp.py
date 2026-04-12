import pandas as pd

# Load the original CSV file
# Ensure Suicide_Detection.csv is in the same directory as your script
try:
    df = pd.read_csv('Suicide_Detection.csv')

    # Extract the first 10 rows
    df_first_10 = df.head(10)

    # Save the result to test.csv
    # index=False prevents pandas from adding an extra column for row numbers
    df_first_10.to_csv('test.csv', index=False)

    print("Successfully saved the first 10 rows to test.csv")
except FileNotFoundError:
    print("Error: The file 'Suicide_Detection.csv' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")