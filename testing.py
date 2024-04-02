import pandas as pd

# Creating a sample DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': ['foo', 'bar', 'foo', 'bar', 'baz']}
df = pd.DataFrame(data)

# Displaying the original DataFrame
print("Original DataFrame:")
print(df)

# Dropping rows where column 'B' has value 'foo'
df_filtered = df[(df['B'] != 'foo') & (df['B'] != 'bar')]

# Displaying the filtered DataFrame
print("\nFiltered DataFrame:")
print(df_filtered)
