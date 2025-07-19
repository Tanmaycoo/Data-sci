import pandas as pd

# Sample data
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 28],
    "Score": [85, 90, 95],
    "marks":[20,50,28]
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("mydata.csv", index=False)

print("CSV file created!")
