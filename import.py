import pandas as pd
from pymongo import MongoClient

# 1. Connect to your local MongoDB instance
client = MongoClient("mongodb://localhost:27017/")

# 2. Define your target Database and Collection names
# MongoDB will automatically create these if they do not exist yet
db = client["agl_data"]
collection = db["raw_match_results"]

# 3. Read the downloaded CSV file using Pandas
# This automatically treats row 1 as object keys/headers
df = pd.read_csv("data.csv")

# 4. Convert the dataframe to a dictionary format MongoDB expects
records = df.to_dict(orient="records")

# 5. Insert documents in bulk
if records:
    result = collection.insert_many(records)
    print(f"✅ Successfully created database and imported {len(result.inserted_ids)} records!")
else:
    print("❌ The CSV file appears to be empty.")

