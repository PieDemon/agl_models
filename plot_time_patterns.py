import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
from datetime import datetime, timedelta

def generate_time_heatmap():
    # 1. Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]

    # 2. Calculate the "one year ago" threshold dynamically based on the current date
    one_year_ago = datetime.now() - timedelta(days=365)
    # Format to match ISO string style (e.g., '2025-06-04T00:00:00Z')
    iso_cutoff_string = one_year_ago.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    matches = list(collection.find({"Time": {"$gte": iso_cutoff_string}}))
    
    if not matches:
        print("❌ No matches found in 'elo_matches' collection.")
        return

    data_rows = []
    
    print("Processing timestamps and grouping data...")
    for match in matches:
        time_str = match.get("Time")
        if not time_str:
            continue
            
        # Parse standard ISO timestamp (e.g., '2020-12-04T15:16:22Z')
        # Using pandas helper handles time string formatting safely
        dt = pd.to_datetime(time_str)
        
        # Extract Day and Hour
        day_name = dt.strftime('%A')  # "Monday", "Tuesday", etc.
        hour = dt.hour
        
        # Segment hour into 6-hour chunks
        if 0 <= hour < 3:
            time_bucket = "00:00-03:00"
        elif 3 <= hour < 6:
            time_bucket = "03:00-06:00"
        elif 6 <= hour < 9:
            time_bucket = "06:00-09:00"
        elif 9 <= hour < 12:
            time_bucket = "09:00-12:00"
        elif 12 <= hour < 15:
            time_bucket = "12:00-15:00"
        elif 15 <= hour < 18:
            time_bucket = "15:00-18:00"
        elif 18 <= hour < 21:
            time_bucket = "18:00-21:00"
        else:
            time_bucket = "21:00-00:00"
            
        # We look at BOTH players in the match to evaluate general skill pool at that time
        # To see the skill level *entering* the match, we look at their old Elo before changes
        w_old_elo = match["winner_elo"] - (match["winner_elo"] - match.get("winner_alltime_wins", 1000)) # fallback safety
        # Better yet, since we calculated Elo linearly, we can easily extrapolate pre-match values:
        # Let's cleanly grab the calculated post-match elos, or look directly at their starting state.
        # To keep it completely precise, we'll average the final resting Elo of both active match participants.
        match_avg_elo = (match["winner_elo"] + match["loser_elo"]) / 2
        
        data_rows.append({
            "Day": day_name,
            "Bucket": time_bucket,
            "AvgElo": match_avg_elo
        })
        
    df = pd.DataFrame(data_rows)
    
    # 2. Pivot the data into a 2D Matrix (Days x Time Buckets), calculating the Mean Elo for each cell
    # Enforce chronological ordering for days and times so the chart layout makes sense
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    buckets_order = ["00:00-03:00", "03:00-06:00", "06:00-09:00", "09:00-12:00", "12:00-15:00", "15:00-18:00", "18:00-21:00", "21:00-00:00"]
    
    # Calculate averages per bucket intersection
    pivot_df = df.pivot_table(
        values="AvgElo", 
        index="Day", 
        columns="Bucket", 
        aggfunc="mean"
    ).reindex(index=days_order, columns=buckets_order)
    
    # Count matches per bucket to display as overlay numbers (annotations)
    count_df = df.pivot_table(
        values="AvgElo", 
        index="Day", 
        columns="Bucket", 
        aggfunc="count"
    ).reindex(index=days_order, columns=buckets_order).fillna(0).astype(int)

    # 3. Render the Visual Heatmap Grid
    plt.figure(figsize=(12, 7))
    
    # Format the grid: 'annot=True' prints text inside squares.
    # We will format it to print: "Average Elo \n (Match Count)"
    labels = np.array([
        [f"{pivot_df.iloc[r, c]:.0f}\n({count_df.iloc[r, c]} m)" if not np.isnan(pivot_df.iloc[r, c]) else "No Data"
         for c in range(pivot_df.shape[1])]
        for r in range(pivot_df.shape[0])
    ])
    
    sns.heatmap(
        pivot_df, 
        annot=labels, 
        fmt="", 
        cmap="coolwarm",  # Blue = Lower Elo pools, Red = High Elo "sweat" hours
        linewidths=0.5, 
        cbar_kws={'label': 'Average Player Elo Rating'}
    )
    
    plt.title("Tournament Activity Profile: Average Elo By Time of Day", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Time of Day (3-Hour Windows)", fontsize=11, labelpad=10)
    plt.ylabel("Day of Week", fontsize=11, labelpad=10)
    plt.tight_layout()
    
    print("📉 Rendering activity profile window...")
    plt.show()

if __name__ == "__main__":
    generate_time_heatmap()

