import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient, ASCENDING

# Set the name of the player you want to evaluate
TARGET_PLAYER = "Jesse R" 

def chart_player_progression():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["my_new_database"]
    collection = db["elo_matches"]
    
    # 1. Query all matches where this player participated, sorted chronologically
    query = {
        "$or": [
            {"Winning_Player": TARGET_PLAYER},
            {"Losing_Player": TARGET_PLAYER}
        ]
    }
    matches = collection.find(query).sort("Time", ASCENDING)
    
    # 2. Extract chronological timeline data
    timeline = []
    
    # Always insert their starting point (1000 Elo before match 1)
    timeline.append({"MatchNum": 0, "Elo": 1000, "MatchCode": "Start"})
    
    for idx, match in enumerate(matches, start=1):
        # Determine if they won or lost this specific match to pull the correct rating
        if match["Winning_Player"] == TARGET_PLAYER:
            end_elo = match["winner_elo"]
        else:
            end_elo = match["loser_elo"]
            
        timeline.append({
            "MatchNum": idx,
            "Elo": end_elo,
            "MatchCode": match.get("MatchCode", f"M{idx}")
        })
        
    if len(timeline) <= 1:
        print(f"❌ No matches found for player: '{TARGET_PLAYER}'")
        return

    # 3. Load into a Pandas DataFrame for painless plotting
    df = pd.DataFrame(timeline)
    
    # 4. Construct the visual chart
    plt.figure(figsize=(10, 5))
    plt.plot(df["MatchNum"], df["Elo"], marker='o', color='#4CAF50', linewidth=2, markersize=6)
    
    # Add labels and styling
    plt.title(f"Elo Rating Progression Timeline: {TARGET_PLAYER}", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Matches Played (Chronological)", fontsize=11, labelpad=10)
    plt.ylabel("Elo Rating", fontsize=11, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Label the X-axis points with match identifiers
    plt.xticks(df["MatchNum"], df["MatchCode"], rotation=45)
    plt.tight_layout()
    
    # Display the graph window on your desktop
    print(f"📉 Generating timeline visualization window for {TARGET_PLAYER}...")
    plt.show()

if __name__ == "__main__":
    chart_player_progression()

