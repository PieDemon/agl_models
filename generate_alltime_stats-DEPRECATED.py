from pymongo import MongoClient, ASCENDING

def generate_alltime_stats():
    # 1. Connect to your local MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    
    source_collection = db["combined_matches"]
    target_collection = db["elo_matches"]
    
    # Clear out target collection if rerunning the script
    target_collection.delete_many({})
    
    # 2. Initialize a local tracking dictionary for player histories
    # Structure: { "Player Name": {"wins": 0, "losses": 0, "matches": 0} }
    player_history = {}
    
    def get_player_stats(player_name):
        """Helper to fetch stats or initialize a new player profile."""
        if player_name not in player_history:
            player_history[player_name] = {"wins": 0, "losses": 0, "matches": 0}
        return player_history[player_name]

    # 3. Stream through matches sorted strictly by Time (oldest to newest)
    match_cursor = source_collection.find().sort("Time", ASCENDING)
    
    bulk_updates = []
    
    print("Processing matches chronologically...")
    for match in match_cursor:
        w_player = match.get("Winning_Player")
        l_player = match.get("Losing_Player")
        
        # Pull up-to-date all-time stats BEFORE this match happens
        w_stats = get_player_stats(w_player)
        l_stats = get_player_stats(l_player)

        # 4. Increment the state values for the NEXT match calculation
        w_stats["wins"] += 1
        w_stats["matches"] += 1

        l_stats["losses"] += 1
        l_stats["matches"] += 1
        
        # Inject the "alltime" history state into the match object
        match["winner_alltime_wins"] = w_stats["wins"]
        match["winner_alltime_losses"] = w_stats["losses"]
        match["winner_alltime_matches"] = w_stats["matches"]
        
        match["loser_alltime_wins"] = l_stats["wins"]
        match["loser_alltime_losses"] = l_stats["losses"]
        match["loser_alltime_matches"] = l_stats["matches"]
        
#        # 4. Increment the state values for the NEXT match calculation
#        w_stats["wins"] += 1
#        w_stats["matches"] += 1
#        
#        l_stats["losses"] += 1
#        l_stats["matches"] += 1
        
        # Keep our local tracker state aligned
        player_history[w_player] = w_stats
        player_history[l_player] = l_stats
        
        # Prepare the modified document for insertion
        bulk_updates.append(match)
        
    # 5. Bulk insert the enhanced data into the new ELO collection
    if bulk_updates:
        target_collection.insert_many(bulk_updates)
        print(f"✅ Successfully created 'elo_matches' with {len(bulk_updates)} entries!")
    else:
        print("❌ No source matches found to process.")

if __name__ == "__main__":
    generate_alltime_stats()

