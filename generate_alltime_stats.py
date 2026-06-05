from pymongo import MongoClient, ASCENDING

def calculate_k_factor(matches_played):
    """Returns a variable K-factor based on prior match experience."""
    if matches_played < 10:
        return 40
    elif matches_played < 100:
        return 20
    else:
        return 10

def calculate_new_elos(winner_old_elo, loser_old_elo, winner_k, loser_k):
    """Executes the standard Elo rating formula."""
    # 1. Calculate Expected Scores (Probability of winning)
    expected_winner = 1 / (1 + 10 ** ((loser_old_elo - winner_old_elo) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_old_elo - loser_old_elo) / 400))
    
    # 2. Update Ratings based on actual results (Winner = 1, Loser = 0)
    # Using round() keeps the database keys clean and readable
    winner_new_elo = round(winner_old_elo + winner_k * (1 - expected_winner))
    loser_new_elo = round(loser_old_elo + loser_k * (0 - expected_loser))
    
    return winner_new_elo, loser_new_elo

def generate_alltime_stats_and_elo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    
    source_collection = db["combined_matches"]
    target_collection = db["elo_matches"]
    
    # Clear out target collection for a clean chronological calculation
    target_collection.delete_many({})
    
    # Track dynamic player state. Fallback initialization is now 1000 Elo
    player_history = {}
    
    def get_player_stats(player_name):
        if player_name not in player_history:
            player_history[player_name] = {
                "wins": 0, 
                "losses": 0, 
                "matches": 0,
                "elo": 1000  # Dynamic starting Elo
            }
        return player_history[player_name]

    # Query and stream matches sorted chronologically
    match_cursor = source_collection.find().sort("Time", ASCENDING)
    bulk_updates = []
    
    print("Calculating chronological Elo changes...")
    for match in match_cursor:
        w_player = match.get("winning_player")
        l_player = match.get("losing_player")
        
        # 1. Retrieve stats BEFORE this current match is evaluated
        w_stats = get_player_stats(w_player)
        l_stats = get_player_stats(l_player)

        # 🔍 NEW & CRITICAL: Save their absolute true Elo BEFORE the match happens
        match["winner_pre_match_elo"] = w_stats["elo"]
        match["loser_pre_match_elo"] = l_stats["elo"]

        # 2. Get current individual K-Factors based on prior matches
        w_k = calculate_k_factor(w_stats["matches"])
        l_k = calculate_k_factor(l_stats["matches"])
        
        # 3. Calculate new post-match Elo ratings
        w_new_elo, l_new_elo = calculate_new_elos(
            winner_old_elo=w_stats["elo"],
            loser_old_elo=l_stats["elo"],
            winner_k=w_k,
            loser_k=l_k
        )
        
        # 4. Inject post-match Elo values directly into the entry row
        match["winner_elo"] = w_new_elo
        match["loser_elo"] = l_new_elo
        
        # 5. Update running state trackers for the next chronological iteration
        w_stats["elo"] = w_new_elo
        
        l_stats["elo"] = l_new_elo
        
        w_stats["wins"] += 1
        w_stats["matches"] += 1
        l_stats["losses"] += 1
        l_stats["matches"] += 1

        # Save historical counts to the match object
        match["winner_alltime_wins"] = w_stats["wins"]
        match["winner_alltime_losses"] = w_stats["losses"]
        match["winner_alltime_matches"] = w_stats["matches"]
        match["loser_alltime_wins"] = l_stats["wins"]
        match["loser_alltime_losses"] = l_stats["losses"]
        match["loser_alltime_matches"] = l_stats["matches"]



        player_history[w_player] = w_stats
        player_history[l_player] = l_stats
        
        bulk_updates.append(match)
        
    if bulk_updates:
        target_collection.insert_many(bulk_updates)
        print(f"✅ Successfully calculated ratings! Added {len(bulk_updates)} matches to 'elo_matches'.")
    else:
        print("❌ No source matches found to process.")

if __name__ == "__main__":
    generate_alltime_stats_and_elo()

