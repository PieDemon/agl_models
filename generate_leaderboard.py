import pandas as pd
import numpy as np
from pymongo import MongoClient

def calculate_smoothed_win_rate(wins, matches, prior_weight=2):
    if matches == 0:
        return 0.50
    return (wins + prior_weight) / (matches + (prior_weight * 2))

def generate_power_leaderboard():
    # 1. Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No match history found. Please run your database setup scripts first.")
        return

    # 2. Extract every player's absolute LATEST state from the chronological timeline
    player_latest_states = {}
    
    for match in matches:
        w_name = match["winning_player"]
        l_name = match["losing_player"]
        
        # Track winner's final state after this match concluded
        player_latest_states[w_name] = {
            "alltime_wins": match["winner_alltime_wins"] + 1,
            "alltime_matches": match["winner_alltime_matches"] + 1,
            "league_wins": match.get("winner_n_wins", 0),
            "league_matches": match.get("winner_n_matches", 0),
            "elo": match["winner_elo"]
        }
        
        # Track loser's final state after this match concluded
        player_latest_states[l_name] = {
            "alltime_wins": match["loser_alltime_wins"],
            "alltime_matches": match["loser_alltime_matches"] + 1,
            "league_wins": match.get("loser_n_wins", 0),
            "league_matches": match.get("loser_n_matches", 0),
            "elo": match["loser_elo"]
        }

    if not player_latest_states:
        print("❌ Failed to parse player states.")
        return

    # 3. Define the exact standardized weights discovered by your 5-variable model
    # (These match your 68.13% leak-proof verification run exactly)
    COEFFICIENTS = {
        "alltime_winrate_diff": 2.2809,
        "league_winrate_diff": -0.2347,
        "alltime_experience_diff": 0.1586,
        "league_experience_diff": 0.1831,
        "pre_match_elo_diff": -1.4443
    }
    INTERCEPT = 0.0  # Symmetric baseline probability anchor

    # 4. Calculate the average baseline player profile to match against head-to-head
    all_profiles = []
    for name, stats in player_latest_states.items():
        all_profiles.append({
            "alltime_rate": calculate_smoothed_win_rate(stats["alltime_wins"], stats["alltime_matches"]),
            "league_rate": calculate_smoothed_win_rate(stats["league_wins"], stats["league_matches"]),
            "alltime_m": stats["alltime_matches"],
            "league_m": stats["league_matches"],
            "elo": stats["elo"]
        })
    
    df_avg = pd.DataFrame(all_profiles)
    avg_alltime_rate = df_avg["alltime_rate"].mean()
    avg_league_rate = df_avg["league_rate"].mean()
    avg_alltime_m = df_avg["alltime_m"].mean()
    avg_league_m = df_avg["league_m"].mean()
    avg_elo = df_avg["elo"].mean()

    # 5. Score every player against the league baseline average
    leaderboard_data = []
    
    for name, stats in player_latest_states.items():
        # Compute individual metrics
        p_alltime_rate = calculate_smoothed_win_rate(stats["alltime_wins"], stats["alltime_matches"])
        p_league_rate = calculate_smoothed_win_rate(stats["league_wins"], stats["league_matches"])
        
        # Calculate relative delta differences against the absolute average league opponent
        diff_alltime_rate = p_alltime_rate - avg_alltime_rate
        diff_league_rate = p_league_rate - avg_league_rate
        diff_alltime_m = stats["alltime_matches"] - avg_alltime_m
        diff_league_m = stats["league_matches"] - avg_league_m
        diff_elo = stats["elo"] - avg_elo
        
        # Run standard Logistic Regression log-odds prediction math:
        # z = Intercept + (β1 * x1) + (β2 * x2) + ...
        log_odds = (
            INTERCEPT +
            (COEFFICIENTS["alltime_winrate_diff"] * diff_alltime_rate) +
            (COEFFICIENTS["league_winrate_diff"] * diff_league_rate) +
            (COEFFICIENTS["alltime_experience_diff"] * diff_alltime_m) +
            (COEFFICIENTS["league_experience_diff"] * diff_league_m) +
            (COEFFICIENTS["pre_match_elo_diff"] * diff_elo)
        )
        
        # Convert log-odds to a clean 0% to 100% win probability curve (Sigmoid Function)
        win_probability = 1 / (1 + np.exp(-log_odds))
        # Turn into a normalized power rating scale centering at 1000
        power_rating = round(1000 + (win_probability - 0.5) * 1000)
        
        leaderboard_data.append({
            "Player": name,
            "PowerRating": power_rating,
            "WinProbVsAvg": f"{win_probability * 100:.1f}%",
            "Elo": stats["elo"],
            "AllTimeRecord": f"{stats['alltime_wins']}-{stats['alltime_matches'] - stats['alltime_wins']}",
            "CurrentLeague": f"{stats['league_wins']}-{stats['league_matches'] - stats['league_wins']}"
        })

    # 6. Sort and display leaderboard layout
    df_leaderboard = pd.DataFrame(leaderboard_data).sort_values(by="PowerRating", ascending=False)
    
    print("\n==================================== 🏆 MODEL POWER RATING LEADERBOARD ====================================")
    print(f"{'Rank'.ljust(6)}{'Player Name'.ljust(20)}{'Power Rating'.ljust(15)}{'Win Prob vs Avg'.ljust(18)}{'Current Elo'.ljust(14)}{'All-Time'.ljust(12)}{'Active League'}")
    print("-----------------------------------------------------------------------------------------------------------")
    
    for rank, (_, row) in enumerate(df_leaderboard.iterrows(), start=1):
        # Format strings cleanly to make standard output perfectly aligned
        print(f"{str(rank).ljust(6)}{row['Player'].ljust(20)}{str(row['PowerRating']).ljust(15)}{row['WinProbVsAvg'].ljust(18)}{str(row['Elo']).ljust(14)}{row['AllTimeRecord'].ljust(12)}{row['CurrentLeague']}")
    print("===========================================================================================================\n")

if __name__ == "__main__":
    generate_power_leaderboard()

