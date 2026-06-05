import pandas as pd
import numpy as np
from pymongo import MongoClient

def calculate_smoothed_win_rate(wins, matches, prior_weight=2):
    if matches == 0:
        return 0.50
    return (wins + prior_weight) / (matches + (prior_weight * 2))

def generate_power_leaderboard():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No match history found.")
        return

    # 1. Gather latest states
    player_latest_states = {}
    for match in matches:
        w_name = match["winning_player"]
        l_name = match["losing_player"]
        
        player_latest_states[w_name] = {
            "alltime_wins": match["winner_alltime_wins"] + 1,
            "alltime_matches": match["winner_alltime_matches"] + 1,
            "league_wins": match.get("winner_n_wins", 0),
            "league_matches": match.get("winner_n_matches", 0),
            "elo": match["winner_elo"]
        }
        player_latest_states[l_name] = {
            "alltime_wins": match["loser_alltime_wins"],
            "alltime_matches": match["loser_alltime_matches"] + 1,
            "league_wins": match.get("loser_n_wins", 0),
            "league_matches": match.get("loser_n_matches", 0),
            "elo": match["loser_elo"]
        }

    # 2. Build population matrix to calculate Means and Standard Deviations
    rows = []
    for name, stats in player_latest_states.items():
        rows.append({
            "Player": name,
            "alltime_winrate": calculate_smoothed_win_rate(stats["alltime_wins"], stats["alltime_matches"]),
            "league_winrate": calculate_smoothed_win_rate(stats["league_wins"], stats["league_matches"]),
            "alltime_experience": stats["alltime_matches"],
            "league_experience": stats["league_matches"],
            "pre_match_elo": stats["elo"],
            "raw_stats": stats # preserve raw dict for later
        })
    df_pop = pd.DataFrame(rows)

    # 🔍 CRITICAL FIX: Extract means and standard deviations to mimic StandardScaler exactly
    means = df_pop.mean(numeric_only=True)
    stds = df_pop.std(numeric_only=True)
    
    # Handle zero-variance edge cases safely to prevent division by zero
    stds[stds == 0] = 1.0

    # 3. Model weights
    COEFFICIENTS = {
        "alltime_winrate_diff": 2.2809,
        "league_winrate_diff": -0.2347,
        "alltime_experience_diff": 0.1586,
        "league_experience_diff": 0.1831,
        "pre_match_elo_diff": -1.4443
    }

    leaderboard_data = []
    
    # 4. Score players using scaled values
    for idx, p in df_pop.iterrows():
        # Scale the inputs exactly like scikit-learn: (Value - Mean) / StdDev
        scaled_alltime_wr = (p["alltime_winrate"] - means["alltime_winrate"]) / stds["alltime_winrate"]
        scaled_league_wr = (p["league_winrate"] - means["league_winrate"]) / stds["league_winrate"]
        scaled_alltime_exp = (p["alltime_experience"] - means["alltime_experience"]) / stds["alltime_experience"]
        scaled_league_exp = (p["league_experience"] - means["league_experience"]) / stds["league_experience"]
        scaled_elo = (p["pre_match_elo"] - means["pre_match_elo"]) / stds["pre_match_elo"]

        # Run logistic calculation on standardized scales
        log_odds = (
            (COEFFICIENTS["alltime_winrate_diff"] * scaled_alltime_wr) +
            (COEFFICIENTS["league_winrate_diff"] * scaled_league_wr) +
            (COEFFICIENTS["alltime_experience_diff"] * scaled_alltime_exp) +
            (COEFFICIENTS["league_experience_diff"] * scaled_league_exp) +
            (COEFFICIENTS["pre_match_elo_diff"] * scaled_elo)
        )
        
        # Sigmoid calculation to get a smooth, non-infinite probability curve
        win_probability = 1 / (1 + np.exp(-log_odds))
        power_rating = round(1000 + (win_probability - 0.5) * 1000)
        
        raw = p["raw_stats"]
        leaderboard_data.append({
            "Player": p["Player"],
            "PowerRating": power_rating,
            "WinProbVsAvg": f"{win_probability * 100:.1f}%",
            "Elo": raw["elo"],
            "AllTimeRecord": f"{raw['alltime_wins']}-{raw['alltime_matches'] - raw['alltime_wins']}",
            "CurrentLeague": f"{raw['league_wins']}-{raw['league_matches'] - raw['league_wins']}"
        })

    df_leaderboard = pd.DataFrame(leaderboard_data).sort_values(by="PowerRating", ascending=False)
    
    print("\n==================================== 🏆 MODEL POWER RATING LEADERBOARD ====================================")
    print(f"{'Rank'.ljust(6)}{'Player Name'.ljust(20)}{'Power Rating'.ljust(15)}{'Win Prob vs Avg'.ljust(18)}{'Current Elo'.ljust(14)}{'All-Time'.ljust(12)}{'Active League'}")
    print("-----------------------------------------------------------------------------------------------------------")
    
    for rank, (_, row) in enumerate(df_leaderboard.iterrows(), start=1):
        print(f"{str(rank).ljust(6)}{row['Player'].ljust(20)}{str(row['PowerRating']).ljust(15)}{row['WinProbVsAvg'].ljust(18)}{str(row['Elo']).ljust(14)}{row['AllTimeRecord'].ljust(12)}{row['CurrentLeague']}")
    print("===========================================================================================================\n")

if __name__ == "__main__":
    generate_power_leaderboard()

