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

    # 1. Extract latest player states
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

    # 2. Build a clean list of individual player profiles
    players_list = []
    for name, stats in player_latest_states.items():
        players_list.append({
            "name": name,
            "alltime_winrate": calculate_smoothed_win_rate(stats["alltime_wins"], stats["alltime_matches"]),
            "league_winrate": calculate_smoothed_win_rate(stats["league_wins"], stats["league_matches"]),
            "alltime_experience": stats["alltime_matches"],
            "league_experience": stats["league_matches"],
            "pre_match_elo": stats["elo"],
            "raw_dict": stats
        })

    # 3. Paste your exact RAW unscaled coefficients from your 68.08% run
    COEFFICIENTS = {
        "alltime_winrate_diff": 13.361626,
        "league_winrate_diff": -1.323635,
        "alltime_experience_diff": 0.000546,
        "league_experience_diff": 0.068337,
        "pre_match_elo_diff": -0.011878
    }
    INTERCEPT = 0.001624

    leaderboard_data = []

    # 4. 🔍 REAL FIXED MULTI-PLAYER MATCHUP ENGINE:
    # Simulates a direct 1v1 matchup for Player A against every possible Player B
    print("Simulating full league round-robin gauntlet...")
    for p_a in players_list:
        individual_matchup_probabilities = []
        
        for p_b in players_list:
            if p_a["name"] == p_b["name"]:
                continue # Skip playing against yourself
                
            # Calculate the true delta difference between these two specific opponents
            diff_alltime_wr = p_a["alltime_winrate"] - p_b["alltime_winrate"]
            diff_league_wr = p_a["league_winrate"] - p_b["league_winrate"]
            diff_alltime_exp = p_a["alltime_experience"] - p_b["alltime_experience"]
            diff_league_exp = p_a["league_experience"] - p_b["league_experience"]
            diff_elo = p_a["pre_match_elo"] - p_b["pre_match_elo"]

            # Calculate log-odds for this explicit 1v1 pairing
            log_odds = (
                INTERCEPT +
                (COEFFICIENTS["alltime_winrate_diff"] * diff_alltime_wr) +
                (COEFFICIENTS["league_winrate_diff"] * diff_league_wr) +
                (COEFFICIENTS["alltime_experience_diff"] * diff_alltime_exp) +
                (COEFFICIENTS["league_experience_diff"] * diff_league_exp) +
                (COEFFICIENTS["pre_match_elo_diff"] * diff_elo)
            )
            
            # Pure, honest 1v1 probability
            prob_a_beats_b = 1 / (1 + np.exp(-log_odds))
            individual_matchup_probabilities.append(prob_a_beats_b)
            
        # Average all their simulated head-to-head matches together
        avg_win_probability = np.mean(individual_matchup_probabilities)
        
        # Power Rating scales naturally: 50% average win rate = 1000 Power Rating
        power_rating = round(1000 + (avg_win_probability - 0.5) * 1000)
        
        raw = p_a["raw_dict"]
        leaderboard_data.append({
            "Player": p_a["name"],
            "PowerRating": power_rating,
            "AvgExpectedWin": f"{avg_win_probability * 100:.1f}%",
            "Elo": raw["elo"],
            "AllTimeRecord": f"{raw['alltime_wins']}-{raw['alltime_matches'] - raw['alltime_wins']}",
            "CurrentLeague": f"{raw['league_wins']}-{raw['league_matches'] - raw['league_wins']}"
        })

    df_leaderboard = pd.DataFrame(leaderboard_data).sort_values(by="PowerRating", ascending=False)
    
    print("\n==================================== 🏆 MODEL POWER RATING LEADERBOARD ====================================")
    print(f"{'Rank'.ljust(6)}{'Player Name'.ljust(20)}{'Power Rating'.ljust(15)}{'Avg Expected Win'.ljust(18)}{'Current Elo'.ljust(14)}{'All-Time'.ljust(12)}{'Active League'}")
    print("-----------------------------------------------------------------------------------------------------------")
    
    for rank, (_, row) in enumerate(df_leaderboard.iterrows(), start=1):
        print(f"{str(rank).ljust(6)}{row['Player'].ljust(20)}{str(row['PowerRating']).ljust(15)}{row['AvgExpectedWin'].ljust(18)}{str(row['Elo']).ljust(14)}{row['AllTimeRecord'].ljust(12)}{row['CurrentLeague']}")
    print("===========================================================================================================\n")

if __name__ == "__main__":
    generate_power_leaderboard()

