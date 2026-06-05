import pandas as pd
import numpy as np
from pymongo import MongoClient

def calculate_bayesian_shrunk_rate(wins, matches, global_mean=0.50, sample_anchor=25):
    """
    Applies a Bayesian Shrinkage anchor. If a player has few matches, 
    their win rate is pulled back toward the 50% league average to 
    eliminate low-sample spikes.
    """
    if matches == 0:
        return 0.50
    return (wins + (sample_anchor * global_mean)) / (matches + sample_anchor)

def generate_power_leaderboard():
    # 1. Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No match history found in 'elo_matches'.")
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

    # Build player profile array
    players_list = []
    for name, stats in player_latest_states.items():
        players_list.append({
            "name": name,
            "alltime_winrate": calculate_bayesian_shrunk_rate(stats["alltime_wins"], stats["alltime_matches"]),
            "league_winrate": calculate_bayesian_shrunk_rate(stats["league_wins"], stats["league_matches"]),
            "alltime_experience": stats["alltime_matches"],
            "league_experience": stats["league_matches"],
            "pre_match_elo": stats["elo"],
            "raw_dict": stats
        })

    # 3. 🎯 YOUR EXACT LOG COEFFICIENTS FROM STEP 1 CALIBRATION
    COEFFICIENTS = {
        "alltime_winrate_diff": 2.396804,
        "league_winrate_diff": -0.216233,
        "alltime_experience_log_diff": 0.326673,
        "league_experience_log_diff": 0.163020,
        "pre_match_elo_diff": -1.600865
    }
    INTERCEPT = 0.0

    # 🎯 YOUR EXACT SCALER SCALE FACTORS FROM STEP 1 CALIBRATION
    SCALE_FACTORS = {
        "alltime_winrate_diff": 0.132578,
        "league_winrate_diff": 0.148575,
        "alltime_experience_log_diff": 1.816197,
        "league_experience_log_diff": 0.415572,
        "pre_match_elo_diff": 88.737497
    }

    leaderboard_data = []

    # 4. Simulated Round-Robin Gauntlet Engine
    print("Simulating log-calibrated round-robin gauntlet...")
    for p_a in players_list:
        individual_matchup_probabilities = []
        
        for p_b in players_list:
            if p_a["name"] == p_b["name"]:
                continue # Skip playing against yourself
                
            # Compute raw delta differences (Applying np.log1p for diminishing returns)
            raw_alltime_wr_diff = p_a["alltime_winrate"] - p_b["alltime_winrate"]
            raw_league_wr_diff = p_a["league_winrate"] - p_b["league_winrate"]
            raw_alltime_exp_diff = np.log1p(p_a["alltime_experience"]) - np.log1p(p_b["alltime_experience"])
            raw_league_exp_diff = np.log1p(p_a["league_experience"]) - np.log1p(p_b["league_experience"])
            raw_elo_diff = p_a["pre_match_elo"] - p_b["pre_match_elo"]

            # Standardize inputs using the explicit scale factor keys
            scaled_alltime_wr = raw_alltime_wr_diff / SCALE_FACTORS["alltime_winrate_diff"]
            scaled_league_wr = raw_league_wr_diff / SCALE_FACTORS["league_winrate_diff"]
            scaled_alltime_exp = raw_alltime_exp_diff / SCALE_FACTORS["alltime_experience_log_diff"]
            scaled_league_exp = raw_league_exp_diff / SCALE_FACTORS["league_experience_log_diff"]
            scaled_elo = raw_elo_diff / SCALE_FACTORS["pre_match_elo_diff"]

            # Compute unwarped log-odds
            log_odds = (
                INTERCEPT +
                (COEFFICIENTS["alltime_winrate_diff"] * scaled_alltime_wr) +
                (COEFFICIENTS["league_winrate_diff"] * scaled_league_wr) +
                (COEFFICIENTS["alltime_experience_log_diff"] * scaled_alltime_exp) +
                (COEFFICIENTS["league_experience_log_diff"] * scaled_league_exp) +
                (COEFFICIENTS["pre_match_elo_diff"] * scaled_elo)
            )
            
            # Pure standard sigmoid function mapping
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

    # 5. Sort and format final output presentation
    df_leaderboard = pd.DataFrame(leaderboard_data).sort_values(by="PowerRating", ascending=False)
    
    print("\n==================================== 🏆 MODEL POWER RATING LEADERBOARD ====================================")
    print(f"{'Rank'.ljust(6)}{'Player Name'.ljust(20)}{'Power Rating'.ljust(15)}{'Avg Expected Win'.ljust(18)}{'Current Elo'.ljust(14)}{'All-Time'.ljust(12)}{'Active League'}")
    print("-----------------------------------------------------------------------------------------------------------")
    
    for rank, (_, row) in enumerate(df_leaderboard.iterrows(), start=1):
        print(f"{str(rank).ljust(6)}{row['Player'].ljust(20)}{str(row['PowerRating']).ljust(15)}{row['AvgExpectedWin'].ljust(18)}{str(row['Elo']).ljust(14)}{row['AllTimeRecord'].ljust(12)}{row['CurrentLeague']}")
    print("===========================================================================================================\n")

if __name__ == "__main__":
    generate_power_leaderboard()

