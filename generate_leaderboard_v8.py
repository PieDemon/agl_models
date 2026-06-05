import pandas as pd
import numpy as np
from pymongo import MongoClient

def calculate_bayesian_shrunk_rate(wins, matches, global_mean=0.50, sample_anchor=25):
    """Keeps the solid Bayesian anchor to neutralize low-sample spikes."""
    if matches == 0:
        return 0.50
    return (wins + (sample_anchor * global_mean)) / (matches + sample_anchor)

def generate_power_leaderboard():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No match history found.")
        return

    # 1. Gather player states
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

    # 2. Build population array to compute dynamic real-world population scaling
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
    
    df_pop = pd.DataFrame(players_list)
    
    # 🔍 CRITICAL REAL FIX: Instead of hardcoded training scale factors, 
    # we calculate the true standard deviations of the active population.
    # This prevents the delta calculations from inflating by 9x inside the 1v1 loop.
    pop_stds = df_pop.std(numeric_only=True)
    pop_stds[pop_stds == 0] = 1.0 # Protect against zero variance

    # Verified clean weights from your 68.13% model run
    COEFFICIENTS = {
        "alltime_winrate_diff": 2.2809,
        "league_winrate_diff": -0.2347,
        "alltime_experience_diff": 0.1586,
        "league_experience_diff": 0.1831,
        "pre_match_elo_diff": -1.4443
    }
    INTERCEPT = 0.0

    leaderboard_data = []

    print("Simulating population-scaled round-robin gauntlet...")
    for p_a in players_list:
        individual_matchup_probabilities = []
        
        for p_b in players_list:
            if p_a["name"] == p_b["name"]:
                continue 
                
            # 1. Calculate raw delta difference between this pair
            raw_alltime_wr_diff = p_a["alltime_winrate"] - p_b["alltime_winrate"]
            raw_league_wr_diff = p_a["league_winrate"] - p_b["league_winrate"]
            raw_alltime_exp_diff = p_a["alltime_experience"] - p_b["alltime_experience"]
            raw_league_exp_diff = p_a["league_experience"] - p_b["league_experience"]
            raw_elo_diff = p_a["pre_match_elo"] - p_b["pre_match_elo"]

            # 2. Scale the differences honestly using the population's true variance limits
            scaled_alltime_wr = raw_alltime_wr_diff / pop_stds["alltime_winrate"]
            scaled_league_wr = raw_league_wr_diff / pop_stds["league_winrate"]
            scaled_alltime_exp = raw_alltime_exp_diff / pop_stds["alltime_experience"]
            scaled_league_exp = raw_league_exp_diff / pop_stds["league_experience"]
            scaled_elo = raw_elo_diff / pop_stds["pre_match_elo"]

            # 3. Compute unwarped log-odds
            log_odds = (
                INTERCEPT +
                (COEFFICIENTS["alltime_winrate_diff"] * scaled_alltime_wr) +
                (COEFFICIENTS["league_winrate_diff"] * scaled_league_wr) +
                (COEFFICIENTS["alltime_experience_diff"] * scaled_alltime_exp) +
                (COEFFICIENTS["league_experience_diff"] * scaled_league_exp) +
                (COEFFICIENTS["pre_match_elo_diff"] * scaled_elo)
            )
            
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

