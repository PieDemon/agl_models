import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def calculate_smoothed_win_rate(wins, matches, prior_weight=2):
    if matches == 0:
        return 0.50
    return (wins + prior_weight) / (matches + (prior_weight * 2))

def train_perfect_5_variable_predictor():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No matches found. Please run calculate_stats.py first.")
        return

    features = []
    targets = []
    
    print("Building 100% leak-proof 5-variable model...")
    for match in matches:
        # Check if the new pre-match keys exist
        if "winner_pre_match_elo" not in match:
            print("❌ Error: 'winner_pre_match_elo' field missing. Did you rerun calculate_stats.py?")
            return
            
        # Fetch pre-match metrics
        w_alltime_w = match["winner_alltime_wins"]
        w_alltime_m = match["winner_alltime_matches"]
        l_alltime_w = match["loser_alltime_wins"]
        l_alltime_m = match["loser_alltime_matches"]
        
        w_league_w = match.get("winner_n_wins", 0) - 1
        w_league_m = match.get("winner_n_matches", 0) - 1
        l_league_w = match.get("loser_n_wins", 0)
        l_league_m = match.get("loser_n_matches", 0) - 1

        w_alltime_rate = calculate_smoothed_win_rate(w_alltime_w, w_alltime_m)
        l_alltime_rate = calculate_smoothed_win_rate(l_alltime_w, l_alltime_m)
        w_league_rate = calculate_smoothed_win_rate(w_league_w, w_league_m)
        l_league_rate = calculate_smoothed_win_rate(l_league_w, l_league_m)
        
        # 🛡️ PURE HONEST DATA READ: No subtraction math guesses!
        w_pre_elo = match["winner_pre_match_elo"]
        l_pre_elo = match["loser_pre_match_elo"]
        
        # Perspective 1: Winner vs Loser
        features.append({
            "alltime_winrate_diff": w_alltime_rate - l_alltime_rate,
            "league_winrate_diff": w_league_rate - l_league_rate,
            "alltime_experience_diff": w_alltime_m - l_alltime_m,
            "league_experience_diff": w_league_m - l_league_m,
            "pre_match_elo_diff": w_pre_elo - l_pre_elo  # Pure strength of schedule indicator
        })
        targets.append(1)
        
        # Perspective 2: Loser vs Winner (Inverted)
        features.append({
            "alltime_winrate_diff": l_alltime_rate - w_alltime_rate,
            "league_winrate_diff": l_league_rate - w_league_rate,
            "alltime_experience_diff": l_alltime_m - w_alltime_m,
            "league_experience_diff": l_league_m - w_league_m,
            "pre_match_elo_diff": l_pre_elo - w_pre_elo
        })
        targets.append(0)

    df = pd.DataFrame(features)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(penalty='l2')
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print(f"Match Prediction Accuracy: {accuracy * 100:.2f}%")
    print("------------------------------------------------------------------")
    print("Standardized Coefficients:")
    
    for col, coef in zip(df.columns, model.coef_.ravel()):
        direction = "🟢 Increases Win Chance" if coef > 0 else "🔴 Decreases Win Chance"
        print(f"  • {col.ljust(25)}: {coef:+.4f} -> {direction}")
    print("==================================================================\n")

if __name__ == "__main__":
    train_perfect_5_variable_predictor()

