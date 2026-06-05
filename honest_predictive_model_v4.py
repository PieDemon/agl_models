import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def calculate_smoothed_win_rate(wins, matches, prior_weight=2):
    """Calculates a regularized win rate using Laplace smoothing."""
    if matches == 0:
        return 0.50
    smoothed_wins = wins + prior_weight
    smoothed_matches = matches + (prior_weight * 2)
    return smoothed_wins / smoothed_matches

def train_expanded_rate_predictor():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No matches found in 'elo_matches'.")
        return

    features = []
    targets = []
    
    print("Extracting honest 6-variable feature matrix from MongoDB...")
    for match in matches:
        # --- 1. Fetch Raw Pre-Match Career Counters ---
        w_alltime_w = match["winner_alltime_wins"]
        w_alltime_m = match["winner_alltime_matches"]
        l_alltime_w = match["loser_alltime_wins"]
        l_alltime_m = match["loser_alltime_matches"]
        
        # --- 2. Fetch Raw Pre-Match League Counters ---
        w_league_w = match.get("winner_n_wins", 0) - 1
        w_league_m = match.get("winner_n_matches", 0) - 1
        l_league_w = match.get("loser_n_wins", 0)
        l_league_m = match.get("loser_n_matches", 0) - 1

        # --- 3. Compute Regularized Rates ---
        w_alltime_rate = calculate_smoothed_win_rate(w_alltime_w, w_alltime_m)
        l_alltime_rate = calculate_smoothed_win_rate(l_alltime_w, l_alltime_m)
        
        w_league_rate = calculate_smoothed_win_rate(w_league_w, w_league_m)
        l_league_rate = calculate_smoothed_win_rate(l_league_w, l_league_m)
        
        # --- 4. Build Perspectives Matrix (6 Features Total) ---
        # Perspective 1: Winner vs Loser
        features.append({
            # The original 4 variables
            "alltime_winrate_diff": w_alltime_rate - l_alltime_rate,
            "league_winrate_diff": w_league_rate - l_league_rate,
            "alltime_experience_diff": w_alltime_m - l_alltime_m,
            "league_experience_diff": w_league_m - l_league_m,
            
            # 🔍 NEW: Raw total volume metrics for the individual player (Winner)
            # This lets the model evaluate if simply being a high-volume veteran or high-volume active grifter matter
            "player_total_matches_alltime": w_alltime_m,
            "player_total_matches_league": w_league_m
        })
        targets.append(1)
        
        # Perspective 2: Loser vs Winner (Inverted)
        features.append({
            "alltime_winrate_diff": l_alltime_rate - w_alltime_rate,
            "league_winrate_diff": l_league_rate - w_league_rate,
            "alltime_experience_diff": l_alltime_m - w_alltime_m,
            "league_experience_diff": l_league_m - w_league_m,
            
            # Inverted player baseline perspective (Loser)
            "player_total_matches_alltime": l_alltime_m,
            "player_total_matches_league": l_league_m
        })
        targets.append(0)

    df = pd.DataFrame(features)
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # Scale values to handle different ranges cleanly (Percentages vs Raw match counts)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a regularized model
    model = LogisticRegression(penalty='l2')
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print("\n================ 🛡️ 6-VARIABLE MODEL RESULTS ================")
    print(f"Honest Match Prediction Accuracy: {accuracy * 100:.2f}%")
    print("------------------------------------------------------------------")
    print("Standardized Coefficients (True Power Comparison):")
    
    for col, coef in zip(df.columns, model.coef_.ravel()):
        direction = "🟢 Increases Win Chance" if coef > 0 else "🔴 Decreases Win Chance"
        print(f"  • {col.ljust(30)}: {coef:+.4f} -> {direction}")
    print("==================================================================\n")

if __name__ == "__main__":
    train_expanded_rate_predictor()

