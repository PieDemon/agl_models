import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_comprehensive_mtg_predictor():
    # 1. Connect to MongoDB and fetch match data
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No matches found in 'elo_matches'. Please run your collection scripts first.")
        return

    features = []
    targets = []
    
    print("Extracting current league and career variables from MongoDB...")
    for match in matches:
        # Perspective 1: Winner vs Loser
        features.append({
            "elo_diff": match["winner_elo"] - match["loser_elo"],
            
            # --- Current League / Season Differences ---
            "league_wins_diff": match.get("winner_n_wins", 0) - match.get("loser_n_wins", 0),
            "league_losses_diff": match.get("winner_n_losses", 0) - match.get("loser_n_losses", 0),
            "league_matches_diff": match.get("winner_n_matches", 0) - match.get("loser_n_matches", 0),
            
            # --- Career / All-Time Historical Differences ---
            "alltime_wins_diff": match["winner_alltime_wins"] - match["loser_alltime_wins"],
            "alltime_losses_diff": match["winner_alltime_losses"] - match["loser_alltime_losses"],
            "alltime_matches_diff": match["winner_alltime_matches"] - match["loser_alltime_matches"]
        })
        targets.append(1) # Player A (Winner) won
        
        # Perspective 2: Inverted Loser vs Winner
        features.append({
            "elo_diff": match["loser_elo"] - match["winner_elo"],
            
            # --- Current League / Season Differences (Inverted) ---
            "league_wins_diff": match.get("loser_n_wins", 0) - match.get("winner_n_wins", 0),
            "league_losses_diff": match.get("loser_n_losses", 0) - match.get("winner_n_losses", 0),
            "league_matches_diff": match.get("loser_n_matches", 0) - match.get("winner_n_matches", 0),
            
            # --- Career / All-Time Historical Differences (Inverted) ---
            "alltime_wins_diff": match["loser_alltime_wins"] - match["winner_alltime_wins"],
            "alltime_losses_diff": match["loser_alltime_losses"] - match["winner_alltime_losses"],
            "alltime_matches_diff": match["loser_alltime_matches"] - match["winner_alltime_matches"]
        })
        targets.append(0) # Player A (Loser) lost

    df = pd.DataFrame(features)
    y = np.array(targets)

    # 2. Split dataset into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # 🛡️ FIX MULTICOLLINEARITY & DATA REDUNDANCY:
    # Scale all features so the Elo differences don't overpower the single-digit match counts.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use 'l2' Ridge Regularization penalty to evenly distribute coefficient weights 
    # among the closely tied 'matches', 'wins', and 'losses' fields.
    model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 3. Evaluate results
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n================ 📊 REGULARIZED MODEL RESULTS ================")
    print(f"Match Prediction Accuracy: {accuracy * 100:.2f}%")
    print("--------------------------------------------------------------")
    print("Standardized Coefficients (Higher absolute value = Stronger overall impact):")
    
    # 4. Map and print variables and their directional weight
    for col, coef in zip(df.columns, model.coef_[0]):
        direction = "🟢 Increases Win Chance" if coef > 0 else "🔴 Decreases Win Chance"
        print(f"  • {col.ljust(22)}: {coef:+.4f} -> {direction}")
    print("==============================================================\n")

if __name__ == "__main__":
    train_comprehensive_mtg_predictor()

