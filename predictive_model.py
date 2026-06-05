import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_mtg_predictor():
    # 1. Connect to MongoDB and fetch match history
    client = MongoClient("mongodb://localhost:27017/")
    db = client["agl_data"]
    collection = db["elo_matches"]
    
    matches = list(collection.find({}))
    if not matches:
        print("❌ No matches found in 'elo_matches'. Run your Elo script first.")
        return

    # 2. Build a symmetric dataset from the match records
    # We must train the model on both wins AND losses from a neutral perspective
    features = []
    targets = [] # 1 for Player A wins, 0 for Player A loses
    
    print("Extracting features and calculating league advantages...")
    for match in matches:
        # Player A = Winner, Player B = Loser
        features.append({
            "elo_diff": match["winner_elo"] - match["loser_elo"],
            "matches_diff": match["winner_alltime_matches"] - match["loser_alltime_matches"],
            # The catch-up mechanic driver: Difference in historical losses (extra packs)
            "losses_diff": match["winner_alltime_losses"] - match["loser_alltime_losses"]
        })
        targets.append(1) # Player A won
        
        # Mirror the match (Player A = Loser, Player B = Winner)
        features.append({
            "elo_diff": match["loser_elo"] - match["winner_elo"],
            "matches_diff": match["loser_alltime_matches"] - match["winner_alltime_matches"],
            "losses_diff": match["loser_alltime_losses"] - match["winner_alltime_losses"]
        })
        targets.append(0) # Player A lost

    df = pd.DataFrame(features)
    y = np.array(targets)

    # 3. Split the data into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # 4. Train the Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate how well our formula predicts reality
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n================ 📊 MODEL RESULTS ================")
    print(f"Match Prediction Accuracy: {accuracy * 100:.2f}%")
    print("--------------------------------------------------")
    print("Formula Weight Significance (Coefficients):")
    
    # Coefficients tell us how much a 1-unit change in a variable swings win probability
    for col, coef in zip(df.columns, model.coef_[0]):
        importance = "Positive (Increases Win Chance)" if coef > 0 else "Negative (Decreases Win Chance)"
        print(f"  • {col.ljust(15)}: {coef:.4f} -> {importance}")
        
    print("==================================================\n")

    # 6. Show a live prediction formula example
    print("🔮 Test Formula Example:")
    # Suppose Player 1 has 1050 Elo, 12 matches, 4 losses
    # Suppose Player 2 has 1000 Elo, 10 matches, 6 losses (Player 2 has +2 extra booster packs!)
    test_match = pd.DataFrame([{
        "elo_diff": 1050 - 1000,   # +50 Elo
        "matches_diff": 12 - 10,   # +2 Matches experience
        "losses_diff": 4 - 6       # -2 Losses (Disadvantage: Player 2 has more packs!)
    }])
    
    prob = model.predict_proba(test_match)[0][1]
    print(f"Player 1's calculated probability of winning this match: {prob * 100:.1f}%")

if __name__ == "__main__":
    train_mtg_predictor()

