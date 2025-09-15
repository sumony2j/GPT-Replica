import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_info_table(log_file):
    with open(log_file,mode="r",encoding="utf-8") as f:
        data = f.readlines()

    rows = []
    cur_row = {}

    for line in data:
        if line.startswith("step"):
            parts = line.split(",")
            cur_row = {
                "Iteration" : int(parts[0].split(":")[1].strip()),
                "Train Loss" : float(parts[1].split(":")[1].strip()),
                "Learning Rate" : float(parts[2].split(":")[1].strip()),
                "Norm" : float(parts[3].split(":")[1].strip()),
                "Tokens/Sec" : float(parts[5].split(":")[1].strip()),
                "Val Loss" : None
            }
            rows.append(cur_row)
        elif line.startswith("validation"):
            val_loss = float(line.split(":")[1].strip())
            if rows:
                rows[-1]["Val Loss"] = val_loss

    all_df = pd.DataFrame(rows)
    df = all_df[all_df["Val Loss"].notna()]
    return df

def plot(df:pd.DataFrame,num_points=100):
    """
    plt.figure(figsize=(12, 6))
    
    df = df[df["Iteration"] % 50000 == 0]
    # Plot train & validation loss
    plt.plot(df["Iteration"], df["Train Loss"], label="Train Loss", color="blue", linewidth=1.5)
    plt.plot(df["Iteration"], df["Val Loss"], label="Validation Loss", color="orange", linewidth=1.5)
    #plt.plot(df["Norm"], label="Norm", color="red", linewidth=1.5)
    # Labels & Title
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs Validation Loss", fontsize=14)
    #plt.title("Learning rate", fontsize=14)
    #plt.title("Norm", fontsize=14)

    # Grid + Legend
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()
    """
    plt.figure(figsize=(12, 6))

    # Always keep first and last index
    idx = np.linspace(0, len(df) - 1, num_points, dtype=int)
    idx = np.unique(np.concatenate(([0], idx, [len(df)-1])))

    df_sampled = df.iloc[idx]

    # Plot train and validation loss
    plt.plot(df_sampled["Iteration"], df_sampled["Train Loss"], 
             label="Train Loss", color="blue", linewidth=1.5)
    plt.plot(df_sampled["Iteration"], df_sampled["Val Loss"], 
             label="Validation Loss", color="orange", linewidth=1.5)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Training vs Validation Loss (sampled {len(df_sampled)} points)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
log_file = "./StoryTeller_Lite"

plot(create_info_table(log_file))