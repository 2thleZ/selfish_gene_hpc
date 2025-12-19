import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_stats(csv_file="results.csv", output_dir="plots"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("Reading data...")
    df = pd.read_csv(csv_file)
    
    # 1. Plot Traits Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['avg_rep'], label='Avg Rep Rate', color='red')
    plt.plot(df['step'], df['avg_death'], label='Avg Death Rate', color='green')
    plt.plot(df['step'], df['avg_mut'], label='Avg Mut Rate', color='blue')
    
    plt.xlabel('Generation (Step)')
    plt.ylabel('Rate')
    plt.title('Evolution of Avg Traits over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    traits_path = os.path.join(output_dir, "evolution_traits.png")
    plt.savefig(traits_path)
    print(f"Saved {traits_path}")
    
    # 2. Plot Population
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['pop'], label='Population', color='black')
    plt.xlabel('Generation (Step)')
    plt.ylabel('Count')
    plt.title('Population Growth')
    plt.grid(True, alpha=0.3)
    
    pop_path = os.path.join(output_dir, "population_growth.png")
    plt.savefig(pop_path)
    print(f"Saved {pop_path}")

if __name__ == "__main__":
    plot_stats()
