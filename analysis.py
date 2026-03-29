import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline
import torch

# 1. Setup & Data Loading
try:
    df = pd.read_csv('data/trending_topics_2026_synthetic.csv')
    print("✅ Data Loaded Successfully")
except FileNotFoundError:
    print("❌ Error: Please ensure the dataset is in the /data folder.")
    exit()

# 2. AI Layer: Transformer-based Sentiment Analysis
print("🤖 Initializing RoBERTa Transformer...")
# Using a model optimized for social media context
sentiment_task = pipeline("sentiment-analysis", 
                          model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                          device=-1) # Set to 0 if using GPU

def get_pro_sentiment(text):
    if pd.isna(text): return 0
    try:
        result = sentiment_task(str(text)[:512])[0]
        score = result['score']
        if result['label'] == 'negative': return -score
        if result['label'] == 'positive': return score
        return 0
    except:
        return 0

print("🧠 Running AI Sentiment Analysis (this may take a moment)...")
df['ai_polarity_score'] = df['short_text'].apply(get_pro_sentiment)

# 3. Network Science Layer: Bipartite Graph Construction
print("🕸️ Building Interaction Network...")
G = nx.Graph()

# Creating connections between Countries and Topics
for _, row in df.iterrows():
    # Logic to extract Country and Topic (Assumes these columns exist or are parsed)
    country = row.get('country', 'Unknown')
    topic = row.get('topic_category', 'General')
    
    # Add Edge with Weight based on AI Sentiment
    G.add_edge(country, topic, weight=row['ai_polarity_score'])

# 4. Visualization: The "Creative Media" Output
print("🎨 Generating Final Visualization...")
plt.figure(figsize=(14, 10))

# Spring layout for natural clustering of "Echo Chambers"
pos = nx.spring_layout(G, k=0.6, seed=42)

# Separate node types for distinct coloring
countries = [n for n in G.nodes() if len(str(n)) <= 3] # Short codes like PK, US, GB
topics = [n for n in G.nodes() if len(str(n)) > 3]

# Draw Nodes
nx.draw_networkx_nodes(G, pos, nodelist=countries, node_color='orange', 
                       node_size=2000, label='Countries')
nx.draw_networkx_nodes(G, pos, nodelist=topics, node_color='beige', 
                       node_size=800, label='Topics')

# Draw Edges & Labels
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=2.0)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.title("Echo-Chamber-Intelligence: Global Topic Influence Network (2026)", fontsize=15)
plt.legend(scatterpoints=1)
plt.axis('off')

# Save and Show
plt.savefig('images/final_network_graph.png', dpi=300)
print("🚀 Analysis Complete! Image saved to /images folder.")
plt.show()
