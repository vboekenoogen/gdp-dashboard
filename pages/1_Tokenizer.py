# Install required packages
!pip install nltk

import pandas as pd
import nltk
import re

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)

# Read raw data
df = pd.read_csv('ig_posts_raw_mini.csv')

# Process data
rows = []
for _, row in df.iterrows():
    post_id = row['shortcode']
    caption = str(row['caption']).strip()
    
    # Clean text
    caption = re.sub(r'\s+', ' ', caption)
    
    # Tokenize sentences
    sentences = nltk.sent_tokenize(caption)
    
    # Add each sentence as a row
    for i, sentence in enumerate(sentences, 1):
        if sentence.strip():
            rows.append({
                'ID': post_id,
                'Sentence ID': i,
                'Context': caption,
                'Statement': sentence.strip()
            })

# Create and save result
result = pd.DataFrame(rows)
result.to_csv('ig_posts_processed.csv', index=False)

print(f"Processed {len(df)} posts into {len(result)} sentences")
print(result.head())
