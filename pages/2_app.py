import pandas as pd
import nltk
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    """Clean and prepare text for sentence tokenization"""
    if pd.isna(text):
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Replace multiple whitespaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

def tokenize_sentences(text):
    """Split text into individual sentences"""
    if not text:
        return []
    
    # Use NLTK sentence tokenizer
    sentences = nltk.sent_tokenize(text)
    
    # Clean sentences and remove empty ones
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Handle hashtags and mentions as separate sentences if they're standalone
            if sentence.startswith('#') and len(sentence.split()) > 1:
                # If hashtags are grouped, keep them as one sentence
                clean_sentences.append(sentence)
            elif sentence:
                clean_sentences.append(sentence)
    
    return clean_sentences

def transform_instagram_data(input_file, output_file):
    """Transform raw Instagram data to sentence-tokenized format"""
    
    # Read raw data
    df_raw = pd.read_csv(input_file)
    
    # Initialize list to store transformed rows
    transformed_rows = []
    
    # Process each post
    for _, row in df_raw.iterrows():
        post_id = row['shortcode']
        caption = preprocess_text(row['caption'])
        
        # Tokenize sentences
        sentences = tokenize_sentences(caption)
        
        # Create rows for each sentence
        for i, sentence in enumerate(sentences, 1):
            transformed_rows.append({
                'ID': post_id,
                'Sentence ID': i,
                'Context': caption,
                'Statement': sentence
            })
    
    # Create transformed dataframe
    df_transformed = pd.DataFrame(transformed_rows)
    
    # Save to CSV
    df_transformed.to_csv(output_file, index=False)
    
    print(f"Transformation complete!")
    print(f"Original posts: {len(df_raw)}")
    print(f"Total sentences: {len(df_transformed)}")
    
    return df_transformed

# Main execution
if __name__ == "__main__":
    # Transform the data
    result_df = transform_instagram_data('ig_posts_raw_mini.csv', 'ig_posts_processed.csv')
    
    # Display sample results
    print("\nSample transformed data:")
    print(result_df.head())
