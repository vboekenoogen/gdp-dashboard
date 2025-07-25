import pandas as pd
import re
from typing import List, Tuple

class InstagramSentenceTokenizer:
    """Minimalist sentence tokenizer for Instagram posts"""
    
    def __init__(self):
        # Simple sentence boundary patterns
        self.sentence_endings = r'[.!?]+(?:\s|$)'
        self.hashtag_pattern = r'#\w+'
    
    def tokenize_sentences(self, text: str, separate_hashtags: bool = True) -> List[str]:
        """
        Split text into sentences with option to separate hashtags
        
        Args:
            text: Input text to tokenize
            separate_hashtags: If True, hashtags become separate sentences
            
        Returns:
            List of sentences
        """
        if not text or pd.isna(text):
            return []
        
        # Clean and normalize text
        text = str(text).strip()
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        sentences = []
        
        if separate_hashtags:
            # Extract hashtags first
            hashtags = re.findall(self.hashtag_pattern, text)
            # Remove hashtags from main text
            text_without_hashtags = re.sub(self.hashtag_pattern, '', text).strip()
            
            # Process main text
            if text_without_hashtags:
                main_sentences = self._split_sentences(text_without_hashtags)
                sentences.extend(main_sentences)
            
            # Add hashtags as separate sentence if they exist
            if hashtags:
                hashtag_sentence = ' '.join(hashtags)
                sentences.append(hashtag_sentence)
        else:
            sentences = self._split_sentences(text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules"""
        # Split on sentence endings
        parts = re.split(self.sentence_endings, text)
        
        sentences = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                # Add back punctuation except for last part
                if i < len(parts) - 1:
                    # Find the punctuation that was used to split
                    next_pos = text.find(part) + len(part)
                    if next_pos < len(text):
                        punct_match = re.match(r'[.!?]+', text[next_pos:])
                        if punct_match:
                            punct = punct_match.group()
                            # Use single punctuation mark
                            part += punct[0]
                sentences.append(part)
        
        return sentences

def process_instagram_data(input_file: str, output_file: str, 
                          id_column: str = "shortcode", 
                          context_column: str = "caption",
                          separate_hashtags: bool = True):
    """
    Process Instagram data from raw format to tokenized sentences
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        id_column: Column to use as ID (default: "shortcode")
        context_column: Column to use as Context (default: "caption")
        separate_hashtags: Whether to separate hashtags as individual sentences
    """
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Validate columns
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found. Available: {list(df.columns)}")
    if context_column not in df.columns:
        raise ValueError(f"Context column '{context_column}' not found. Available: {list(df.columns)}")
    
    # Initialize tokenizer
    tokenizer = InstagramSentenceTokenizer()
    
    # Process data
    print("Processing sentences...")
    results = []
    
    for _, row in df.iterrows():
        post_id = row[id_column]
        context = row[context_column]
        
        # Tokenize sentences
        sentences = tokenizer.tokenize_sentences(context, separate_hashtags)
        
        # Create output rows
        for sentence_id, sentence in enumerate(sentences, 1):
            results.append({
                'ID': post_id,
                'Sentence ID': sentence_id,
                'Context': context,
                'Statement': sentence
            })
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Save results
    output_df.to_csv(output_file, index=False)
    print(f"Saved {len(output_df)} tokenized sentences to {output_file}")
    
    return output_df

# Example usage for Colab
def main():
    """Main function for easy execution in Colab"""
    
    # Configuration
    INPUT_FILE = "ig_posts_raw_mini.csv"
    OUTPUT_FILE = "ig_posts_processed.csv"
    
    # Process with default settings
    print("=== Instagram Sentence Tokenizer ===")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    try:
        result_df = process_instagram_data(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            id_column="shortcode",      # Change as needed
            context_column="caption",   # Change as needed
            separate_hashtags=True      # Set False to keep hashtags with text
        )
        
        print("\n=== Processing Complete ===")
        print(f"Total sentences extracted: {len(result_df)}")
        print(f"Unique posts processed: {result_df['ID'].nunique()}")
        print("\nSample output:")
        print(result_df.head(3).to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
