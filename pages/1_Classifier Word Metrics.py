# Install chardet if needed: !pip install chardet

import pandas as pd
import chardet
from typing import Dict, Set, List

# Define dictionaries
dictionaries = {
    'urgency_marketing': {
        'limited', 'limited time', 'limited run', 'limited edition', 'order now',
        'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
        'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
        'expires soon', 'final hours', 'almost gone'
    },
    'exclusive_marketing': {
        'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
        'members only', 'vip', 'special access', 'invitation only',
        'premium', 'privileged', 'limited access', 'select customers',
        'insider', 'private sale', 'early access'
    }
}

def detect_encoding(file_path: str) -> str:
    """Detect file encoding."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
        result = chardet.detect(raw_data)
        return result['encoding']
    except:
        return 'utf-8'  # fallback

def classify_text(text: str, dictionaries: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Classify text against multiple dictionaries."""
    if pd.isna(text) or text == '':
        return {dict_name: [] for dict_name in dictionaries.keys()}
    
    text_lower = str(text).lower()
    results = {}
    
    for dict_name, terms in dictionaries.items():
        matches = []
        for term in terms:
            if term.lower() in text_lower:
                matches.append(term)
        results[dict_name] = matches
    
    return results

def load_csv_safe(csv_file: str) -> pd.DataFrame:
    """Load CSV with automatic encoding detection."""
    # Method 1: Try chardet
    try:
        encoding = detect_encoding(csv_file)
        print(f"Detected encoding: {encoding}")
        return pd.read_csv(csv_file, encoding=encoding)
    except:
        pass
    
    # Method 2: Try common encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding")
            return df
        except:
            continue
    
    # Method 3: Force with error handling
    try:
        return pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
    except:
        raise ValueError("Could not read CSV file with any method")

def process_data(csv_file: str, text_column: str = 'Statement') -> pd.DataFrame:
    """Process CSV and add classification columns."""
    # Load data safely
    df = load_csv_safe(csv_file)
    print(f"Loaded data shape: {df.shape}")
    
    # Apply classification
    classifications = df[text_column].apply(lambda x: classify_text(x, dictionaries))
    
    # Add results as new columns
    for dict_name in dictionaries.keys():
        df[f'{dict_name}_matches'] = classifications.apply(lambda x: x[dict_name])
        df[f'{dict_name}_count'] = df[f'{dict_name}_matches'].apply(len)
        df[f'{dict_name}_binary'] = (df[f'{dict_name}_count'] > 0).astype(int)
    
    return df

# Main execution
if __name__ == "__main__":
    try:
        # Process the data
        result_df = process_data('sample_data.csv')
        
        # Display results
        print("\nClassification Results:")
        print("=" * 50)
        
        # Show summary
        for dict_name in dictionaries.keys():
            count_col = f'{dict_name}_count'
            total_matches = result_df[count_col].sum()
            texts_with_matches = (result_df[count_col] > 0).sum()
            print(f"{dict_name}: {total_matches} total matches in {texts_with_matches} texts")
        
        print("\nDetailed Results:")
        print("-" * 30)
        
        # Show each row with matches
        for idx, row in result_df.iterrows():
            statement = str(row['Statement'])
            statement_preview = statement[:50] + "..." if len(statement) > 50 else statement
            print(f"\nID: {row['ID']}")
            print(f"Statement: {statement_preview}")
            
            has_matches = False
            for dict_name in dictionaries.keys():
                matches = row[f'{dict_name}_matches']
                if matches:
                    print(f"  {dict_name}: {matches}")
                    has_matches = True
            
            if not has_matches:
                print("  No matches found")
        
        # Save results
        result_df.to_csv('classified_data.csv', index=False, encoding='utf-8')
        print(f"\nResults saved to 'classified_data.csv'")
        print(f"Final shape: {result_df.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Try installing chardet: !pip install chardet")
