import pandas as pd
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

def load_csv_safe(csv_file: str) -> pd.DataFrame:
    """Load CSV with encoding fallback."""
    encodings = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            print(f"✓ Loaded with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding}: {e}")
            continue
    
    raise ValueError("Could not read CSV with any encoding")

def classify_text(text: str, dictionaries: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Classify text against dictionaries."""
    if pd.isna(text) or text == '':
        return {dict_name: [] for dict_name in dictionaries.keys()}
    
    text_lower = str(text).lower()
    results = {}
    
    for dict_name, terms in dictionaries.items():
        matches = [term for term in terms if term.lower() in text_lower]
        results[dict_name] = matches
    
    return results

def process_data(csv_file: str, text_column: str = 'Statement') -> pd.DataFrame:
    """Process CSV and add classification columns."""
    # Load data
    df = load_csv_safe(csv_file)
    print(f"Data shape: {df.shape}")
    
    # Apply classification
    classifications = df[text_column].apply(lambda x: classify_text(x, dictionaries))
    
    # Add new columns
    for dict_name in dictionaries.keys():
        df[f'{dict_name}_matches'] = classifications.apply(lambda x: x[dict_name])
        df[f'{dict_name}_count'] = df[f'{dict_name}_matches'].apply(len)
        df[f'{dict_name}_binary'] = (df[f'{dict_name}_count'] > 0).astype(int)
    
    return df

# Main execution
if __name__ == "__main__":
    try:
        # Process
        result_df = process_data('sample_data.csv')
        
        # Summary
        print("\n" + "="*40)
        print("CLASSIFICATION SUMMARY")
        print("="*40)
        
        for dict_name in dictionaries.keys():
            count_col = f'{dict_name}_count'
            total = result_df[count_col].sum()
            texts = (result_df[count_col] > 0).sum()
            print(f"{dict_name}: {total} matches in {texts} texts")
        
        # Details
        print("\n" + "-"*40)
        print("DETAILED RESULTS")
        print("-"*40)
        
        for idx, row in result_df.iterrows():
            statement = str(row['Statement'])[:60] + "..." if len(str(row['Statement'])) > 60 else str(row['Statement'])
            
            matches_found = False
            result_text = f"\nID: {row['ID']}\nText: {statement}\n"
            
            for dict_name in dictionaries.keys():
                matches = row[f'{dict_name}_matches']
                if matches:
                    result_text += f"  → {dict_name}: {matches}\n"
                    matches_found = True
            
            if matches_found:
                print(result_text)
        
        # Save
        result_df.to_csv('classified_data.csv', index=False, encoding='utf-8')
        print(f"\n✓ Saved to 'classified_data.csv'")
        print(f"✓ Added {len([col for col in result_df.columns if '_matches' in col or '_count' in col or '_binary' in col])} new columns")
        
    except Exception as e:
        print(f"❌ Error: {e}")
