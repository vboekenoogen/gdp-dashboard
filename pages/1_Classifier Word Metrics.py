import pandas as pd

dictionaries = {
    'urgency_marketing': {
        'limited', 'limited time', 'limited run', 'limited edition', 'order now',
        'last chance', 'hurry', 'while supplies last', 'before they are gone',
        'selling out', 'selling fast', 'act now', 'dont wait', 'today only',
        'expires soon', 'final hours', 'almost gone'
    },
    'exclusive_marketing': {
        'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
        'members only', 'vip', 'special access', 'invitation only',
        'premium', 'privileged', 'limited access', 'select customers',
        'insider', 'private sale', 'early access'
    }
}

def classify_text(text, dictionaries):
    if pd.isna(text):
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

def load_csv_safe(csv_file):
    encodings = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            print(f"Loaded with {encoding}")
            return df
        except:
            continue
    
    raise ValueError("Could not read CSV")

def process_data(csv_file):
    df = load_csv_safe(csv_file)
    
    classifications = df['Statement'].apply(lambda x: classify_text(x, dictionaries))
    
    for dict_name in dictionaries.keys():
        df[dict_name + '_matches'] = classifications.apply(lambda x: x[dict_name])
        df[dict_name + '_count'] = df[dict_name + '_matches'].apply(len)
        df[dict_name + '_binary'] = (df[dict_name + '_count'] > 0).astype(int)
    
    return df

result_df = process_data('sample_data.csv')

for dict_name in dictionaries.keys():
    count_col = dict_name + '_count'
    total = result_df[count_col].sum()
    texts = (result_df[count_col] > 0).sum()
    print(f"{dict_name}: {total} matches in {texts} texts")

for idx, row in result_df.iterrows():
    statement = str(row['Statement'])
    if len(statement) > 50:
        statement = statement[:50] + "..."
    
    print(f"\nID: {row['ID']}")
    print(f"Text: {statement}")
    
    for dict_name in dictionaries.keys():
        matches = row[dict_name + '_matches']
        if matches:
            print(f"  {dict_name}: {matches}")

result_df.to_csv('classified_data.csv', index=False, encoding='utf-8')
print("Saved to classified_data.csv")
