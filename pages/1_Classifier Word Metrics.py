import pandas as pd

urgency_terms = [
    'limited', 'limited time', 'limited run', 'limited edition', 'order now',
    'last chance', 'hurry', 'while supplies last', 'before they are gone',
    'selling out', 'selling fast', 'act now', 'dont wait', 'today only',
    'expires soon', 'final hours', 'almost gone'
]

exclusive_terms = [
    'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
    'members only', 'vip', 'special access', 'invitation only',
    'premium', 'privileged', 'limited access', 'select customers',
    'insider', 'private sale', 'early access'
]

dictionaries = {}
dictionaries['urgency_marketing'] = urgency_terms
dictionaries['exclusive_marketing'] = exclusive_terms

def classify_text(text, dict_list, dict_name):
    if pd.isna(text):
        return []
    
    text_lower = str(text).lower()
    matches = []
    
    for term in dict_list:
        if term.lower() in text_lower:
            matches.append(term)
    
    return matches

def load_csv_safe(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding='latin-1')
        print("Loaded with latin-1")
        return df
    except:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            print("Loaded with utf-8")
            return df
        except:
            df = pd.read_csv(csv_file, encoding='cp1252')
            print("Loaded with cp1252")
            return df

df = load_csv_safe('sample_data.csv')
print("Data shape:", df.shape)

df['urgency_matches'] = df['Statement'].apply(lambda x: classify_text(x, urgency_terms, 'urgency'))
df['urgency_count'] = df['urgency_matches'].apply(len)
df['urgency_binary'] = (df['urgency_count'] > 0).astype(int)

df['exclusive_matches'] = df['Statement'].apply(lambda x: classify_text(x, exclusive_terms, 'exclusive'))
df['exclusive_count'] = df['exclusive_matches'].apply(len)
df['exclusive_binary'] = (df['exclusive_count'] > 0).astype(int)

print("\nResults:")
print("Urgency matches:", df['urgency_count'].sum(), "in", (df['urgency_count'] > 0).sum(), "texts")
print("Exclusive matches:", df['exclusive_count'].sum(), "in", (df['exclusive_count'] > 0).sum(), "texts")

print("\nDetailed results:")
for idx, row in df.iterrows():
    statement = str(row['Statement'])
    if len(statement) > 50:
        statement = statement[:50] + "..."
    
    print("\nID:", row['ID'])
    print("Text:", statement)
    
    if row['urgency_matches']:
        print("  Urgency:", row['urgency_matches'])
    if row['exclusive_matches']:
        print("  Exclusive:", row['exclusive_matches'])

df.to_csv('classified_data.csv', index=False, encoding='utf-8')
print("\nSaved to classified_data.csv")
