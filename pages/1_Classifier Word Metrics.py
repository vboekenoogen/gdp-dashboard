import streamlit as st
import pandas as pd

st.title("Dictionary Classification Tool")

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

def classify_text(text, term_list):
    if pd.isna(text) or text == '':
        return []
    
    text_lower = str(text).lower()
    matches = []
    
    for term in term_list:
        if term.lower() in text_lower:
            matches.append(term)
    
    return matches

def load_csv_safe(csv_file):
    encodings = ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            st.success("File loaded successfully with " + encoding + " encoding")
            return df
        except:
            continue
    
    st.error("Could not read CSV file")
    return None

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Data preview:")
    st.dataframe(df.head())
    
    if 'Statement' in df.columns:
        df['urgency_matches'] = df['Statement'].apply(lambda x: classify_text(x, urgency_terms))
        df['urgency_count'] = df['urgency_matches'].apply(len)
        df['urgency_binary'] = (df['urgency_count'] > 0).astype(int)
        
        df['exclusive_matches'] = df['Statement'].apply(lambda x: classify_text(x, exclusive_terms))
        df['exclusive_count'] = df['exclusive_matches'].apply(len)
        df['exclusive_binary'] = (df['exclusive_count'] > 0).astype(int)
        
        st.subheader("Classification Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            urgency_total = df['urgency_count'].sum()
            urgency_texts = (df['urgency_count'] > 0).sum()
            st.metric("Urgency Marketing", urgency_total, delta=str(urgency_texts) + " texts")
        
        with col2:
            exclusive_total = df['exclusive_count'].sum()
            exclusive_texts = (df['exclusive_count'] > 0).sum()
            st.metric("Exclusive Marketing", exclusive_total, delta=str(exclusive_texts) + " texts")
        
        st.subheader("Detailed Results")
        
        for idx, row in df.iterrows():
            if row['urgency_matches'] or row['exclusive_matches']:
                with st.expander("ID: " + str(row['ID'])):
                    st.write("**Statement:**", str(row['Statement']))
                    
                    if row['urgency_matches']:
                        st.write("🚨 **Urgency terms:**", row['urgency_matches'])
                    
                    if row['exclusive_matches']:
                        st.write("⭐ **Exclusive terms:**", row['exclusive_matches'])
        
        st.subheader("Download Results")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download classified data as CSV",
            data=csv,
            file_name='classified_data.csv',
            mime='text/csv'
        )
        
        st.dataframe(df)
    
    else:
        st.error("No 'Statement' column found in the CSV file")

else:
    st.info("Please upload a CSV file to begin classification")
    
    st.subheader("Sample Dictionary Terms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Urgency Marketing Terms:**")
        for term in urgency_terms[:10]:
            st.write("•", term)
        if len(urgency_terms) > 10:
            st.write("... and", len(urgency_terms) - 10, "more")
    
    with col2:
        st.write("**Exclusive Marketing Terms:**")
        for term in exclusive_terms[:10]:
            st.write("•", term)
        if len(exclusive_terms) > 10:
            st.write("... and", len(exclusive_terms) - 10, "more")
