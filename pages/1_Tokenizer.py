import streamlit as st
import pandas as pd
import nltk
import re
import io

# Download NLTK data if not already present
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    return True

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
            clean_sentences.append(sentence)
    
    return clean_sentences

def transform_instagram_data(df_raw):
    """Transform raw Instagram data to sentence-tokenized format"""
    
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
    
    return df_transformed

def main():
    st.set_page_config(
        page_title="Instagram Sentence Tokenizer",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 Instagram Sentence Tokenizer")
    st.markdown("Transform Instagram post data for text analysis using sentence tokenization")
    
    # Download NLTK data
    with st.spinner("Initializing NLTK resources..."):
        download_nltk_data()
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload your CSV file with Instagram posts
        2. Ensure it has columns: `shortcode` and `caption`
        3. Click 'Process Data' to tokenize sentences
        4. Download the transformed data
        """)
        
        st.header("📊 Expected Format")
        st.markdown("""
        **Input columns:**
        - `shortcode`: Post ID
        - `caption`: Post text
        
        **Output columns:**
        - `ID`: Post identifier
        - `Sentence ID`: Sequential number
        - `Context`: Original caption
        - `Statement`: Individual sentence
        """)
    
    # File upload section
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your ig_posts_raw_mini.csv file"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df_raw = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_columns = ['shortcode', 'caption']
            if not all(col in df_raw.columns for col in required_columns):
                st.error(f"❌ Missing required columns. Expected: {required_columns}")
                st.stop()
            
            # Display raw data
            st.header("📋 Raw Data Preview")
            st.dataframe(df_raw.head(), use_container_width=True)
            st.info(f"Total posts: {len(df_raw)}")
            
            # Process button
            if st.button("🔄 Process Data", type="primary"):
                with st.spinner("Processing sentences..."):
                    # Transform the data
                    df_transformed = transform_instagram_data(df_raw)
                
                # Display results
                st.header("✅ Transformed Data")
                st.dataframe(df_transformed, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Posts", len(df_raw))
                with col2:
                    st.metric("Total Sentences", len(df_transformed))
                with col3:
                    avg_sentences = len(df_transformed) / len(df_raw) if len(df_raw) > 0 else 0
                    st.metric("Avg Sentences/Post", f"{avg_sentences:.1f}")
                
                # Sample transformation display
                st.header("🔍 Sample Transformation")
                if not df_transformed.empty:
                    sample_id = df_transformed['ID'].iloc[0]
                    sample_data = df_transformed[df_transformed['ID'] == sample_id]
                    
                    st.subheader(f"Post ID: {sample_id}")
                    st.write(f"**Original Caption:** {sample_data['Context'].iloc[0]}")
                    st.write("**Tokenized Sentences:**")
                    for _, row in sample_data.iterrows():
                        st.write(f"{row['Sentence ID']}. {row['Statement']}")
                
                # Download section
                st.header("💾 Download Results")
                
                # Convert dataframe to CSV
                csv_buffer = io.StringIO()
                df_transformed.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Transformed Data",
                    data=csv_data,
                    file_name="ig_posts_processed.csv",
                    mime="text/csv",
                    help="Download the sentence-tokenized data"
                )
                
                # Display transformation summary
                st.success(f"✅ Successfully processed {len(df_raw)} posts into {len(df_transformed)} sentences!")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check that your CSV file has the correct format and columns.")
    
    else:
        # Show example when no file is uploaded
        st.header("📄 Example Data Format")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Format (ig_posts_raw_mini.csv)")
            example_input = pd.DataFrame({
                'shortcode': ['Cc8dyfCLwTX', 'CTFccmHFfZQ'],
                'caption': [
                    'No other way to walk into warmer weather✨',
                    'Allow me to introduce myself. I make custom clothing for business professionals!'
                ]
            })
            st.dataframe(example_input, use_container_width=True)
        
        with col2:
            st.subheader("Output Format (ig_posts_transformed_mini.csv)")
            example_output = pd.DataFrame({
                'ID': ['Cc8dyfCLwTX', 'CTFccmHFfZQ', 'CTFccmHFfZQ'],
                'Sentence ID': [1, 1, 2],
                'Context': [
                    'No other way to walk into warmer weather✨',
                    'Allow me to introduce myself. I make custom clothing for business professionals!',
                    'Allow me to introduce myself. I make custom clothing for business professionals!'
                ],
                'Statement': [
                    'No other way to walk into warmer weather.',
                    'Allow me to introduce myself.',
                    'I make custom clothing for business professionals!'
                ]
            })
            st.dataframe(example_output, use_container_width=True)

if __name__ == "__main__":
    main()
