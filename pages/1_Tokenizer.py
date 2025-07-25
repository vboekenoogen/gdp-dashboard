import streamlit as st
import pandas as pd
import re
from typing import List
import io

class InstagramSentenceTokenizer:
    """Minimalist sentence tokenizer for Instagram posts"""
    
    def __init__(self):
        self.sentence_endings = r'[.!?]+(?:\s|$)'
        self.hashtag_pattern = r'#\w+'
    
    def tokenize_sentences(self, text: str, separate_hashtags: bool = True) -> List[str]:
        """Split text into sentences with option to separate hashtags"""
        if not text or pd.isna(text):
            return []
        
        # Clean and normalize text
        text = str(text).strip()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        sentences = []
        
        if separate_hashtags:
            hashtags = re.findall(self.hashtag_pattern, text)
            text_without_hashtags = re.sub(self.hashtag_pattern, '', text).strip()
            
            if text_without_hashtags:
                main_sentences = self._split_sentences(text_without_hashtags)
                sentences.extend(main_sentences)
            
            if hashtags:
                hashtag_sentence = ' '.join(hashtags)
                sentences.append(hashtag_sentence)
        else:
            sentences = self._split_sentences(text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules"""
        parts = re.split(self.sentence_endings, text)
        
        sentences = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                if i < len(parts) - 1:
                    if not part.endswith(('.', '!', '?')):
                        part += '.'
                sentences.append(part)
        
        return sentences

def process_dataframe(df: pd.DataFrame, id_column: str, context_column: str, separate_hashtags: bool) -> pd.DataFrame:
    """Process dataframe and return tokenized results"""
    tokenizer = InstagramSentenceTokenizer()
    results = []
    
    for _, row in df.iterrows():
        post_id = row[id_column]
        context = row[context_column]
        
        sentences = tokenizer.tokenize_sentences(context, separate_hashtags)
        
        for sentence_id, sentence in enumerate(sentences, 1):
            results.append({
                'ID': post_id,
                'Sentence ID': sentence_id,
                'Context': context,
                'Statement': sentence
            })
    
    return pd.DataFrame(results)

def main():
    st.set_page_config(
        page_title="Instagram Sentence Tokenizer",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 Instagram Sentence Tokenizer")
    st.markdown("Transform your Instagram data into tokenized sentences for text analysis")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file containing Instagram post data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(df)} rows and {len(df.columns)} columns")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
            # Column selection
            st.sidebar.subheader("Column Selection")
            
            available_columns = list(df.columns)
            
            id_column = st.sidebar.selectbox(
                "Select ID Column",
                options=available_columns,
                index=0 if "shortcode" not in available_columns else available_columns.index("shortcode"),
                help="Column to use as unique identifier"
            )
            
            context_column = st.sidebar.selectbox(
                "Select Context Column",
                options=available_columns,
                index=1 if "caption" not in available_columns else available_columns.index("caption"),
                help="Column containing text to tokenize"
            )
            
            # Processing options
            st.sidebar.subheader("Processing Options")
            separate_hashtags = st.sidebar.checkbox(
                "Separate hashtags as individual sentences",
                value=True,
                help="If checked, hashtags will be extracted as separate sentences"
            )
            
            # Sample text processing demo
            st.sidebar.subheader("🔍 Live Preview")
            sample_text = st.sidebar.text_area(
                "Test your settings:",
                value="Hello world! Check this out. #awesome #cool",
                height=100
            )
            
            if sample_text:
                tokenizer = InstagramSentenceTokenizer()
                sample_sentences = tokenizer.tokenize_sentences(sample_text, separate_hashtags)
                
                st.sidebar.write("**Tokenized sentences:**")
                for i, sentence in enumerate(sample_sentences, 1):
                    st.sidebar.write(f"{i}. {sentence}")
            
            # Process button
            if st.button("🚀 Process Data", type="primary"):
                with st.spinner("Processing your data..."):
                    try:
                        result_df = process_dataframe(df, id_column, context_column, separate_hashtags)
                        
                        st.success(f"✅ Processing complete! Generated {len(result_df)} tokenized sentences from {result_df['ID'].nunique()} unique posts")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Sentences", len(result_df))
                        
                        with col2:
                            st.metric("Unique Posts", result_df['ID'].nunique())
                        
                        # Show results
                        st.subheader("📋 Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Processed Data",
                            data=csv_data,
                            file_name="ig_posts_processed.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
                        # Show sample transformations
                        st.subheader("🔍 Sample Transformations")
                        
                        for post_id in result_df['ID'].unique()[:3]:  # Show first 3 posts
                            post_data = result_df[result_df['ID'] == post_id]
                            original_context = post_data.iloc[0]['Context']
                            
                            with st.expander(f"Post ID: {post_id}"):
                                st.write("**Original Context:**")
                                st.write(f"_{original_context}_")
                                
                                st.write("**Extracted Sentences:**")
                                for _, row in post_data.iterrows():
                                    st.write(f"{row['Sentence ID']}. {row['Statement']}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing data: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started")
        
        st.subheader("📖 Instructions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Input Format:**
            - Upload a CSV file with your Instagram data
            - Should contain at least 2 columns:
              - ID column (e.g., 'shortcode')
              - Text column (e.g., 'caption')
            
            **Example Input:**
            ```
            shortcode,caption
            ABC123,"Hello world! #awesome"
            DEF456,"Great day today."
            ```
            """)
        
        with col2:
            st.markdown("""
            **Output Format:**
            - **ID**: Your selected ID column
            - **Sentence ID**: Sequential numbering (1, 2, 3...)
            - **Context**: Original text
            - **Statement**: Individual sentences
            
            **Example Output:**
            ```
            ID,Sentence ID,Context,Statement
            ABC123,1,"Hello world! #awesome","Hello world!"
            ABC123,2,"Hello world! #awesome","#awesome"
            ```
            """)
        
        # Feature highlights
        st.subheader("✨ Features")
        
        features = [
            "🎯 **Configurable Columns**: Choose any columns as ID and Context",
            "🏷️ **Hashtag Handling**: Option to separate hashtags as individual sentences", 
            "📊 **Live Preview**: Test your settings before processing",
            "📥 **Easy Download**: Get your processed data as CSV",
            "🔍 **Data Validation**: Built-in error checking and validation"
        ]
        
        for feature in features:
            st.markdown(feature)

if __name__ == "__main__":
    main()
    
