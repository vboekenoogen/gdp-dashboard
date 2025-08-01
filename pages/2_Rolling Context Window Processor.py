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
    """Clean and prepare text for processing"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize_sentences(text):
    """Split text into individual sentences"""
    if not text:
        return []
    sentences = nltk.sent_tokenize(text)
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            clean_sentences.append(sentence)
    return clean_sentences

def create_rolling_context(statements, window_size, include_current=True):
    """Create rolling window context for each statement"""
    contexts = []
    for i, statement in enumerate(statements):
        if include_current:
            # Include current statement in context
            start_idx = max(0, i - window_size + 1)
            context_window = statements[start_idx:i+1]
        else:
            # Only previous statements (not including current)
            start_idx = max(0, i - window_size)
            context_window = statements[start_idx:i] if i > 0 else []
        
        context = ' '.join(context_window) if context_window else statement
        contexts.append(context)
    
    return contexts

def process_conversation_data(df, conversation_id_col, turn_col, message_col, speaker_col, 
                            statement_level, context_type, window_size, include_current, speaker_filter):
    """Process conversation data with rolling context"""
    
    processed_rows = []
    
    # Group by conversation ID
    for conv_id, conv_group in df.groupby(conversation_id_col):
        
        # Sort by turn order
        conv_group = conv_group.sort_values(turn_col)
        
        # Apply speaker filter if specified
        if speaker_filter:
            conv_group = conv_group[conv_group[speaker_col].isin(speaker_filter)]
            if conv_group.empty:
                continue
        
        if statement_level == "Sentence Level":
            # Process each turn and break into sentences
            all_sentences = []
            sentence_metadata = []
            
            for _, row in conv_group.iterrows():
                turn_text = preprocess_text(row[message_col])
                sentences = tokenize_sentences(turn_text)
                
                for sentence in sentences:
                    all_sentences.append(sentence)
                    sentence_metadata.append({
                        'turn': row[turn_col],
                        'speaker': row[speaker_col],
                        'original_message': turn_text
                    })
            
            # Create contexts based on type
            if context_type == "Rolling Window":
                contexts = create_rolling_context(all_sentences, window_size, include_current)
            else:  # Whole Conversation
                full_context = ' '.join(all_sentences)
                contexts = [full_context] * len(all_sentences)
            
            # Create rows for each sentence
            for i, (sentence, context, metadata) in enumerate(zip(all_sentences, contexts, sentence_metadata)):
                processed_rows.append({
                    'Conversation ID': conv_id,
                    'Turn': metadata['turn'],
                    'Sentence ID': i + 1,
                    'Context': context,
                    'Statement': sentence,
                    'Speaker': metadata['speaker'],
                    'Original Message': metadata['original_message'],
                    'Statement Level': 'Sentence',
                    'Context Type': context_type,
                    'Window Size': window_size if context_type == "Rolling Window" else 'N/A'
                })
        
        elif statement_level == "Turn Level":
            # Use entire turns as statements
            all_turns = []
            turn_metadata = []
            
            for _, row in conv_group.iterrows():
                turn_text = preprocess_text(row[message_col])
                all_turns.append(turn_text)
                turn_metadata.append({
                    'turn': row[turn_col],
                    'speaker': row[speaker_col]
                })
            
            # Create contexts based on type
            if context_type == "Rolling Window":
                contexts = create_rolling_context(all_turns, window_size, include_current)
            else:  # Whole Conversation
                full_context = ' '.join(all_turns)
                contexts = [full_context] * len(all_turns)
            
            # Create rows for each turn
            for i, (turn, context, metadata) in enumerate(zip(all_turns, contexts, turn_metadata)):
                processed_rows.append({
                    'Conversation ID': conv_id,
                    'Turn': metadata['turn'],
                    'Sentence ID': i + 1,
                    'Context': context,
                    'Statement': turn,
                    'Speaker': metadata['speaker'],
                    'Original Message': turn,
                    'Statement Level': 'Turn',
                    'Context Type': context_type,
                    'Window Size': window_size if context_type == "Rolling Window" else 'N/A'
                })
        
        else:  # Conversation Level
            # Use entire conversation as single statement
            all_messages = []
            all_speakers = set()
            
            for _, row in conv_group.iterrows():
                all_messages.append(preprocess_text(row[message_col]))
                all_speakers.add(row[speaker_col])
            
            full_statement = ' '.join(all_messages)
            
            processed_rows.append({
                'Conversation ID': conv_id,
                'Turn': 'All',
                'Sentence ID': 1,
                'Context': full_statement,
                'Statement': full_statement,
                'Speaker': ', '.join(all_speakers),
                'Original Message': full_statement,
                'Statement Level': 'Conversation',
                'Context Type': context_type,
                'Window Size': 'N/A'
            })
    
    return pd.DataFrame(processed_rows)

def main():
    st.set_page_config(
        page_title="Rolling Context Window Processor",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Rolling Context Window Processor")
    st.markdown("Upload your conversation dataset and process it with rolling context windows")
    
    # Download NLTK data
    with st.spinner("Initializing NLTK resources..."):
        download_nltk_data()
    
    # Instructions
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
        1. **Upload CSV file** - Your conversation dataset
        2. **Map columns** - Select conversation ID, turn, message, and speaker columns
        3. **Choose statement level** - Sentence, Turn, or Conversation level processing
        4. **Configure context** - Rolling window or whole conversation context
        5. **Set window size** - Number of previous statements to include in context
        6. **Filter speakers** - Optional speaker filtering
        7. **Process data** - Generate rolling context windows
        8. **Download results** - Get processed data with context windows
        """)
    
    # Expected format
    with st.expander("📊 Expected Input Format"):
        st.markdown("""
        Your CSV should contain columns for:
        * **Conversation ID** - Unique identifier for each conversation
        * **Turn** - Sequential turn number or timestamp
        * **Message** - The actual message/utterance text
        * **Speaker** - Who sent the message (e.g., customer, agent, user1, user2)
        """)
        
        # Example data
        example_df = pd.DataFrame({
            'conv_id': ['conv_001', 'conv_001', 'conv_001', 'conv_001'],
            'turn_num': [1, 2, 3, 4],
            'message_text': [
                'Hi, I need help with my order',
                'Sure! What seems to be the issue?',
                'My order hasnt arrived yet. I ordered it 5 days ago.',
                'Let me check that for you. Can you provide your order number?'
            ],
            'speaker_role': ['customer', 'agent', 'customer', 'agent']
        })
        st.dataframe(example_df, use_container_width=True)
    
    # File upload
    st.header("📁 Upload Conversation Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your conversation dataset",
        key="conversation_file_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df_raw = pd.read_csv(uploaded_file)
            
            # Display data preview
            st.header("📋 Data Preview")
            st.dataframe(df_raw.head(), use_container_width=True)
            st.info(f"Total records: {len(df_raw)} | Columns: {', '.join(df_raw.columns)}")
            
            # Auto-detect column mappings
            def auto_detect_columns(columns):
                """Auto-detect column mappings based on common naming patterns"""
                mappings = {}
                
                # Conversation ID patterns
                id_patterns = ['id', 'conv', 'conversation', 'chat', 'session']
                mappings['conversation_id'] = next((col for col in columns 
                                                  if any(pattern in col.lower() for pattern in id_patterns)), 
                                                 columns[0])
                
                # Turn patterns
                turn_patterns = ['turn', 'step', 'sequence', 'order', 'number', 'time']
                mappings['turn'] = next((col for col in columns 
                                       if any(pattern in col.lower() for pattern in turn_patterns)), 
                                      columns[1] if len(columns) > 1 else columns[0])
                
                # Message patterns
                message_patterns = ['message', 'text', 'content', 'utterance', 'statement', 'msg']
                mappings['message'] = next((col for col in columns 
                                          if any(pattern in col.lower() for pattern in message_patterns)), 
                                         columns[-2] if len(columns) >= 2 else columns[0])
                
                # Speaker patterns
                speaker_patterns = ['speaker', 'user', 'role', 'agent', 'customer', 'who', 'from']
                mappings['speaker'] = next((col for col in columns 
                                          if any(pattern in col.lower() for pattern in speaker_patterns)), 
                                         columns[-1])
                
                return mappings
            
            # Get auto-detected mappings
            auto_mappings = auto_detect_columns(df_raw.columns.tolist())
            
            # Column mapping with auto-detection
            st.header("🗺️ Column Mapping")
            st.markdown("Map your columns to the required fields (auto-detected defaults shown):")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Find index of auto-detected column for default selection
                conv_id_default = list(df_raw.columns).index(auto_mappings['conversation_id'])
                conversation_id_col = st.selectbox(
                    "Conversation ID Column",
                    options=df_raw.columns,
                    index=conv_id_default,
                    help="Column that identifies each conversation",
                    key="conv_id_selectbox"
                )
                
                turn_default = list(df_raw.columns).index(auto_mappings['turn'])
                turn_col = st.selectbox(
                    "Turn Column",
                    options=df_raw.columns,
                    index=turn_default,
                    help="Column with turn numbers or timestamps",
                    key="turn_selectbox"
                )
            
            with col2:
                message_default = list(df_raw.columns).index(auto_mappings['message'])
                message_col = st.selectbox(
                    "Message Column",
                    options=df_raw.columns,
                    index=message_default,
                    help="Column containing the message text",
                    key="message_selectbox"
                )
                
                speaker_default = list(df_raw.columns).index(auto_mappings['speaker'])
                speaker_col = st.selectbox(
                    "Speaker Column", 
                    options=df_raw.columns,
                    index=speaker_default,
                    help="Column identifying who sent the message",
                    key="speaker_selectbox"
                )
            
            # Show mapping summary
            mapping_summary = st.container()
            with mapping_summary:
                st.info(f"**Mapping:** Conversation ID → `{conversation_id_col}` | Turn → `{turn_col}` | Message → `{message_col}` | Speaker → `{speaker_col}`")
            
            # Processing configuration
            st.header("⚙️ Processing Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                statement_level = st.selectbox(
                    "Statement Level",
                    ["Sentence Level", "Turn Level", "Conversation Level"],
                    help="How to split the text into statements",
                    key="statement_level_selectbox"
                )
            
            with col2:
                context_type = st.selectbox(
                    "Context Type",
                    ["Rolling Window", "Whole Conversation"],
                    help="Type of context to provide for each statement",
                    key="context_type_selectbox"
                )
            
            with col3:
                if context_type == "Rolling Window":
                    window_size = st.number_input(
                        "Window Size",
                        min_value=1,
                        max_value=20,
                        value=3,
                        help="Number of previous statements to include",
                        key="window_size_input"
                    )
                else:
                    window_size = 0
                    st.metric("Window Size", "N/A")
            
            # Additional options
            col1, col2 = st.columns(2)
            
            with col1:
                if context_type == "Rolling Window":
                    include_current = st.checkbox(
                        "Include current statement in context",
                        value=True,
                        help="Whether to include the current statement in its own context"
                    )
                else:
                    include_current = True
            
            with col2:
                # Get unique speakers for filter
                unique_speakers = df_raw[speaker_col].unique().tolist()
                speaker_filter = st.multiselect(
                    "Filter by Speaker (optional)",
                    options=unique_speakers,
                    help="Only process messages from selected speakers"
                )
            
            # Preview configuration
            st.subheader("🔍 Configuration Summary")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.info(f"**Statement Level:** {statement_level}")
                st.info(f"**Context Type:** {context_type}")
                if context_type == "Rolling Window":
                    st.info(f"**Window Size:** {window_size}")
                    st.info(f"**Include Current:** {include_current}")
            
            with config_col2:
                st.info(f"**Conversations:** {df_raw[conversation_id_col].nunique()}")
                st.info(f"**Total Turns:** {len(df_raw)}")
                if speaker_filter:
                    st.info(f"**Speaker Filter:** {', '.join(speaker_filter)}")
                else:
                    st.info(f"**All Speakers:** {', '.join(unique_speakers)}")
            
            # Process button
            if st.button("🚀 Process Data", type="primary"):
                with st.spinner("Processing conversation data with rolling context..."):
                    # Process the data
                    df_processed = process_conversation_data(
                        df_raw,
                        conversation_id_col,
                        turn_col, 
                        message_col,
                        speaker_col,
                        statement_level,
                        context_type,
                        window_size,
                        include_current,
                        speaker_filter
                    )
                
                if df_processed.empty:
                    st.error("❌ No data was processed. Check your speaker filter settings.")
                    st.stop()
                
                # Display results
                st.header("✅ Processed Data")
                st.dataframe(df_processed, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Input Records", len(df_raw))
                with col2:
                    st.metric("Output Records", len(df_processed))
                with col3:
                    st.metric("Conversations", df_processed['Conversation ID'].nunique())
                with col4:
                    expansion_ratio = len(df_processed) / len(df_raw) if len(df_raw) > 0 else 0
                    st.metric("Expansion Ratio", f"{expansion_ratio:.2f}x")
                
                # Sample transformation
                st.header("🔍 Sample Context Windows")
                if not df_processed.empty:
                    sample_conv = df_processed['Conversation ID'].iloc[0]
                    sample_data = df_processed[df_processed['Conversation ID'] == sample_conv].head(3)
                    
                    for _, row in sample_data.iterrows():
                        with st.expander(f"Turn {row['Turn']} - {row['Speaker']}"):
                            st.write(f"**Statement:** {row['Statement']}")
                            st.write(f"**Context:** {row['Context']}")
                            if row['Context'] != row['Statement']:
                                st.caption(f"Context includes {len(row['Context'].split()) - len(row['Statement'].split())} additional words from previous turns")
                
                # Download section
                st.header("💾 Download Results")
                
                # Convert to CSV
                csv_buffer = io.StringIO()
                df_processed.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                filename = f"rolling_context_{statement_level.lower().replace(' ', '_')}_{context_type.lower().replace(' ', '_')}.csv"
                
                st.download_button(
                    label="📥 Download Processed Data",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help="Download conversation data with rolling context windows"
                )
                
                st.success(f"✅ Successfully processed {len(df_raw)} records into {len(df_processed)} statements with context windows!")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check your CSV file format and column selections.")
    
    else:
        # Show example output
        st.header("📄 Example Output")
        
        st.subheader("Sample: Rolling Window (Size=2, Sentence Level)")
        example_output = pd.DataFrame({
            'Conversation ID': ['conv_001', 'conv_001', 'conv_001', 'conv_001'],
            'Turn': [1, 2, 2, 3],
            'Sentence ID': [1, 2, 3, 4],
            'Context': [
                'Hi, I need help with my order',
                'Hi, I need help with my order Sure!',
                'I need help with my order Sure! What seems to be the issue?',
                'Sure! What seems to be the issue? My order hasnt arrived yet.'
            ],
            'Statement': [
                'Hi, I need help with my order',
                'Sure!',
                'What seems to be the issue?', 
                'My order hasnt arrived yet.'
            ],
            'Speaker': ['customer', 'agent', 'agent', 'customer']
        })
        st.dataframe(example_output, use_container_width=True)
        st.caption("Notice how each statement's context includes previous statements in the rolling window")

if __name__ == "__main__":
    main()
