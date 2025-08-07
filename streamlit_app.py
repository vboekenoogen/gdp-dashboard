import streamlit as st
import pandas as pd
import re
from typing import Dict, Set, List
import json
import io

# Set page config
st.set_page_config(
    page_title="Text Classification Tool",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for dictionaries
if 'dictionaries' not in st.session_state:
    st.session_state.dictionaries = {
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

def classify_text(text: str, dictionaries: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Classify text against multiple dictionaries."""
    if pd.isna(text):
        return {dict_name: [] for dict_name in dictionaries.keys()}

    text_lower = text.lower()
    results = {}

    for dict_name, terms in dictionaries.items():
        matches = []
        for term in terms:
            if term.lower() in text_lower:
                matches.append(term)
        results[dict_name] = matches

    return results

def process_data(df: pd.DataFrame, text_column: str, dictionaries: Dict[str, Set[str]]) -> pd.DataFrame:
    """Process DataFrame and add classification columns."""
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Apply classification
    classifications = result_df[text_column].apply(lambda x: classify_text(x, dictionaries))

    # Add results as new columns
    for dict_name in dictionaries.keys():
        result_df[f'{dict_name}_matches'] = classifications.apply(lambda x: x[dict_name])
        result_df[f'{dict_name}_count'] = result_df[f'{dict_name}_matches'].apply(len)
        result_df[f'{dict_name}_binary'] = (result_df[f'{dict_name}_count'] > 0).astype(int)

    return result_df

# Main app
def main():
    st.title("📊 Text Classification Tool")
    st.markdown("Upload your dataset and classify text using customizable dictionaries")

    # Sidebar for dictionary management
    st.sidebar.header("📚 Dictionary Management")

    # Dictionary editor
    st.sidebar.subheader("Edit Dictionaries")

    # Select dictionary to edit
    dict_names = list(st.session_state.dictionaries.keys())
    selected_dict = st.sidebar.selectbox("Select dictionary to edit:", dict_names)

    if selected_dict:
        st.sidebar.write(f"**{selected_dict}** terms:")

        # Display current terms
        current_terms = list(st.session_state.dictionaries[selected_dict])

        # Text area for editing terms
        terms_text = st.sidebar.text_area(
            "Edit terms (one per line):",
            value='\n'.join(current_terms),
            height=200,
            key=f"terms_{selected_dict}"
        )

        # Update button
        if st.sidebar.button(f"Update {selected_dict}"):
            new_terms = set(term.strip() for term in terms_text.split('\n') if term.strip())
            st.session_state.dictionaries[selected_dict] = new_terms
            st.sidebar.success(f"Updated {selected_dict}!")

    # Add new dictionary
    st.sidebar.subheader("Add New Dictionary")
    new_dict_name = st.sidebar.text_input("Dictionary name:")
    new_dict_terms = st.sidebar.text_area("Terms (one per line):", height=100)

    if st.sidebar.button("Add Dictionary"):
        if new_dict_name and new_dict_terms:
            terms_set = set(term.strip() for term in new_dict_terms.split('\n') if term.strip())
            st.session_state.dictionaries[new_dict_name] = terms_set
            st.sidebar.success(f"Added {new_dict_name}!")
        else:
            st.sidebar.error("Please provide both name and terms")

    # Remove dictionary
    if len(dict_names) > 1:
        dict_to_remove = st.sidebar.selectbox("Remove dictionary:", [""] + dict_names)
        if st.sidebar.button("Remove Dictionary") and dict_to_remove:
            del st.session_state.dictionaries[dict_to_remove]
            st.sidebar.success(f"Removed {dict_to_remove}!")
            st.rerun()

    # Export/Import dictionaries
    st.sidebar.subheader("Export/Import")

    # Export dictionaries
    if st.sidebar.button("Export Dictionaries"):
        # Convert sets to lists for JSON serialization
        export_dict = {k: list(v) for k, v in st.session_state.dictionaries.items()}
        json_str = json.dumps(export_dict, indent=2)
        st.sidebar.download_button(
            label="Download JSON",
            data=json_str,
            file_name="dictionaries.json",
            mime="application/json"
        )

    # Import dictionaries
    uploaded_dict_file = st.sidebar.file_uploader("Import dictionaries (JSON):", type=['json'])
    if uploaded_dict_file:
        try:
            import_dict = json.load(uploaded_dict_file)
            # Convert lists back to sets
            for k, v in import_dict.items():
                st.session_state.dictionaries[k] = set(v)
            st.sidebar.success("Dictionaries imported successfully!")
        except Exception as e:
            st.sidebar.error(f"Error importing dictionaries: {e}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📂 Upload Dataset")

        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)

                st.success(f"File uploaded successfully! Shape: {df.shape}")

                # Column selection
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    selected_column = st.selectbox(
                        "Select the text column to classify:",
                        text_columns,
                        index=0 if 'Statement' in text_columns else 0
                    )

                    # Preview data
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(), use_container_width=True)

                    # Process button
                    if st.button("🚀 Classify Text", type="primary"):
                        with st.spinner("Classifying text..."):
                            try:
                                result_df = process_data(df, selected_column, st.session_state.dictionaries)

                                # Store results in session state
                                st.session_state.results = result_df

                                st.success("Classification completed!")

                            except Exception as e:
                                st.error(f"Error during classification: {e}")
                else:
                    st.error("No text columns found in the dataset")

            except Exception as e:
                st.error(f"Error reading file: {e}")

    with col2:
        st.header("📖 Current Dictionaries")

        # Display current dictionaries
        for dict_name, terms in st.session_state.dictionaries.items():
            with st.expander(f"{dict_name} ({len(terms)} terms)"):
                st.write(", ".join(sorted(terms)))

    # Results section
    if 'results' in st.session_state:
        st.header("📊 Classification Results")

        result_df = st.session_state.results

        # Summary statistics
        st.subheader("📈 Summary Statistics")
        summary_cols = st.columns(len(st.session_state.dictionaries))

        for i, dict_name in enumerate(st.session_state.dictionaries.keys()):
            with summary_cols[i]:
                count_col = f'{dict_name}_count'
                total_matches = result_df[count_col].sum()
                texts_with_matches = (result_df[count_col] > 0).sum()
                percentage = (texts_with_matches / len(result_df)) * 100

                st.metric(
                    label=dict_name.replace('_', ' ').title(),
                    value=f"{texts_with_matches} texts",
                    delta=f"{total_matches} total matches"
                )
                st.write(f"{percentage:.1f}% of texts")

        # Detailed results
        st.subheader("🔍 Detailed Results")

        # Filter options
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            show_only_matches = st.checkbox("Show only texts with matches")

        with filter_col2:
            dict_filter = st.selectbox(
                "Filter by dictionary:",
                ["All"] + list(st.session_state.dictionaries.keys())
            )

        # Apply filters
        filtered_df = result_df.copy()

        if show_only_matches:
            # Show only rows with at least one match in any dictionary
            match_mask = False
            for dict_name in st.session_state.dictionaries.keys():
                match_mask |= (filtered_df[f'{dict_name}_count'] > 0)
            filtered_df = filtered_df[match_mask]

        if dict_filter != "All":
            # Show only rows with matches for specific dictionary
            filtered_df = filtered_df[filtered_df[f'{dict_filter}_count'] > 0]

        # Display results
        st.dataframe(filtered_df, use_container_width=True)

        # Download button
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="classified_results.csv",
            mime="text/csv"
        )

        # Show column information
        new_columns = [col for col in result_df.columns if col.endswith(('_matches', '_count', '_binary'))]
        if new_columns:
            st.info(f"**New columns added:** {', '.join(new_columns)}")

if __name__ == "__main__":
    main()
