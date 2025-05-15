import streamlit as st
import requests
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, Polypeptide
from Bio.Data.IUPACData import protein_letters
from io import StringIO
from datetime import datetime

# Initialize session state
if 'uniprot_data' not in st.session_state:
    st.session_state.uniprot_data = {}

# --------------------------
# Helper Functions (defined first)
# --------------------------

def add_uniprot_ids(new_uniprot_id):
    """Add new UniProt IDs to the processing list"""
    ids_to_add = [id.strip().upper() for id in new_uniprot_id.split(',') if id.strip()]
    for uniprot_id in ids_to_add:
        if uniprot_id not in st.session_state.uniprot_data:
            st.session_state.uniprot_data[uniprot_id] = {
                'status': 'pending',
                'error_message': '',
                'pdb_entries': [],
                'selected_pdb': None,
                'chains': {},
                'uniprot_seq': '',
                'alignments': [],
                'mappings': [],
                'chain_descriptions': {}
            }
    st.rerun()

def display_uniprot_card(uniprot_id):
    """Display a card for a single UniProt ID with remove option"""
    data = st.session_state.uniprot_data[uniprot_id]
    status_color = {
        'pending': 'gray',
        'processing': 'orange',
        'complete': 'green',
        'error': 'red'
    }.get(data['status'], 'gray')
    
    container = st.container()
    container.markdown(f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <div style="color: {status_color}; font-size: 0.8em;">
            {uniprot_id} - {data['status'].capitalize()}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if container.button("Remove", key=f"remove_{uniprot_id}"):
        del st.session_state.uniprot_data[uniprot_id]
        st.rerun()

def process_all_uniprot_ids():
    """Process all UniProt IDs in sequence with error handling"""
    for uniprot_id, data in st.session_state.uniprot_data.items():
        if data['status'] in ['pending', 'error']:  # Retry failed ones
            try:
                data['status'] = 'processing'
                data['error_message'] = ''
                process_single_uniprot(uniprot_id)
            except Exception as e:
                data['status'] = 'error'
                data['error_message'] = str(e)
                st.error(f"Failed to process {uniprot_id}: {str(e)}")

def process_single_uniprot(uniprot_id):
    """Process a single UniProt ID with detailed error handling and debugging"""
    data = st.session_state.uniprot_data[uniprot_id]
    
    try:
        st.write(f"⏳ Starting processing for {uniprot_id}...")
        
        # 1. Fetch PDB entries with debugging
        if not data['pdb_entries']:
            st.write(f"🔍 Fetching PDB entries for {uniprot_id}...")
            try:
                data['pdb_entries'] = _fetch_pdb_entries_task(uniprot_id)
                st.write(f"✅ Found {len(data['pdb_entries'])} PDB entries")
                if not data['pdb_entries']:
                    raise ValueError(f"No PDB entries found for {uniprot_id}")
            except Exception as e:
                st.error(f"❌ PDB fetch failed: {str(e)}")
                raise
        
        # 2. Fetch UniProt sequence with debugging
        if not data['uniprot_seq']:
            st.write(f"🔍 Fetching UniProt sequence for {uniprot_id}...")
            try:
                data['uniprot_seq'] = _fetch_uniprot_sequence_task(uniprot_id)
                st.write(f"✅ Got sequence (length: {len(data['uniprot_seq'])})")
                if not data['uniprot_seq']:
                    raise ValueError(f"Empty sequence returned for {uniprot_id}")
            except Exception as e:
                st.error(f"❌ Sequence fetch failed: {str(e)}")
                raise
        
        # 3. Process each PDB entry with debugging
        for pdb_entry in data['pdb_entries']:
            pdb_id = pdb_entry['id']
            st.write(f"🔬 Processing PDB {pdb_id}...")
            
            try:
                if pdb_id not in data['chains']:
                    st.write(f"🔗 Fetching chains for {pdb_id}...")
                    chains_data, chain_descriptions = _fetch_chains_task(pdb_id)
                    
                    if not chains_data:
                        raise ValueError(f"No valid chains found in {pdb_id}")
                    
                    data['chains'][pdb_id] = chains_data
                    data['chain_descriptions'] = chain_descriptions
                    st.write(f"✅ Found {len(chains_data)} chains")
                
                # 4. Perform alignments for each chain
                for chain_id, (pdb_seq, pdb_res_ids) in data['chains'][pdb_id].items():
                    st.write(f"🔄 Aligning chain {chain_id} (length: {len(pdb_seq)})...")
                    try:
                        alignment, mapping = _perform_alignment(pdb_seq, pdb_res_ids, data['uniprot_seq'])
                        data['alignments'].append(f"=== {pdb_id} Chain {chain_id} ===\n{alignment}\n")
                        data['mappings'].append(f"=== {pdb_id} Chain {chain_id} ===\n{mapping}\n")
                        st.write(f"✔️ Alignment successful for chain {chain_id}")
                    except Exception as e:
                        error_msg = f"Alignment failed for {pdb_id} chain {chain_id}: {str(e)}"
                        st.error(f"❌ {error_msg}")
                        data['alignments'].append(error_msg)
                        data['mappings'].append(error_msg)
                        raise Exception(error_msg)
                        
            except Exception as e:
                st.error(f"❌ PDB processing failed: {str(e)}")
                raise Exception(f"Failed to process PDB {pdb_id}: {str(e)}")
        
        data['status'] = 'complete'
        st.success(f"🎉 Processing complete for {uniprot_id}")
    
    except Exception as e:
        data['status'] = 'error'
        data['error_message'] = str(e)
        st.error(f"🔥 Processing failed for {uniprot_id}: {str(e)}")
        raise  # Re-raise to see full traceback in logs
def display_results(uniprot_id):
    """Display results for a single UniProt ID"""
    data = st.session_state.uniprot_data[uniprot_id]
    
    st.divider()
    st.subheader(f"Results for UniProt ID: {uniprot_id}")
    
    # Show status and error message if any
    if data['status'] == 'error':
        st.error(f"Processing failed: {data.get('error_message', 'Unknown error')}")
    elif data['status'] == 'complete':
        st.success("Processing complete")
    elif data['status'] == 'processing':
        st.warning("Processing in progress...")
    
    # PDB entries section
    if data['pdb_entries']:
        st.info(f"Found {len(data['pdb_entries'])} PDB entries")
        
        # Let user select a PDB to focus on
        pdb_options = [f"{entry['id']} (Resolution: {entry['resolution']}, Length: {entry['length']})" 
                      for entry in data['pdb_entries']]
        selected_idx = st.selectbox(
            f"Select PDB entry for {uniprot_id} to view details:",
            range(len(pdb_options)),
            format_func=lambda x: pdb_options[x],
            key=f"pdb_select_{uniprot_id}"
        )
        
        if selected_idx is not None:
            selected_pdb = data['pdb_entries'][selected_idx]
            data['selected_pdb'] = selected_pdb['id']
            
            # Show chains for selected PDB
            if data['selected_pdb'] in data['chains']:
                st.subheader(f"Chains in PDB {data['selected_pdb']}")
                for chain_id, chain_data in data['chains'][data['selected_pdb']].items():
                    seq_len = len(chain_data[0])
                    st.write(f"**Chain {chain_id}** ({seq_len} aa): {data['chain_descriptions'].get(chain_id, 'No description')}")
    
    # Results section
    if data['alignments'] or data['mappings']:
        st.subheader("Alignment Results")
        
        tab1, tab2 = st.tabs(["Alignments", "Residue Mappings"])
        
        with tab1:
            if data['alignments']:
                st.text("\n".join(data['alignments']))
            else:
                st.warning("No alignment results available")
            
            if data['alignments']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label=f"Download Alignments for {uniprot_id}",
                    data="\n".join(data['alignments']),
                    file_name=f"alignments_{uniprot_id}_{timestamp}.txt",
                    mime="text/plain",
                    key=f"dl_align_{uniprot_id}"
                )
        
        with tab2:
            if data['mappings']:
                st.text("\n".join(data['mappings']))
            else:
                st.warning("No residue mappings available")
            
            if data['mappings']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label=f"Download Mappings for {uniprot_id}",
                    data="\n".join(data['mappings']),
                    file_name=f"mappings_{uniprot_id}_{timestamp}.txt",
                    mime="text/plain",
                    key=f"dl_map_{uniprot_id}"
                )

# --------------------------
# Main Application
# --------------------------

def main():
    st.set_page_config(page_title="UniProt-PDB Residue mapping", layout="wide")
    st.title("UniProt-PDB Residue mapping")
    
    # Input Section
    with st.expander("Input Parameters", expanded=True):
        new_uniprot_id = st.text_input("Enter UniProt ID(s), comma separated (e.g., P12345, Q98765)", 
                                     key="uniprot_input")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add UniProt ID(s)") and new_uniprot_id.strip():
                add_uniprot_ids(new_uniprot_id)
        with col2:
            if st.button("Clear All IDs"):
                st.session_state.uniprot_data = {}
                st.rerun()
    
    # Display current UniProt IDs
    if st.session_state.uniprot_data:
        st.subheader("Current UniProt IDs to Process:")
        cols = st.columns(4)
        for i, uniprot_id in enumerate(st.session_state.uniprot_data.keys()):
            with cols[i % 4]:
                display_uniprot_card(uniprot_id)
    
    # Process button with better error handling
    if st.session_state.uniprot_data:
        if st.button("Process All UniProt IDs"):
            with st.spinner("Processing all IDs..."):
                process_all_uniprot_ids()
                st.rerun()

    # Display results for each UniProt ID
    for uniprot_id in st.session_state.uniprot_data.keys():
        display_results(uniprot_id)

# --------------------------
# Include all original helper functions:
# _fetch_pdb_entries_task
# _get_pdb_metadata
# _fetch_chains_task
# _fetch_uniprot_sequence_task
# _perform_alignment
# --------------------------

if __name__ == "__main__":
    main()
