import streamlit as st
import requests
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, Polypeptide
from Bio.Data.IUPACData import protein_letters
from io import StringIO
from datetime import datetime

# Define helper functions first
def remove_uniprot_id(uniprot_id):
    """Remove a UniProt ID from the processing list"""
    st.session_state.uniprot_ids.remove(uniprot_id)
    del st.session_state.processing_state[uniprot_id]

def process_all_uniprot_ids():
    """Process all UniProt IDs in sequence"""
    for uniprot_id in st.session_state.uniprot_ids:
        data = st.session_state.processing_state[uniprot_id]
        data['status'] = 'processing'
        
        try:
            # 1. Fetch PDB entries
            if not data['pdb_entries']:
                with st.spinner(f"Fetching PDB entries for {uniprot_id}..."):
                    data['pdb_entries'] = _fetch_pdb_entries_task(uniprot_id)
            
            # 2. Fetch UniProt sequence
            if not data['uniprot_seq']:
                with st.spinner(f"Fetching sequence for {uniprot_id}..."):
                    data['uniprot_seq'] = _fetch_uniprot_sequence_task(uniprot_id)
            
            # 3. Process each PDB entry
            for pdb_entry in data['pdb_entries']:
                pdb_id = pdb_entry['id']
                if pdb_id not in data['chains']:
                    with st.spinner(f"Processing {pdb_id} for {uniprot_id}..."):
                        chains_data, chain_descriptions = _fetch_chains_task(pdb_id)
                        data['chains'][pdb_id] = chains_data
                        data['chain_descriptions'] = chain_descriptions
                        
                        # 4. Perform alignments for each chain
                        if chains_data:
                            for chain_id, (pdb_seq, pdb_res_ids) in chains_data.items():
                                alignment, mapping = _perform_alignment(pdb_seq, pdb_res_ids, data['uniprot_seq'])
                                data['alignments'].append(f"=== {pdb_id} Chain {chain_id} ===\n{alignment}\n")
                                data['mappings'].append(f"=== {pdb_id} Chain {chain_id} ===\n{mapping}\n")
            
            data['status'] = 'complete'
        
        except Exception as e:
            data['status'] = 'error'
            st.error(f"Error processing {uniprot_id}: {str(e)}")
        
        st.rerun()

def main():
    st.set_page_config(page_title="UniProt-PDB Residue mapping", layout="wide")
    st.title("UniProt-PDB Residue mapping")
    
    # Initialize session state variables
    if 'uniprot_ids' not in st.session_state:
        st.session_state.uniprot_ids = []
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = {}
    
    # Input Section
    with st.expander("Input Parameters", expanded=True):
        new_uniprot_id = st.text_input("Enter UniProt ID(s), comma separated (e.g., P12345, Q98765)", key="uniprot_input")
        col1, col2 = st.columns(2)
        with col1:
            add_btn = st.button("Add UniProt ID(s)")
        with col2:
            clear_btn = st.button("Clear All IDs")
        
        if add_btn and new_uniprot_id.strip():
            # Split by comma and clean up the IDs
            ids_to_add = [id.strip().upper() for id in new_uniprot_id.split(',') if id.strip()]
            for uniprot_id in ids_to_add:
                if uniprot_id not in st.session_state.uniprot_ids:
                    st.session_state.uniprot_ids.append(uniprot_id)
                    st.session_state.processing_state[uniprot_id] = {
                        'status': 'pending',
                        'pdb_entries': [],
                        'selected_pdb': None,
                        'chains': {},
                        'selected_chains': [],
                        'uniprot_seq': '',
                        'alignments': [],
                        'mappings': [],
                        'chain_descriptions': {}
                    }
            st.rerun()
        
        if clear_btn:
            st.session_state.uniprot_ids = []
            st.session_state.processing_state = {}
            st.rerun()
    
    # Display current UniProt IDs
    if st.session_state.uniprot_ids:
        st.subheader("Current UniProt IDs to Process:")
        cols = st.columns(4)
        for i, uniprot_id in enumerate(st.session_state.uniprot_ids):
            with cols[i % 4]:
                status = st.session_state.processing_state[uniprot_id]['status']
                status_color = {
                    'pending': 'gray',
                    'processing': 'orange',
                    'complete': 'green',
                    'error': 'red'
                }.get(status, 'gray')
                
                st.markdown(f"""
                <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>{uniprot_id}</span>
                        <button style="background-color: #ff4b4b; color: white; border: none; border-radius: 3px; padding: 2px 5px;" 
                                onclick="document.getElementById('remove_{uniprot_id}').click()">×</button>
                    </div>
                    <div style="color: {status_color}; font-size: 0.8em;">
                        Status: {status.capitalize()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.button("Remove", key=f"remove_{uniprot_id}", on_click=lambda uid=uniprot_id: remove_uniprot_id(uid), 
                         label_visibility="hidden", id=f"remove_{uniprot_id}")
    
    # Process button
    if st.session_state.uniprot_ids:
        if st.button("Process All UniProt IDs"):
            process_all_uniprot_ids()

    # Display results for each UniProt ID
    for uniprot_id in st.session_state.uniprot_ids:
        data = st.session_state.processing_state[uniprot_id]
        
        st.divider()
        st.subheader(f"Results for UniProt ID: {uniprot_id}")
        
        # Show status
        if data['status'] == 'error':
            st.error("Processing failed for this ID")
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

# ... [Keep all the original helper functions (_fetch_pdb_entries_task, _get_pdb_metadata, 
# _fetch_chains_task, _fetch_uniprot_sequence_task, _perform_alignment) the same] ...

if __name__ == "__main__":
    main()
