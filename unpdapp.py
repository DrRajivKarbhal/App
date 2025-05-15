import streamlit as st
import sys
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, Polypeptide
from Bio.Data.IUPACData import protein_letters
from io import StringIO
from datetime import datetime
import time

def main():
    st.set_page_config(page_title="UniProt-PDB Residue mapping", layout="wide")
    st.title("UniProt-PDB Residue mapping")
    
    # Initialize session state variables
    if 'uniprot_ids' not in st.session_state:
        st.session_state.uniprot_ids = []
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Input Section
    with st.expander("Input Parameters", expanded=True):
        # Multi-input for UniProt IDs
        input_ids = st.text_area("Enter UniProt IDs (one per line)", 
                                help="Enter multiple UniProt IDs, one on each line")
        fetch_pdb_btn = st.button("Process UniProt IDs")
        
        if fetch_pdb_btn:
            if not input_ids.strip():
                st.error("Please enter at least one UniProt ID")
            else:
                # Process each UniProt ID
                uniprot_ids = [id.strip() for id in input_ids.split('\n') if id.strip()]
                st.session_state.uniprot_ids = uniprot_ids
                
                with st.spinner(f"Processing {len(uniprot_ids)} UniProt IDs..."):
                    for uniprot_id in uniprot_ids:
                        if uniprot_id not in st.session_state.results:
                            st.session_state.results[uniprot_id] = {
                                'status': 'processing',
                                'pdb_entries': [],
                                'selected_pdb': None,
                                'chain_data': {},
                                'chain_descriptions': {},
                                'selected_chains': [],
                                'uni_seq': "",
                                'alignment_results': [],
                                'mapping_results': []
                            }
                            
                            try:
                                # Fetch PDB entries
                                pdb_entries = _fetch_pdb_entries_task(uniprot_id)
                                st.session_state.results[uniprot_id]['pdb_entries'] = pdb_entries
                                st.session_state.results[uniprot_id]['status'] = 'pdb_fetched'
                                
                            except Exception as e:
                                st.session_state.results[uniprot_id]['status'] = f'error: {str(e)}'
                                st.error(f"Error processing {uniprot_id}: {str(e)}")
                
                st.rerun()
    
    # Display results for each UniProt ID
    if st.session_state.uniprot_ids:
        for uniprot_id in st.session_state.uniprot_ids:
            if uniprot_id not in st.session_state.results:
                continue
                
            result = st.session_state.results[uniprot_id]
            
            # Display UniProt ID header
            st.markdown(f"---\n### UniProt ID: {uniprot_id}")
            
            if result['status'].startswith('error'):
                st.error(f"Error: {result['status'][6:]}")
                continue
            
            # PDB Count Label
            if result['pdb_entries']:
                st.info(f"Total PDB Structures: {len(result['pdb_entries'])}")
            
            # PDB Table Section
            if result['pdb_entries']:
                st.subheader("Available PDB Entries:")
                
                # Create a radio button selection for PDB entries
                pdb_options = [f"{entry['id']} (Resolution: {entry['resolution']}, Length: {entry['length']})" 
                              for entry in result['pdb_entries']]
                selected_pdb = st.radio(f"Select a PDB entry for {uniprot_id}:", 
                                       pdb_options, 
                                       index=None,
                                       key=f"pdb_select_{uniprot_id}")
                
                if selected_pdb:
                    selected_pdb_id = selected_pdb.split()[0]
                    result['selected_pdb'] = selected_pdb_id
                    
                    if st.button(f"Fetch Chains for {selected_pdb_id}", key=f"fetch_chains_{uniprot_id}"):
                        with st.spinner(f"Fetching chains for {selected_pdb_id}..."):
                            try:
                                chains_data, chain_descriptions = _fetch_chains_task(selected_pdb_id)
                                result['chain_data'] = chains_data
                                result['chain_descriptions'] = chain_descriptions
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to fetch chains: {str(e)}")
            
            # Chains Section
            if result['chain_data']:
                st.subheader("Available Chains")
                
                # Display chain checkboxes
                result['selected_chains'] = []
                for chain_id in sorted(result['chain_data'].keys()):
                    seq_length = len(result['chain_data'][chain_id][0])
                    desc = result['chain_descriptions'].get(chain_id, "No description available")
                    if st.checkbox(f"Chain {chain_id} ({seq_length} aa): {desc}", 
                                 key=f"chain_{uniprot_id}_{chain_id}", 
                                 value=True):
                        result['selected_chains'].append(chain_id)
                
                if st.button(f"Process Selected Chains for {uniprot_id}", key=f"process_chains_{uniprot_id}"):
                    if not result['selected_chains']:
                        st.warning("Please select at least one chain")
                    else:
                        # First fetch UniProt sequence if not already done
                        if not result['uni_seq']:
                            with st.spinner(f"Fetching UniProt sequence for {uniprot_id}..."):
                                try:
                                    uni_seq = _fetch_uniprot_sequence_task(uniprot_id)
                                    result['uni_seq'] = uni_seq
                                except Exception as e:
                                    st.error(f"Failed to fetch UniProt sequence: {str(e)}")
                                    continue
                        
                        # Process chains
                        with st.spinner(f"Processing chains for {uniprot_id}..."):
                            progress_bar = st.progress(0)
                            alignment_results = []
                            mapping_results = []
                            
                            for i, chain_id in enumerate(result['selected_chains']):
                                progress_bar.progress((i + 1) / len(result['selected_chains']))
                                if chain_id in result['chain_data']:
                                    pdb_seq, pdb_res_ids = result['chain_data'][chain_id]
                                    
                                    # Validate PDB sequence
                                    if not pdb_seq or not all(aa.upper() in protein_letters for aa in pdb_seq):
                                        alignment_results.append(f"=== Chain {chain_id} ===\nInvalid PDB sequence (contains non-standard amino acids)\n")
                                        mapping_results.append(f"=== Chain {chain_id} ===\nInvalid PDB sequence (contains non-standard amino acids)\n")
                                        continue
                                    
                                    alignment, mapping = _perform_alignment(pdb_seq, pdb_res_ids, result['uni_seq'])
                                    alignment_results.append(f"=== Chain {chain_id} ===\n{alignment}\n")
                                    mapping_results.append(f"=== Chain {chain_id} ===\n{mapping}\n")
                            
                            result['alignment_results'] = alignment_results
                            result['mapping_results'] = mapping_results
                            progress_bar.empty()
                            st.rerun()
            
            # Results Section
            if result['alignment_results'] or result['mapping_results']:
                st.subheader(f"Results for {uniprot_id}")
                
                tab1, tab2 = st.tabs(["Alignment", "Residue Mapping"])
                
                with tab1:
                    st.text("\n".join(result['alignment_results']))
                    
                    # Download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_name = f"alignment_{uniprot_id}_{result['selected_pdb']}_{timestamp}.txt"
                    st.download_button(
                        label=f"Save Alignment for {uniprot_id}",
                        data="\n".join(result['alignment_results']),
                        file_name=default_name,
                        mime="text/plain",
                        key=f"dl_align_{uniprot_id}"
                    )
                
                with tab2:
                    st.text("\n".join(result['mapping_results']))
                    
                    # Download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    default_name = f"mapping_{uniprot_id}_{result['selected_pdb']}_{timestamp}.txt"
                    st.download_button(
                        label=f"Save Mapping for {uniprot_id}",
                        data="\n".join(result['mapping_results']),
                        file_name=default_name,
                        mime="text/plain",
                        key=f"dl_map_{uniprot_id}"
                    )

# [Keep all the helper functions (_fetch_pdb_entries_task, _get_pdb_metadata, 
#  _fetch_chains_task, _fetch_uniprot_sequence_task, _perform_alignment) 
#  exactly the same as in the original script]
# ... (rest of the helper functions remain unchanged)

if __name__ == "__main__":
    main()
