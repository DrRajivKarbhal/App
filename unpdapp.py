import streamlit as st
import requests
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, Polypeptide
from Bio.Data.IUPACData import protein_letters
from io import StringIO
from datetime import datetime
import traceback
import time

def main():
    st.set_page_config(page_title="UniProt-PDB Residue Mapping", layout="wide")
    st.title("UniProt-PDB Residue Mapping")
    
    # Initialize session state
    if 'uniprot_ids' not in st.session_state:
        st.session_state.uniprot_ids = []
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = {}
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Input Section
    with st.expander("Input Parameters", expanded=True):
        new_uniprot_id = st.text_input(
            "Enter UniProt ID(s), comma separated (e.g., P01308, P69905)",
            key="uniprot_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add UniProt ID(s)") and new_uniprot_id.strip():
                add_uniprot_ids(new_uniprot_id)
        with col2:
            if st.button("Clear All IDs"):
                clear_all_ids()

    # Display current UniProt IDs
    display_current_ids()

    # Process button
    if st.session_state.uniprot_ids and not st.session_state.processing:
        if st.button("Process All UniProt IDs"):
            st.session_state.processing = True
            st.rerun()

    # Processing section
    if st.session_state.processing:
        process_all_uniprot_ids()

    # Display results
    display_results()

def add_uniprot_ids(new_uniprot_id):
    """Add new UniProt IDs to the processing list"""
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
                'chain_descriptions': {},
                'error_message': '',
                'error_traceback': ''
            }
    st.rerun()

def clear_all_ids():
    """Clear all UniProt IDs from the processing list"""
    st.session_state.uniprot_ids = []
    st.session_state.processing_state = {}
    st.session_state.processing = False
    st.rerun()

def display_current_ids():
    """Display the current list of UniProt IDs"""
    if st.session_state.uniprot_ids:
        st.subheader("Current UniProt IDs to Process:")
        cols = st.columns(4)
        for i, uniprot_id in enumerate(st.session_state.uniprot_ids):
            with cols[i % 4]:
                container = st.container(border=True)
                status = st.session_state.processing_state[uniprot_id]['status']
                container.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>{uniprot_id}</span>
                    <div style="color: #666; font-size: 0.8em;">
                        Status: {status}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if container.button("Remove", key=f"remove_{uniprot_id}"):
                    remove_uniprot_id(uniprot_id)

def remove_uniprot_id(uniprot_id):
    """Remove a specific UniProt ID from the processing list"""
    st.session_state.uniprot_ids.remove(uniprot_id)
    del st.session_state.processing_state[uniprot_id]
    st.rerun()

def process_all_uniprot_ids():
    """Process all UniProt IDs in sequence"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_ids = len(st.session_state.uniprot_ids)
    for idx, uniprot_id in enumerate(st.session_state.uniprot_ids):
        data = st.session_state.processing_state[uniprot_id]
        data['status'] = 'processing'
        progress = (idx / total_ids) * 100
        progress_bar.progress(int(progress))
        status_text.text(f"Processing {uniprot_id} ({idx+1}/{total_ids})...")
        st.rerun()
        
        try:
            # 1. Fetch PDB entries
            if not data['pdb_entries']:
                data['pdb_entries'] = fetch_pdb_entries(uniprot_id)
                if not data['pdb_entries']:
                    raise ValueError(f"No PDB entries found for {uniprot_id}")

            # 2. Fetch UniProt sequence
            if not data['uniprot_seq']:
                data['uniprot_seq'] = fetch_uniprot_sequence(uniprot_id)
                if not data['uniprot_seq']:
                    raise ValueError(f"Could not fetch sequence for {uniprot_id}")

            # 3. Process each PDB entry
            for pdb_entry in data['pdb_entries']:
                pdb_id = pdb_entry['id']
                if pdb_id not in data['chains']:
                    chains_data, chain_descriptions = fetch_pdb_chains(pdb_id)
                    if not chains_data:
                        st.warning(f"No chains found in PDB {pdb_id}")
                        continue
                        
                    data['chains'][pdb_id] = chains_data
                    data['chain_descriptions'].update(chain_descriptions)
                    
                    # 4. Perform alignments for each chain
                    for chain_id, (pdb_seq, pdb_res_ids) in chains_data.items():
                        try:
                            alignment, mapping = perform_alignment(pdb_seq, pdb_res_ids, data['uniprot_seq'])
                            data['alignments'].append(f"=== {pdb_id} Chain {chain_id} ===\n{alignment}\n")
                            data['mappings'].append(f"=== {pdb_id} Chain {chain_id} ===\n{mapping}\n")
                        except Exception as e:
                            st.warning(f"Alignment failed for {pdb_id} chain {chain_id}: {str(e)}")
                            continue

            data['status'] = 'complete'
        
        except Exception as e:
            data['status'] = 'error'
            data['error_message'] = str(e)
            data['error_traceback'] = traceback.format_exc()
            st.error(f"Error processing {uniprot_id}: {str(e)}")
        
        # Update progress
        progress = ((idx + 1) / total_ids) * 100
        progress_bar.progress(int(progress))
        time.sleep(0.1)  # Small delay to allow UI to update
        st.rerun()
    
    # Final update
    progress_bar.progress(100)
    status_text.text("Processing complete!")
    time.sleep(1)
    st.session_state.processing = False
    st.rerun()

def display_results():
    """Display results for each processed UniProt ID"""
    for uniprot_id in st.session_state.uniprot_ids:
        data = st.session_state.processing_state[uniprot_id]
        
        st.divider()
        st.subheader(f"Results for UniProt ID: {uniprot_id}")
        
        # Show status
        if data['status'] == 'error':
            st.error(f"Processing failed: {data.get('error_message', 'Unknown error')}")
            if st.button(f"Show details for {uniprot_id}", key=f"details_{uniprot_id}"):
                st.text(data.get('error_traceback', 'No traceback available'))
        elif data['status'] == 'complete':
            st.success("Processing complete")
        elif data['status'] == 'processing':
            st.warning("Processing in progress...")
        
        # Show results if available
        if data['status'] == 'complete':
            display_pdb_entries(uniprot_id, data)
            display_alignment_results(uniprot_id, data)

def display_pdb_entries(uniprot_id, data):
    """Display PDB entries for a UniProt ID"""
    if data['pdb_entries']:
        st.info(f"Found {len(data['pdb_entries'])} PDB entries")
        
        pdb_options = [f"{entry['id']} (Resolution: {entry.get('resolution', 'N/A')}, Length: {entry.get('length', 'N/A')})" 
                      for entry in data['pdb_entries']]
        selected_idx = st.selectbox(
            f"Select PDB entry for {uniprot_id} to view details:",
            range(len(pdb_options)),
            format_func=lambda x: pdb_options[x],
            key=f"pdb_select_{uniprot_id}"
        )
        
        if selected_idx is not None and data['pdb_entries']:
            selected_pdb = data['pdb_entries'][selected_idx]
            data['selected_pdb'] = selected_pdb['id']
            
            if data['selected_pdb'] in data['chains']:
                st.subheader(f"Chains in PDB {data['selected_pdb']}")
                for chain_id, chain_data in data['chains'][data['selected_pdb']].items():
                    seq_len = len(chain_data[0]) if chain_data and len(chain_data) > 0 else 0
                    st.write(f"**Chain {chain_id}** ({seq_len} aa): {data['chain_descriptions'].get(chain_id, 'No description')}")

def display_alignment_results(uniprot_id, data):
    """Display alignment results for a UniProt ID"""
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

# API and processing functions
def fetch_pdb_entries(uniprot_id):
    """Fetch PDB entries for a given UniProt ID"""
    try:
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pdb_entries = []
        for pdb_id, entry_data in data.get(uniprot_id, {}).items():
            try:
                metadata = get_pdb_metadata(pdb_id)
                pdb_entries.append({
                    'id': pdb_id,
                    'resolution': metadata.get('resolution', 'N/A'),
                    'length': metadata.get('length', 'N/A')
                })
            except:
                pdb_entries.append({
                    'id': pdb_id,
                    'resolution': 'N/A',
                    'length': 'N/A'
                })
        
        return pdb_entries
    
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch PDB entries for {uniprot_id}: {str(e)}")
        return []
    except Exception as e:
        st.warning(f"Error processing PDB entries for {uniprot_id}: {str(e)}")
        return []

def get_pdb_metadata(pdb_id):
    """Get basic metadata for a PDB entry"""
    try:
        url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pdb_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            entry = data.get(pdb_id, [{}])[0]
            return {
                'resolution': entry.get('resolution', 'N/A'),
                'length': entry.get('chain_count', {}).get('protein', 'N/A')
            }
        return {}
    except:
        return {}

def fetch_pdb_chains(pdb_id):
    """Fetch chain sequences and residue IDs from a PDB file"""
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        pdb_file = StringIO(response.text)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)
        
        chains_data = {}
        chain_descriptions = {}
        
        for model in structure:
            for chain in model:
                try:
                    chain_id = chain.id
                    polypeptides = Polypeptide.Polypeptide(chain)
                    seq = polypeptides.get_sequence()
                    res_ids = [f"{res.parent.id}_{res.id[1]}" for res in polypeptides]
                    
                    chains_data[chain_id] = (str(seq), res_ids)
                    chain_descriptions[chain_id] = f"Chain {chain_id} of {pdb_id}"
                except Exception as e:
                    st.warning(f"Could not process chain {chain.id} in {pdb_id}: {str(e)}")
                    continue
        
        return chains_data, chain_descriptions
    
    except Exception as e:
        st.warning(f"Could not fetch chains for {pdb_id}: {str(e)}")
        return {}, {}

def fetch_uniprot_sequence(uniprot_id):
    """Fetch the UniProt sequence"""
    try:
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        fasta = StringIO(response.text)
        record = SeqIO.read(fasta, "fasta")
        return str(record.seq)
    
    except Exception as e:
        st.warning(f"Could not fetch sequence for {uniprot_id}: {str(e)}")
        return ""

def perform_alignment(pdb_seq, pdb_res_ids, uniprot_seq):
    """Perform sequence alignment between PDB and UniProt sequences"""
    try:
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.substitution_matrix = protein_letters
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -0.5
        
        alignments = aligner.align(uniprot_seq, pdb_seq)
        best_alignment = alignments[0]
        
        # Generate alignment string
        alignment_str = f"UniProt: {best_alignment[0]}\nPDB:     {best_alignment[1]}"
        
        # Generate residue mapping
        mapping = []
        uniprot_pos = 0
        pdb_pos = 0
        
        for u, p in zip(best_alignment[0], best_alignment[1]):
            if u != '-':
                uniprot_pos += 1
            if p != '-':
                pdb_pos += 1
                if p != '-' and u != '-':
                    mapping.append(f"UniProt {uniprot_pos} <-> PDB {pdb_res_ids[pdb_pos-1]}")
        
        return alignment_str, "\n".join(mapping)
    
    except Exception as e:
        raise ValueError(f"Alignment failed: {str(e)}")

if __name__ == "__main__":
    main()
