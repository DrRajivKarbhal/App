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
                    timestamp = datetime.now().strftime("%Ym%d_%H%M%S")
                    default_name = f"mapping_{uniprot_id}_{result['selected_pdb']}_{timestamp}.txt"
                    st.download_button(
                        label=f"Save Mapping for {uniprot_id}",
                        data="\n".join(result['mapping_results']),
                        file_name=default_name,
                        mime="text/plain",
                        key=f"dl_map_{uniprot_id}"
                    )

def _fetch_pdb_entries_task(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch UniProt entry {uniprot_id}")
        
    data = response.json()
    pdbs = []
    for xref in data.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "PDB":
            pdbs.append(xref.get("id"))
            
    pdb_entries = []
    for pdb_id in sorted(pdbs):
        meta = _get_pdb_metadata(pdb_id)
        pdb_entries.append({
            "id": pdb_id,
            "title": meta.get("title", "N/A"),
            "resolution": meta.get("resolution", "N/A"),
            "length": meta.get("length", "N/A"),
            "organisms": meta.get("organisms", "N/A")
        })
        
    return pdb_entries
    
def _get_pdb_metadata(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code != 200:
        return {}
        
    data = response.json()
    title = data.get("struct", {}).get("title", "N/A")
    resolution = data.get("rcsb_entry_info", {}).get("resolution_combined", ["N/A"])[0]
    length = data.get("rcsb_entry_info", {}).get("deposited_polymer_monomer_count", "N/A")
    organisms = []
    
    for entity in data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", []):
        entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity}"
        entity_resp = requests.get(entity_url)
        if entity_resp.status_code == 200:
            entity_data = entity_resp.json()
            source = entity_data.get("rcsb_entity_source_organism", [{}])[0]
            organism = source.get("scientific_name", "N/A")
            if organism not in organisms:
                organisms.append(organism)
                
    return {
        "title": title,
        "resolution": resolution,
        "length": length,
        "organisms": ", ".join(organisms)
    }
    
def _fetch_chains_task(pdb_id):
    try:
        # Download PDB structure
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDB {pdb_id}")
        pdb_text = response.text

        # Extract chain sequences and descriptions
        chains_data = {}
        chain_descriptions = {}
        
        # First parse the header for chain descriptions
        current_compound = ""
        for line in pdb_text.split('\n'):
            if line.startswith('COMPND'):
                current_compound += line[10:].strip() + " "
                if ';' in line:
                    # End of compound record
                    if 'CHAIN:' in current_compound:
                        parts = current_compound.split('CHAIN:')
                        if len(parts) > 1:
                            chains_part = parts[1].split(';')[0].strip()
                            chains = [c.strip() for c in chains_part.split(',')]
                            description = parts[0].strip()
                            for chain in chains:
                                chain_descriptions[chain] = description
                    current_compound = ""
        
        # Then parse the structure for sequences
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pdb", StringIO(pdb_text))
        
        standard_aa = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
            'THR', 'TRP', 'TYR', 'VAL'
        }

        for model in structure:
            for chain in model:
                seq = ""
                res_ids = []
                for residue in chain:
                    hetflag, resseq, icode = residue.get_id()
                    if hetflag != " ":
                        continue
                    resname = residue.get_resname().strip()
                    if resname not in standard_aa:
                        continue
                    try:
                        aa = Polypeptide.three_to_one(resname)
                    except Exception:
                        continue
                    seq += aa
                    res_ids.append(resseq)
                if seq:
                    chains_data[chain.id] = (seq, res_ids)
            break  # Only process first model

        # If no chains found, try alternative approach from FASTA
        if not chains_data:
            fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
            fasta_response = requests.get(fasta_url)
            if fasta_response.status_code == 200:
                for record in SeqIO.parse(StringIO(fasta_response.text), "fasta"):
                    desc = record.description
                    if "Chain" in desc:
                        chain_id = desc.split("Chain")[1].split(",")[0].strip()
                        chains_data[chain_id] = (str(record.seq), list(range(1, len(record.seq)+1)))
                        chain_descriptions[chain_id] = desc

        return chains_data, chain_descriptions

    except Exception as e:
        raise Exception(f"Error processing PDB chains: {str(e)}")
        
def _fetch_uniprot_sequence_task(uniprot_id):
    try:
        # Use the new UniProt REST API endpoint
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch UniProt entry {uniprot_id}")
        
        # Parse the FASTA record
        fasta = SeqIO.read(StringIO(response.text), "fasta")
        return str(fasta.seq)
    except Exception as e:
        raise Exception(f"Error fetching UniProt sequence: {str(e)}")
    
def _perform_alignment(pdb_seq, pdb_res_ids, uni_seq):
    try:
        # Verify sequences
        if not pdb_seq or not uni_seq:
            return "Empty sequence encountered", "Cannot align empty sequences"
        
        # Initialize aligner with optimized parameters
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -0.5
        
        # Use the correct substitution matrix syntax for your Biopython version
        try:
            # Try the new syntax first (Biopython 1.78+)
            from Bio.Align import substitution_matrices
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        except (ImportError, AttributeError):
            # Fallback to older syntax
            try:
                aligner.substitution_matrix = PairwiseAligner.substitution_matrices.load("BLOSUM62")
            except AttributeError:
                # If neither works, use a simple match/mismatch scoring
                aligner.match_score = 2
                aligner.mismatch_score = -1
        
        # Perform alignment
        alignments = aligner.align(pdb_seq, uni_seq)
        
        if not alignments:
            return "No alignment could be generated", "No residue mapping available"
        
        alignment = alignments[0]  # Take the best alignment
        
        # Format alignment output
        alignment_str = f"Alignment between PDB chain and UniProt {uniprot_id}\n"
        alignment_str += f"Alignment score: {alignment.score:.2f}\n"
        alignment_str += f"PDB sequence length: {len(pdb_seq)}\n"
        alignment_str += f"UniProt sequence length: {len(uni_seq)}\n\n"
        alignment_str += str(alignment)
        
        # Generate residue mapping
        mapping_str = "PDB_ResID  UniProt_Pos  PDB_AA  UniProt_AA\n"
        mapping_str += "-"*50 + "\n"
        
        # Get aligned sequences
        aligned_pdb = alignment.aligned[0]
        aligned_uni = alignment.aligned[1]
        
        pdb_pos = 0
        uni_pos = 0
        
        for (p_start, p_end), (u_start, u_end) in zip(aligned_pdb, aligned_uni):
            # Handle gaps before this alignment block
            while pdb_pos < p_start:
                mapping_str += f"{pdb_res_ids[pdb_pos]:<10}{'-':<12}{pdb_seq[pdb_pos]:<8}{'-':<8}\n"
                pdb_pos += 1
                
            while uni_pos < u_start:
                mapping_str += f"{'-':<10}{uni_pos+1:<12}{'-':<8}{uni_seq[uni_pos]:<8}\n"
                uni_pos += 1
                
            # Handle aligned residues
            while p_start < p_end and u_start < u_end:
                mapping_str += f"{pdb_res_ids[pdb_pos]:<10}{uni_pos+1:<12}{pdb_seq[pdb_pos]:<8}{uni_seq[uni_pos]:<8}\n"
                pdb_pos += 1
                uni_pos += 1
                p_start += 1
                u_start += 1
        
        # Handle any remaining residues
        while pdb_pos < len(pdb_seq):
            mapping_str += f"{pdb_res_ids[pdb_pos]:<10}{'-':<12}{pdb_seq[pdb_pos]:<8}{'-':<8}\n"
            pdb_pos += 1
            
        while uni_pos < len(uni_seq):
            mapping_str += f"{'-':<10}{uni_pos+1:<12}{'-':<8}{uni_seq[uni_pos]:<8}\n"
            uni_pos += 1
            
        return alignment_str, mapping_str
        
    except Exception as e:
        error_msg = f"Alignment error: {str(e)}"
        return error_msg, error_msg

if __name__ == "__main__":
    main()
