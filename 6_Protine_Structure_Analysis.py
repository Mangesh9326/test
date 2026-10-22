from IPython.display import display
from Bio.Blast import NCBIWWW, NCBIXML
import py3Dmol
import requests

# HELPER FUNCTIONS
def get_sequence_from_uniprot(uniprot_id: str) -> str | None:
    """Fetch FASTA sequence from UniProt given a UniProt ID."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        print("✓ Sequence retrieved successfully!\n")
        return response.text

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching protein sequence: {e}")
        return None


def run_blast_pdb(
    fasta_sequence: str,
    min_identity: float = 30.0,
    min_coverage: float = 70.0
):
    """Run a BLAST search against the PDB database using Biopython."""
    print("Running BLAST search against PDB (this may take a few minutes)...")

    try:
        result_handle = NCBIWWW.qblast(
            "blastp", "pdb", fasta_sequence
        )
        blast_record = NCBIXML.read(result_handle)

    except Exception as e:
        print(f"✗ BLAST execution failed: {e}")
        return

    print("\n" + "=" * 60)
    print(" TOP MATCHING TEMPLATES ")
    print("=" * 60)

    found = False

    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            identity = (hsp.identities / hsp.align_length) * 100
            coverage = (
                (hsp.query_end - hsp.query_start + 1)
                / blast_record.query_length
            ) * 100

            if identity >= min_identity and coverage >= min_coverage:
                found = True
                print(f"PDB ID   : {alignment.hit_id}")
                print(f"Identity : {identity:.2f}%")
                print(f"Coverage : {coverage:.2f}%")
                print(f"E-value  : {hsp.expect:.2e}")
                print("-" * 50)

    if not found:
        print("No suitable template found meeting the threshold criteria.")


def visualize_structure(
    pdb_id: str,
    width: int = 900,
    height: int = 700
):
    """Download and render a 3D structure in Jupyter using py3Dmol."""
    print(f"Loading structure {pdb_id} ...")

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        pdb_data = response.text

    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download PDB structure: {e}")
        return

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(pdb_data, "pdb")
    viewer.setStyle({"cartoon": {"color": "spectrum"}})
    viewer.addSurface(
        py3Dmol.VDW,
        {"opacity": 0.25, "color": "white"}
    )
    viewer.zoomTo()

    return viewer.show()

# MAIN EXECUTION

if __name__ == "__main__":
    PROTEIN_ID = "P01308"
    PDB_ID = "4INS"

    fasta_seq = get_sequence_from_uniprot(PROTEIN_ID)

    if fasta_seq:
        print(fasta_seq)
        # run_blast_pdb(fasta_seq)
        visualize_structure(PDB_ID)
