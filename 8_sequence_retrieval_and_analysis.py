# SEQUENCE RETRIEVAL AND ANALYSIS

from Bio import Entrez, SeqIO

# Set your email address for NCBI Entrez requests
Entrez.email = "type_your_gmail_@gmail.com"


def retrieve_genbank_record(accession):
    # Retrieve a GenBank record from the NCBI nucleotide database
    print(f"\nRetrieving GenBank record: {accession}")

    with Entrez.efetch(
        db="nucleotide",
        id=accession,
        rettype="gb",
        retmode="text"
    ) as handle:
        record = SeqIO.read(handle, "genbank")

    return record


def display_record_details(record):
    # Display the main information stored in a GenBank record
    print("\nRECORD INFORMATION")
    print("-" * 60)
    print(f"Record ID   : {record.id}")
    print(f"Description : {record.description}")
    print(f"Sequence Length: {len(record.seq)} bases")
    print(f"Molecule Type: {record.annotations.get('molecule_type', 'Not available')}")

    # Display the record annotations
    print("\nRECORD ANNOTATIONS")
    print("-" * 60)
    for key, value in record.annotations.items():
        print(f"{key}: {value}")


def display_dna_and_protein(record):
    # Display the DNA sequence and translate it into amino acids
    print("\nDNA SEQUENCE")
    print("-" * 60)
    print(record.seq)

    # Translate the nucleotide sequence using the standard genetic code
    protein_seq = record.seq.translate()
    print("\nTRANSLATED PROTEIN SEQUENCE")
    print("-" * 60)
    print(protein_seq)


def display_genbank_comment(record):
    # Display the GenBank comment if it exists
    print("\nGENBANK COMMENT")
    print("-" * 60)
    comment = record.annotations.get("comment", "No comment available.")
    print(comment)


def display_fasta(record):
    # Display the record in FASTA format with 60 bases per line
    print("\nFASTA FORMAT")
    print("-" * 60)
    print(f">{record.id} {record.description}")

    sequence = str(record.seq)
    for i in range(0, len(sequence), 60):
        print(sequence[i:i + 60])


def main():
    # Retrieve and analyze both example GenBank records
    accessions = ["KT220438", "FJ966082"]

    for accession in accessions:
        try:
            record = retrieve_genbank_record(accession)

            print("\n" + "=" * 60)
            print(f"SEQUENCE ANALYSIS: {accession}")
            print("=" * 60)

            display_record_details(record)
            display_dna_and_protein(record)
            display_genbank_comment(record)
            display_fasta(record)

        except Exception as e:
            print(f"\nError processing {accession}: {e}")


if __name__ == "__main__":
    main()