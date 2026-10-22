# Retrieve genomic sequence from NCBI.
from Bio import Entrez, SeqIO
from Bio.Seq import UndefinedSequenceError
Entrez.email = "your.email@example.com"
handle = Entrez.esearch(db="nucleotide", term="BRCA1[Gene] AND Homo sapiens[Organism]", retmax=1)
record = Entrez.read(handle)
gene_id = record['IdList'][0]
print(f"Gene ID: {gene_id}")
try:
   fetch_handle = Entrez.efetch(db="nucleotide", id=gene_id, rettype="gb", retmode="text")
   seq_record = SeqIO.read(fetch_handle, "genbank")
   fetch_handle.close()
   seq_str = str(seq_record.seq)
except UndefinedSequenceError:
   print("Sequence undefined in GenBank -- retrying with FASTA...")
   fetch_handle = Entrez.efetch(db="nucleotide", id=gene_id, rettype="fasta", retmode="text")
   seq_record = SeqIO.read(fetch_handle, "fasta")
   fetch_handle.close()
   seq_str = str(seq_record.seq)
print(f"Gene Name: {seq_record.name}")
print(f"Description: {seq_record.description}")
print(f"Sequence length: {len(seq_str)}")
print(seq_str[:200])

from Bio import Entrez, SeqIO
Entrez.email = "your.email@example.com"
handle = Entrez.efetch(db="nucleotide", id="NM_007294", rettype="gb", retmode="text")
seq_record = SeqIO.read(handle, "genbank")
print(f"Number of features: {len(seq_record.features)}")
for feature in seq_record.features:
   print(feature.type)
for feature in seq_record.features:
   if feature.type == "CDS":
       print("CDS (Coding Sequence):")
       print("Location:", feature.location)
       print("Protein:", feature.qualifiers.get('product', ['N/A'])[0])
   elif feature.type == "gene":
       print("Gene:")
       print("Location:", feature.location)
       print("Gene Name:", feature.qualifiers.get('gene', ['N/A'])[0])
gene_start = feature.location.start
promoter_seq = seq_record.seq[gene_start - 1000:gene_start]
print("Promoter region (1000 bp upstream):")
print(promoter_seq)
vcf_content = """##fileformat=VCFv4.2
##source=Sample_Bioinformatics_Practical
##reference=GRCh38
#CHROM  POS ID  REF ALT QUAL  FILTER  INFO
17  43045700  rs80357906  C T 99  PASS  GENE=BRCA1
17  43045701  rs80357907  A G 95  PASS  GENE=BRCA1
17  43045702  rs80357908  G A 80  PASS  GENE=BRCA1
17  43045703  rs80357909  T C 65  PASS  GENE=BRCA1
17  43045704  rs80357910  C G 40  PASS  GENE=BRCA1
"""
with open("sample.vcf", "w") as f:
   f.write(vcf_content)
print("sample.vcf created successfully!")
print("Complete VCF Data:")
print(vcf_content)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
vcf_file = "sample.vcf"
with open(vcf_file, "r") as f:
    lines = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("##")
    ]
columns = lines[0].split()
data = [line.split() for line in lines[1:]]
vcf_df = pd.DataFrame(data, columns=columns)
print("Sample Variants:")
print(vcf_df[["#CHROM", "POS", "ID", "REF", "ALT", "QUAL"]].head())
vcf_df["QUAL"] = pd.to_numeric(vcf_df["QUAL"], errors="coerce")
high_quality = vcf_df[vcf_df["QUAL"] >= 90]
print("\nHigh-quality variants (QUAL >= 90):")
print(high_quality[["#CHROM", "POS", "ID", "REF", "ALT", "QUAL"]])
plt.figure(figsize=(10, 4))
sns.histplot(vcf_df["QUAL"].dropna(), bins=10)
plt.title("Variant Quality Score Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Number of Variants")
plt.tight_layout()
plt.show()