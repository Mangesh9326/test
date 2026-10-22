Phase 1 : Obtain the target protein sequence

Go to UniProt https://www.uniprot.org/
search this in search bar Human insulin
open protein sequence data for exmple.  P01308
Click on "Downnload" and select Dataset: Entry and Format: FASTS (canonical) and Download it


Phase 2 : Find the template using BLAST

Visit NCBI BLASTp:  https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastp
Click on protien BLAST card
select blastp
Paste the downloaded protien sequence and select Databse to "Protein Data Bank proteins(pdb)"
Scroll down and "Click on Blast"



Percent Identity    |    Query Coveragefrom
   30 to 100        |       70 to 100

Filter the sequences


Click on sequence & on the right side structure in 3D display



Phase 3 : Sequence Alignment

Method A: Online Alignment Tool

Go to Clustal Omega https://www.ebi.ac.uk/jdispatcher/
In the tool categories section, select cluster omega from Multiple Sequence Alignment
Paste target and template sequences.
Run Alignment (Clcick Submit)
Download/save the aligned sequences

Run 6th pract code


Phase 4 : Build 3D Model

Method A: Swiss-Model.

Visit SWISS-MODEL https://swissmodel.expasy.org/
Choose “Start modelling”
Paste your FASTA sequence
Download the 3D model file.




Phase 5:  Model Validation

https://saves.mbi.ucla.edu/
Upload the downloaded model to: SAVES server (includes PROCHECK, ERRAT, VERIFY3D) 


ProSA-web https://prosa.services.came.sbg.ac.at/prosa.php



Phase 6:  Visualize the model

Use tools like:
PyMOL -  https://pymol.org/
Mol* Viewer - https://molstar.org/viewer/

Open your .pdb file to:
Inspect structure
Compare with template
Analyze secondary structures, active site, loops, etc.