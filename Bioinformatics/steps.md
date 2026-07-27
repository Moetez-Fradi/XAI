# XQdrant: Explainable Protein Embedding Retrieval — PSB 2027 Paper Plan
 
**Target venue:** Pacific Symposium on Biocomputing (PSB) 2027, Big Island of Hawaii, Jan 3–7, 2027
**Target session:** *Biological Molecular Function: Beyond Homology*
**Paper deadline:** August 3, 2026, 11:59 PM PT
**Page limit:** 12 pages (excludes cover letter, title page, references)
**Format:** PDF, PSB template (World Scientific), cover letter required (must include LLM-use disclosure)
 
---
 
## 1. Project Summary
 
We are building an explainable protein similarity search system:
 
**Pipeline:** `.pdb file → ESM2 embedding → XQdrant indexing/retrieval → dimension-level attribution → structural interpretation`
 
Unlike standard vector databases that return a ranked top-k list, **XQdrant** returns *which specific embedding dimensions* drove a match and *how much* each contributed. The paper's core claim: these high-contributing dimensions correspond to real, physically coherent structural/functional features (hydrophobic core packing, helix density, thermostability determinants, active-site geometry) — not noise. This reframes protein similarity search from "what is similar" to "similar *how*," with direct utility for functional annotation and protein engineering.
 
**Team:**
- Computer science lead (pipeline, ESM2 embedding, XQdrant engineering, statistical analysis)
- Biology co-author (INSAT, structural biochemistry/molecular biology) — ground-truth curation, structural validation, PyMOL mapping, biological framing
---
 
## 2. Title & Abstract (working draft)
 
**Title:** *Beyond Top-k: Dimension-Resolved Explainability in Protein Language Model Embeddings for Mechanistic Functional Annotation*
(Alt: *XQdrant: Explainable Embedding Retrieval Reveals the Physicochemical Basis of Protein Similarity*)
 
**Abstract (draft):**
> Protein language models such as ESM2 encode rich structural and functional signal in high-dimensional embeddings, yet similarity search built on these embeddings remains a black box: existing tools return a ranked list of neighbors without explaining *why* two proteins are close in embedding space. This opacity limits their utility for hypothesis-driven biology, where researchers need to know whether a match reflects shared catalytic architecture, a conserved hydrophobic core, thermal stability determinants, or coincidental sequence similarity. We present **XQdrant**, an explainable vector retrieval system that decomposes protein similarity into the specific embedding dimensions driving it, together with each dimension's quantitative contribution. Applied to ESM2 embeddings derived directly from PDB structures, XQdrant allows a query protein to be screened not just for "what is similar" but "similar *how*" — surfacing dimensions that correlate with annotated structural motifs and functional properties. We validate these dimension-level explanations against ground truth from DSSP/PyMOL-derived structural annotations and known functional families, showing that high-contributing dimensions map onto physically coherent regions of the protein rather than diffuse background signal, via permutation-tested statistical significance. In case studies spanning thermostability determinants and enzyme active-site geometry, we show XQdrant identifies functionally relevant matches and recovers structural rationale automatically, where alignment-based tools (TM-align) require a separate manual step — at embedding-search speed. This establishes a new paradigm for explainable, mechanism-aware protein screening.
 
*(Finalize numbers/claims only after Experiments B–D produce results.)*
 
---
 
## 3. Datasets to Acquire
 
| Dataset | Source | Purpose |
|---|---|---|
| SCOP / SCOPe | scop.berkeley.edu / scop2.mrc-lmb.cam.ac.uk | Structural family gold-standard labels |
| CATH | cathdb.info | Independent structural classification cross-check |
| PDB structures (bulk) | RCSB PDB FTP / `pypdb` / REST API | Raw input structures |
| Pfam / InterPro | pfam.xfam.org, ebi.ac.uk/interpro | Functional domain labels |
| UniProt / SwissProt + GO | uniprot.org | Functional benchmark ground truth (EC numbers, GO terms, active-site annotations) |
| Thermophile/mesophile ortholog pairs | OMA Browser (omabrowser.org) + literature-curated pairs (e.g., *Thermus thermophilus* vs *E. coli*) | Thermostability case study |
| CASP / functional benchmark subsets | predictioncenter.org, or subsets used in DeepFRI/ProteInfer papers | Functional annotation transfer benchmark + baseline comparability |
| ESM2 checkpoints | huggingface.co/facebook (esm2_t33_650M primary; note as limitation if not testing multiple sizes) | Embedding backbone |
| DSSP (`mkdssp`) | Anaconda / dssp binary | Per-residue secondary structure, solvent accessibility |
| Foldseek | github.com/steineggerlab/foldseek | Structure-based retrieval baseline |
| TM-align | zhanggroup.org/TM-align | Structural alignment / explanation baseline |
| BLAST+ (BLASTp) | NCBI | Sequence-based retrieval baseline |
 
**Held-out test set:** ~500–2,000 proteins, stratified across SCOP folds, never used during development/tuning.
 
---
 
## 4. Experiments
 
### A — Retrieval Accuracy (baseline sanity check)
- Metrics: Recall@1/5/10, mAP, fold-classification accuracy (family/superfamily/fold level).
- Baselines: BLASTp, Foldseek, raw ESM2 cosine similarity (no attribution layer), TM-align.
- Claim: attribution layer doesn't degrade retrieval quality vs. plain ESM2 search.
### B — Dimension-Attribution Validity (core novelty, most scrutinized)
- For each retrieved pair, extract top-N contributing dimensions from XQdrant.
- Compute independent per-residue structural features via DSSP: secondary structure %, Kyte-Doolittle hydrophobicity, solvent accessibility, B-factor.
- Statistical test: permutation/bootstrap — shuffle dimension-to-feature mapping N times, compare real attribution correlation vs. null distribution. Report p-values.
### C — Negative Controls (specificity)
- Random unrelated pairs (different fold/function) → attributions should be diffuse.
- Sequence-similar/structurally-divergent pairs and structurally-convergent/sequence-divergent pairs → shows tracking of structure/function, not trivial sequence identity.
- Report as ROC/precision-recall curve (attribution confidence vs. true structural relatedness).
### D — Thermostability Case Study
- Run on thermophile/mesophile ortholog pairs.
- Test correlation of top dimensions with known thermostability correlates: ion-pair density, charged-residue surface content, loop length, packing density (FreeSASA or similar).
- PyMOL structure coloring by attribution weight, thermophile vs. mesophile side-by-side.
### E — Functional Annotation Transfer
- SwissProt/GO-labeled subset: do top-attributed dimensions correspond to residues near annotated active sites/catalytic residues (UniProt feature tables)?
- Compare against Foldseek/TM-align: do their alignments also localize to the active site, or only report global fold similarity?
### F — Runtime / Workflow Comparison
- Time: XQdrant single query vs. (BLAST/Foldseek retrieval → TM-align on candidates → manual PyMOL inspection).
- Report wall-clock time and step count.
### Ablations (2–3 minimum)
- ESM2 layer choice (embedding source layer) vs. attribution quality.
- Number of attributed dimensions (top-5/20/50) sensitivity.
- Pooling strategy (mean-pool vs. per-residue vs. CLS-equivalent).
---
 
## 5. Statistical Rigor Requirements
- Confidence intervals or significance tests for every headline claim (no bare point estimates).
- Report variance across ≥3 independent runs/seeds where stochasticity exists.
- Explicitly state in Methods that the held-out test set was not used to tune attribution thresholds.
---
 
## 6. Biology Co-Author Deliverables
 
1. **Ground-truth structural annotation sets** — 3–5 well-characterized protein families (e.g., TIM barrels, four-helix bundles, beta-propellers) from PDB/SCOP/CATH, with per-residue annotation tables (DSSP, hydrophobicity, solvent accessibility, B-factor, catalytic/active-site residues).
2. **Thermophile/mesophile ortholog pairs** — matched pairs with literature-supported thermostability rationale.
3. **PyMOL structural mapping** — top-N attributed dimensions projected onto 3D structure (via per-residue attribution or perturbation/ablation mapping), rendered as attribution-weighted heatmap/putty coloring. Target 2–3 polished figures over many rough ones.
4. **Negative control curation** — random unrelated pairs; sequence-similar/structure-divergent and structure-similar/sequence-divergent (convergent evolution) pairs.
5. **Functional annotation benchmark** — independent assessment of whether XQdrant explanations align with real shared mechanism (e.g., catalytic triad geometry), not spurious correlates.
6. **Writing** — biological framing of Introduction/Discussion (why explainability matters for enzyme engineering, thermostability engineering, drug target screening) and the Limitations paragraph (correlational vs. causal explanation caveat).
---
 
## 7. Figures & Tables to Produce
 
1. Retrieval accuracy table: XQdrant vs. BLAST / Foldseek / TM-align / raw ESM2.
2. Attribution validity plot: real vs. permuted-null correlation with structural features, significance markers (violin or bar chart).
3. 2–3 PyMOL structures colored by dimension-attribution weight (thermophile/mesophile pair + one active-site case).
4. Negative-control precision/specificity plot (ROC/PR curve).
5. Workflow comparison table: steps, wall-clock time, automatic vs. manual rationale.
6. Pipeline schematic: PDB → ESM2 → XQdrant → attributed dimensions → structural mapping (secondary priority vs. biology figures).
---
 
## 8. Paper Structure (target ≤12 pages)
 
1. **Introduction** — the explainability gap in embedding-based protein search; framing vs. BLAST/TM-align/Foldseek; contribution statement.
2. **Related Work** — protein language models, vector similarity search, existing interpretability approaches in bioinformatics.
3. **Methods**
   - ESM2 embedding pipeline (.pdb → embedding)
   - XQdrant indexing and dimension-attribution algorithm (full detail for reproducibility)
   - Datasets and baselines
   - Statistical testing procedures (permutation test, significance thresholds)
4. **Results**
   - A: Retrieval accuracy
   - B: Attribution validity (permutation test results)
   - C: Negative controls / specificity
   - D: Thermostability case study
   - E: Functional annotation transfer
   - F: Runtime/workflow comparison
   - Ablations
5. **Discussion** — implications for protein engineering workflows; hypothesis-generation use case; honest limitations (ambiguous/multi-factor dimensions, single ESM2 checkpoint, correlational nature of explanations unless perturbation-validated).
6. **Conclusion**
7. **Cover letter (separate, first page)** — corresponding author email, target session name ("Biological Molecular Function: Beyond Homology"), originality statement, co-author concurrence statement, LLM-use disclosure.
---
 
## 9. Submission Checklist
 
- [ ] All datasets downloaded and held-out test set locked (untouched during development)
- [ ] Experiments A–F completed with statistical tests
- [ ] Ablations (≥2) completed
- [ ] PyMOL figures finalized (2–3 polished)
- [ ] Source code repository public, documented, with minimal test dataset (good practice even though PSB doesn't mandate it like RECOMB)
- [ ] Paper drafted in PSB template, ≤12 pages excluding cover letter/title/references
- [ ] Cover letter drafted with LLM-use disclosure and session targeting statement
- [ ] Preprint deposited on bioRxiv at time of submission (optional but encouraged by PSB)
- [ ] Submitted via PSB 2027 paper management system (psb.wufoo.com/forms/psb-paper-submission) by **Aug 3, 2026, 11:59 PM PT**
---
 
## 10. Fallback Venues (if PSB timeline slips)
 
- **RECOMB 2027** — Toronto, CFP not yet posted (watch recomb.org); historically abstract ~Nov, full paper ~1 week later; requires public source code at submission; 10-page limit.
- **ISMB/ECCB 2027** — CFP not yet posted (watch iscb.org); proceedings published open-access in *Bioinformatics* (Oxford).
- **Journal tracks (rolling, no fixed deadline):** *Bioinformatics*/*Bioinformatics Advances*, *PLOS Computational Biology*, *NAR Genomics and Bioinformatics*.