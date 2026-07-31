# PSB 2027 paper build
TEX      = xqdrant_psb2027
BIB      = xqdrant
TEMPLATE = template
PS       = ../plot_scripts/outputs

.PHONY: all clean pagecount figures plots

all: $(TEX).pdf

plots:
	@cd .. && ./plot_scripts/run_all.sh

figures:
	@mkdir -p figures
	@cp $(PS)/exp_a_retrieval_recall.pdf \
	      $(PS)/exp_b_v3_violin_spearman.pdf \
	      $(PS)/exp_b_v3_label_perm.pdf \
	      $(PS)/exp_b_v3_ablation_topn.pdf \
	      $(PS)/exp_c_roc.pdf \
	      $(PS)/exp_d_scatter_attribution_vs_delta.pdf \
	      $(PS)/exp_d_forest_significance.pdf \
	      $(PS)/pipeline_schematic.pdf \
	      $(PS)/exp_f_workflow_time.png \
	      $(PS)/exp_ablation_layer_pooling.png \
	      ../plot_scripts/outputs/pymol/exp_d_groel_rsa.png \
	      ../plot_scripts/outputs/pymol/exp_d_groel_charged.png \
	      $(PS)/exp_e_site_enrichment.pdf \
	      ../plot_scripts/outputs/pymol/exp_e_active_site.png \
	      figures/ 2>/dev/null || true
	@# Exp E attribution panel (first matching png)
	@for f in $(PS)/exp_e_*_attribution.png; do \
	  if [ -f "$$f" ]; then cp "$$f" figures/exp_e_attribution.png; break; fi; \
	done

$(TEX).pdf: $(TEX).tex cover_letter.tex $(BIB).bib $(TEMPLATE)/ws-procs11x85.cls figures
	@cp $(TEMPLATE)/ws-procs11x85.cls $(TEMPLATE)/ws-procs-thm.sty $(TEMPLATE)/ws-procs11x85.bst \
	      $(TEMPLATE)/ws-rotating.sty $(TEMPLATE)/cite.sty chapterbib.sty infwarerr.sty . 2>/dev/null || true
	pdflatex -interaction=nonstopmode $(TEX)
	bibtex $(TEX) || true
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)
	@echo "Built $(TEX).pdf"

pagecount: $(TEX).pdf
	@pdfinfo $(TEX).pdf 2>/dev/null | grep Pages || echo "Install pdfinfo (poppler) for page count"
	@echo "PSB limit: 12 pages EXCLUDING cover letter, title page, references."

clean:
	rm -f *.aux *.log *.bbl *.blg *.out $(TEX).pdf
	rm -f ws-procs11x85.cls ws-procs-thm.sty ws-procs11x85.bst ws-rotating.sty cite.sty
