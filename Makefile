PYTHON ?= python3

.PHONY: all figures paper check clean

all: figures paper

figures:
	$(PYTHON) scripts/make_figures.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error finite_N_PTA.tex

check:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	cd paper && latexmk -c finite_N_PTA.tex
