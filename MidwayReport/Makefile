TARGET=report

all: 
	pdflatex ${TARGET}.tex
	pdflatex ${TARGET}.tex
	bibtex ${TARGET} 	
clean: 
	$(RM) *.dvi *.log *.aux ${TARGET}.pdf  *.out

