#Makefile previsto per il progetto Mandelbrot
#written by Riccardo Fontanini
OBJECTS=

#compiler
CC=nvcc
#general flags
CFLAGS=

#libs
MANDELBROTLIB = build/linux/mandelbrotlib.o


NAMEFILE = main.cu
PROGRAM_NAME = mandelbrot

LIBS = $(MANDELBROTLIB)

mandelbrotlib:
	$(CC) $(CFLAGS) -o $(MANDELBROTLIB) -c libs/mandelbrotlib.cu


$(PROGRAM_NAME): mandelbrotlib
	@mkdir -p build/linux
	$(CC) $(CFLAGS) $(NAMEFILE) -o build/linux/$(PROGRAM_NAME) $(OBJECTS) $(LIBS)
	@echo " "
	@echo "Compilazione completata! il file si trova nella cartella build/linux!"
	@echo " "