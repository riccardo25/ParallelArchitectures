#Makefile previsto per il progetto Mandelbrot
#written by Riccardo Fontanini
OBJECTS=

#compiler
CC=gcc

#general flags
CFLAGS= 

#libs

ARGS=$(D)

NAMEFILE =main.c
PROGRAM_NAME =orand

LIBS =-std=c11


$(PROGRAM_NAME): 
	@mkdir -p build/linux
	$(CC) $(ARGS) $(CFLAGS) $(NAMEFILE) -o build/linux/$(PROGRAM_NAME) $(OBJECTS)   $(LIBS)
	@echo " "
	@echo "Compilazione completata! il file si trova nella cartella build/linux!"
	@echo " "