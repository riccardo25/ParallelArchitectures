#Makefile previsto per il progetto Mandelbrot
#written by Riccardo Fontanini
OBJECTS=

#compiler
CC=gcc
#general flags
CFLAGS= -g -w

#libs
MANDELBROTLIB = build/linux/mandelbrotlib.o
LOGLIB = build/linux/log.o


NAMEFILE = main.c
PROGRAM_NAME = mandelbrot

LIBS = $(MANDELBROTLIB) $(LOGLIB) -std=gnu99 

mandelbrotlib:
	$(CC) $(CFLAGS) -o $(MANDELBROTLIB) -c libs/mandelbrotlib.c -std=gnu99 -pthread


logg:
	$(CC) $(CFLAGS) -o $(LOGLIB) -c libs/log.c -std=gnu99

$(PROGRAM_NAME): logg mandelbrotlib
	@mkdir -p build/linux
	$(CC) $(CFLAGS) $(NAMEFILE) -o build/linux/$(PROGRAM_NAME) $(OBJECTS) $(LIBS) -pthread
	@echo " "
	@echo "Compilazione completata! il file si trova nella cartella build/linux!"
	@echo " "