#Makefile previsto per il progetto PDE
#written by Riccardo Fontanini
OBJECTS=

#compiler
CC=nvcc

#general flags
CFLAGS= 

#libs

ARGS=fontanini,$(D)

NAMEFILE = main.cu
PROGRAM_NAME = pde
LIBS = 


$(PROGRAM_NAME): 
	@mkdir -p build/linux
	$(CC) $(CFLAGS) $(NAMEFILE) -o build/linux/$(PROGRAM_NAME) $(OBJECTS) $(LIBS) -D$(ARGS) 
	@echo " "
	@echo "Compilazione completata! il file si trova nella cartella build/linux!"
	@echo " "