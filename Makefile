CC = gcc
CPPC = g++
LDLIBS = -lgsl -lgslcblas -lm -lcjson
DCFLAGS = -ggdb

FILE_BASE_NAME = main

FILES = $(FILE_BASE_NAME) utils mnist network network_utils config 
C_FILE = $(FILE_BASE_NAME).c
O_FILE = $(FILE_BASE_NAME).o

# O_FILES = $(O_FILE) utils.o mnist.o network.o network_utils.o config.o file_utils.o
O_FILES = $(addsuffix .o, $(FILES))
C_FILES = $(addsuffix .c, $(FILES))
H_FILES = $(addprefix ./include/, $(addsuffix .h, $(FILES))))

%.o: %.c
	$(CC) -c $< -o $@ -g

%.o: %.c include/%.h
	$(CC) -c $< -o $@ -g 

%.o: %.cpp include/%.h
	$(CPPC) -c $< -o $@ -g

$(FILE_BASE_NAME): $(O_FILES)
	$(CC) -o $(FILE_BASE_NAME) $(O_FILES) $(LDLIBS)

.PHONY: build run default gdb valgrind clean rebuild
build: $(FILE_BASE_NAME)

run: build
	./$(FILE_BASE_NAME) $(INPUT)

default: build
	./$(FILE_BASE_NAME) -c config.json -s
	./processing

gdb: build
	gdb ./$(FILE_BASE_NAME)

valgrind: build
	valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all ./$(FILE_BASE_NAME) $(INPUT)

clean:
	rm -f $(FILE_BASE_NAME) $(O_FILES)

rebuild: clean build