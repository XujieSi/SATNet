
CC=/usr/bin/gcc
# GCC_LIB_INCLUDE=/usr/lib/gcc/x86_64-linux-gnu/11/include 
# CC_FLAGS = -I ${GCC_LIB_INCLUDE} -I .

learn.exe: learn.o satnet_cpu.o
	${CC} -g -o $@ $^ -lm -lstdc++

test.exe: test.o satnet_cpu.o
	${CC} -o $@ $^ -lm -lstdc++

%.o: %.cpp satnet.h
	${CC} -g -c $<   -msse4.1

clear:
	rm *.o *.exe