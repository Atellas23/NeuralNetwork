all: console

clean:
	rm -f console

console: console_app.cc
	g++ -Wall -o console console_app.cc -O2 -std=c++17

testrun: clean console
	./console
	make clean