all: console

clean:
	rm -f console

console: console_app.cc
	g++ -Wall -o console console_app.cc -O2 -D_GLIBCXX_DEBUG -std=c++17

testrun: clean console
	./console
	make clean

documentation:
	pandoc doc/neural.md -o doc/'Neural Documentation.html'
	pandoc doc/console.md -o doc/'Console Documentation.html'
	pandoc doc/datahandler.md -o doc/'DataHandler Documentation.html'