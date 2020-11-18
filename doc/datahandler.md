# `NeuralNetwork`: `dataHandler` library documentation (v0.0.1).
## Contents:
0. [Introduction & general usage](#0-introduction-&-general-usage).
1. [Notation and user-defined types](#1-notation-and-user-defined-types).
2. [Review of the `dataFrame` class](#2-review-of-the-dataFrame-class).
3. [Utility functions](#3-review-of-utility-functions-used-in-the-code).
4. [Who am I + contact info](#4-who-am-i).
5. [To-Do list](#5-to-do-list).

----------

## 0. Introduction & general usage.
The `dataHandler` header/library is a C++ header file that implements data handling methods: mainly, a dataFrame class, read/write to file opertaions and string handling methods.

one-author project ([me!](#2-who-am-i)), aiming at creating a console app to use the `NeuralNetwork` library capabilities without having to hardcode the operations and focusing more on the "what to do" instead of the "how to do it". In this sense, this package is a functional approach to the already existing declarative approach.

### 0.0. How to use `dataHandler`.
To use the `dataHandler` header/library you just have to use the following line in your **C++** main file (or in the file where you want to use the library):
```{c++}
#include "<path-to-header-file>/dataHandler"
```
In this way, you can just compile the program as you were doing before, for example by using
```{sh}
g++ -Wall -std=c++17 -O2 -o main main.cc
```
So, you can go ahead and use the functions and classes from the library without much difficulty.

## 1. Notation and user-defined types.
Throughout the documentation (and the code itself) `vd` is used as a `vector<double>` **C++** identifier. Likewise, `vs` refers to a `vector<string>`, `matd` refers to a `vector<vd>` and `tend` refers to a *3-tensor double* (using tensor as the multidimensional array concept rather than the multilinear map), or a `vector<matd>`. Both `matd` and `tend` have little regard towards dimension consistency, as in the code itself it is often considered that both 2- and 3-dimensional arrays can have slices of different dimensions.

## 2. Review of the `dataFrame` class.

The `dataFrame` class is the only class declared in this header file. It implements the common concept of a dataset or data frame. Although still in development, it supports numerical matrices with named columns. The class can read from a file, write to a file, wrtie data to the console, retrieve its underlying data and the names of each column.

#### 2.0. Constructors.
There is just one constructor for the `dataFrame` class, and it's the empty constructor. The idea here was to create an empty object that could then be used to load data. The constructor is:

```c++
dataFrame() : _name(""), _filepath(""), _data(matd(0)), _colnames(vs(0)),
              _rownames(vs(0)) {}
```

#### 2.1. Setters.
The following member functions can be used to set a `dataFrame` object attributes to a particular given value.
- `void read(string)`: reads the data in the file represented by the `string` parameter and saves it to the data frame data matrix.
- `void setName(string)`: sets the data frame name to the `string` parameter.

#### 2.2. Getters.
The following functions are used to get the value of the data frame attributes.
- `matd getData() const`: it returns the data of the data frame in a matrix form.
- `vs colnames() const`: returns the column names of the dataset.
- `vs rownames() const`: returns the row names of the dataset (dev).

#### 2.4. `void print() const`.
This function prints the data frame to the console standard output, detailing its name and its file path relative to the one where the program is.

#### 2.3. `void write(string) const`.
This function writes the data frame to the file indicated by the `string` parameter.

## 3. Review of utility functions used in the code.

### 3.0. `void trimWhitespace(string &)`.
This function trims the whitespaces at the end and at the beginning of the given string.

### 3.1. `bool isWord(string)`.
Returns `true` if the string starts with a letter, and false otherwise.

### 3.2. `bool isNumber(string)`.
Returns `true` if the string starts with a number, and false otherwise.

### 3.3. `vs tokenize(string &)`.
Returns a vector of strings that correspond to each "word" in the original string, separated by whitespaces. (in dev)

## 4. Who am I.
My name's Àlex. I'm a student at **Universitat Politècnica de Catalunya (UPC-BarcelonaTech)**, currently studying the Bachelors of [Mathematics](https://www.fme.upc.edu/en/studies/degrees/bachelors-degree-in-mathematics-1) and [Data Science and Engineering](https://www.dse.upc.edu) through the double-degree plan in [CFIS](https://www.cfis.upc.edu). I'm from a town called Arenys de Munt, a 40-minute car-drive from Barcelona. If you want to contact me, feel free to do so emailing me at [alex.batlle01@gmail.com](mailto:alex.batlle01@gmail.com?subject=GitHub%20NeuralNetwork%20contact).

## 5. To-Do list.
To-do list in order of priority:

- Write more documentation.
- Fix bugs in the dataFrame.print() method.
