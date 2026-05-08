CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

TARGET = lab1
SRC = src/main.cpp src/timer.cpp src/matrix_dot.cpp src/sum_reduce.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)