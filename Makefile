CXX      =  g++
CFLAGS   = -Wall -O2 -g
LIB      = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d
INCLUDES = 
CXXFLAGS += $(INCLUDES)
OBJ      = main.o RBFinterpolation.o
RM       = rm -f
BIN      = radial
DIRNAME  = $(shell basename $$PWD)
BACKUP   = $(shell date +`basename $$PWD`-%m.%d.%H.%M.tgz)
STDNAME  = $(DIRNAME).tgz

all : $(BIN)

$(BIN) : $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) $(LIB) $(INCLUDES)  -o $(BIN)
	@echo "--------------------------------------------------------------"
	@echo "                 to execute type: ./$(BIN) &"
	@echo "--------------------------------------------------------------"

RBFinterpolation.o : RBFinterpolation.cpp RBFinterpolation.hpp
	@echo "compile RBFinterpolation"
	$(CXX) $(CXXFLAGS) -c $<  
	@echo "done..."

main.o : main.cpp RBFinterpolation.o
	@echo "compile main"
	$(CXX) $(CXXFLAGS) -c $<  
	@echo "done..."

clean :	
	@echo "**************************"
	@echo "CLEAN"
	@echo "**************************"
	$(RM) *~ $(OBJ) $(BIN) 

bigclean :
	@echo "**************************"
	@echo "BIG CLEAN"
	@echo "**************************"
	find . -name '*~' -exec rm -fv {} \;
	$(RM) *~ $(OBJ) $(BIN) output/*

tar : clean 
	@echo "**************************"
	@echo "TAR"
	@echo "**************************"
	cd .. && tar cvfz $(BACKUP) $(DIRNAME)
