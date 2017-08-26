COMPILER = g++
CFLAGS   = -Wall -O2 -MMD -MP
LDFLAG   = 
LIBS     = 
INCLUDE  = -I./include
TARGET   = ./bin/my_actor_critic_test
OBJDIR   = ./build
SRCDIR   = ./src
SOURCES  = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS  = $(addprefix $(OBJDIR)/, $(notdir $(SOURCES:.cpp=.o)))
DEPENDS   = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(COMPILER) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-mkdir -p $(OBJDIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: clean $(TARGET)

clean:
	-rm -f $(OBJECTS) $(DEPENDS) $(TARGET)

-include $(DEPENDS)

