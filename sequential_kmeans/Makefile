CC       := gcc
CSTD     := -std=c11
CFLAGS   := $(CSTD) -O2 -Wall -Wextra -Wpedantic -I.
LDFLAGS  := 

TARGET   := seq_main
SRCS     := seq_main.c seq_kmeans.c file_io.c wtime.c
OBJS     := $(SRCS:.c=.o)
DEPS     := $(OBJS:.o=.d)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) kmeans.h
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c kmeans.h
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)
