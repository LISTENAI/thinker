TOPDIR=../..

#lib or target you want to build: eg. LIB=libmisc.a  or TARGET=wlan
LIB		= libthinker.a
TARGET	= 

#libraries depend: eg. LIBS = -lbsp -ldrv -lrtos
LIBS	=
ifeq ($(IC_BOARD), 0)
LIBS	+= 
else
LIBS	+= 
endif

#exclude subdirs
exclude_dirs=

#files want to build: eg. CSRCS=a.c  CPPSRCS=b.cpp  SSRCS=c.S
CSRCS	= $(wildcard ./c_api/*.c \
			./core/*.c \
			./core/comm/*.c \
			./core/ops/*.c \
			./core/shape_infer/*.c \
			./core/ops/venus/*.c)
#CSRCS	= $(shell find . -name "*.c")
CPPSRCS	=
SSRCS	= 

CFLAGS  = -I ./include \
		  -I ./include/thinker \
		  -I ./include/luna \
		  -I ./include/nnblas \
		  -I ./include/opi_cpy \
          -I ./c_api \
          -I ./core \
          -I ./core/comm \
          -I ./core/ops \
          -I ./core/ops/arcs \
          -I ./core/ops/arcs/luna \
          -I ./core/ops/arcs/nnblas \
          -I ./core/ops/venusa \
          -I ./core/ops/venusa/luna \
          -I ./core/ops/venusa/nnblas \
          -I . \

#-mcoproc
ifeq ($(NNBLAS_ON), 1)
CFLAGS  += -I ./core/ops/arcs/nnblas
else
CFLAGS  += -I ./core/ops/arcs/luna
endif 
CFLAGS  += -I ./core/ops/arcs/opi_cpy
include $(TOPDIR)/rules.mk

