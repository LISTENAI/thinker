TOPDIR=../..

THINKER_TARGET_PLATFORM ?= VENUSA
THINKER_USE_MOSS ?= 1
CLEAN_EXTRA += ./c_api/thinker_moss_host*.c ./c_api/thinker_moss_register.c

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
CSRCS	= $(wildcard ./core/*.c \
			./core/comm/*.c \
			./core/ops/*.c \
			./core/shape_infer/*.c)
#CSRCS	= $(shell find . -name "*.c")
CPPSRCS	=
SSRCS	=

ifeq ($(THINKER_USE_MOSS),1)
CSRCS	+= ./c_api/thinker_api_moss.c
else
CSRCS	+= ./c_api/thinker_api.c
endif

ifneq ($(filter venus VENUS,$(THINKER_TARGET_PLATFORM)),)
CSRCS	+= $(wildcard ./core/ops/venus/*.c)
endif

ifneq ($(filter venusa VENUSA VENUSA_CPU,$(THINKER_TARGET_PLATFORM)),)
CSRCS	+= $(wildcard ./core/ops/venusA/*.c)
endif

ifneq ($(filter arcs ARCS,$(THINKER_TARGET_PLATFORM)),)
CSRCS	+= $(wildcard ./core/ops/arcs/*.c)
endif

CFLAGS  = -I ./include \
		  -I ./include/thinker \
		  -I ./include/luna \
		  -I ./include/nnblas \
		  -I ./include/opi_cpy \
          -I ./c_api \
          -I ./core \
          -I ./core/comm \
          -I ./core/ops \
          -I . \

ifneq ($(filter venus VENUS,$(THINKER_TARGET_PLATFORM)),)
CFLAGS	+= -DTHINKER_USE_VENUS \
		   -I ./core/ops/venus \
		   -I ./core/ops/venus/luna \
		   -I ./core/ops/venus/nnblas \
		   -I ./core/ops/venus/hifi
endif

ifneq ($(filter venusa VENUSA VENUSA_CPU,$(THINKER_TARGET_PLATFORM)),)
CFLAGS	+= -DTHINKER_USE_VENUSA \
		   -I ./core/ops/venusA \
		   -I ./core/ops/venusA/luna
endif

ifneq ($(filter arcs ARCS,$(THINKER_TARGET_PLATFORM)),)
CFLAGS	+= -DTHINKER_USE_ARCS \
		   -I ./core/ops/arcs \
		   -I ./core/ops/arcs/luna \
		   -I ./core/ops/arcs/nnblas \
		   -I ./core/ops/arcs/opi_cpy
endif

ifeq ($(THINKER_USE_MOSS),1)
CFLAGS	+= -DTHINKER_USE_MOSS=1 \
		   -I ./include/moss

comma := ,
empty :=
space := $(empty) $(empty)
THINKER_MOSS_RESOURCE_DIR ?= moss_res

ifneq ($(strip $(THINKER_MOSS_HOST_SOURCES)),)
THINKER_MOSS_HOST_SOURCE_LIST := $(subst $(comma),$(space),$(strip $(THINKER_MOSS_HOST_SOURCES)))
THINKER_MOSS_MODEL_GETTER_LIST := $(subst $(comma),$(space),$(strip $(THINKER_MOSS_MODEL_GETTERS)))
THINKER_MOSS_MODEL_NAME_LIST := $(subst $(comma),$(space),$(strip $(THINKER_MOSS_MODEL_NAMES)))
else
ifneq ($(strip $(THINKER_MOSS_MODELS)),)
THINKER_MOSS_MODEL_LIST := $(subst $(comma),$(space),$(strip $(THINKER_MOSS_MODELS)))
THINKER_MOSS_HOST_SOURCE_LIST := $(foreach model,$(THINKER_MOSS_MODEL_LIST),$(THINKER_MOSS_RESOURCE_DIR)/$(model)_host.c)
THINKER_MOSS_MODEL_GETTER_LIST := $(foreach model,$(THINKER_MOSS_MODEL_LIST),mGetModel_$(model))
THINKER_MOSS_MODEL_NAME_LIST := $(THINKER_MOSS_MODEL_LIST)
else
THINKER_MOSS_HOST_SOURCE_LIST := $(strip $(THINKER_MOSS_HOST_SOURCE))
THINKER_MOSS_MODEL_GETTER_LIST := $(strip $(THINKER_MOSS_MODEL_GETTER))
THINKER_MOSS_MODEL_NAME_LIST := $(strip $(THINKER_MOSS_MODEL_NAME))
endif
endif

ifeq ($(strip $(THINKER_MOSS_MODEL_NAME_LIST)),)
THINKER_MOSS_MODEL_NAME_LIST := $(THINKER_MOSS_MODEL_GETTER_LIST)
endif

THINKER_MOSS_REGISTRY_SOURCE ?= ./c_api/thinker_moss_register.c
THINKER_MOSS_HOST_INDEXES := 1 2 3 4 5 6 7 8

ifneq ($(strip $(THINKER_MOSS_HOST_SOURCE_LIST)),)
ifeq ($(strip $(THINKER_MOSS_MODEL_GETTER_LIST)),)
$(error THINKER_MOSS_MODEL_GETTER(S) must be set when THINKER_MOSS_HOST_SOURCE(S) is used)
endif

ifneq ($(words $(THINKER_MOSS_HOST_SOURCE_LIST)),$(words $(THINKER_MOSS_MODEL_GETTER_LIST)))
$(error THINKER_MOSS_HOST_SOURCES and THINKER_MOSS_MODEL_GETTERS count mismatch)
endif
ifneq ($(words $(THINKER_MOSS_HOST_SOURCE_LIST)),$(words $(THINKER_MOSS_MODEL_NAME_LIST)))
$(error THINKER_MOSS_HOST_SOURCES and THINKER_MOSS_MODEL_NAMES count mismatch)
endif

THINKER_MOSS_HOST_SOURCE_PATHS := $(foreach src,$(THINKER_MOSS_HOST_SOURCE_LIST),$(if $(wildcard $(src)),$(src),$(TOPDIR)/$(src)))
$(foreach src,$(THINKER_MOSS_HOST_SOURCE_PATHS),$(if $(wildcard $(src)),,$(error THINKER_MOSS_HOST_SOURCE not found: $(src))))

THINKER_MOSS_ACTIVE_INDEXES := $(wordlist 1,$(words $(THINKER_MOSS_HOST_SOURCE_LIST)),$(THINKER_MOSS_HOST_INDEXES))
THINKER_MOSS_HOST_WRAPPER_SOURCES := $(foreach getter,$(THINKER_MOSS_MODEL_GETTER_LIST),./c_api/thinker_moss_host_$(getter).c)

CSRCS	+= $(THINKER_MOSS_HOST_WRAPPER_SOURCES) $(THINKER_MOSS_REGISTRY_SOURCE)
CFLAGS	+= $(foreach src,$(THINKER_MOSS_HOST_SOURCE_PATHS),-I $(dir $(src))) -DTHINKER_MOSS_REQUIRE_REGISTRY

define THINKER_MOSS_HOST_RULE
./c_api/thinker_moss_host_$(2).c: $(1) FORCE
	@mkdir -p $$(dir $$@)
	@printf '#include "%s"\n' '$$(notdir $(1))' > $$@
endef
$(foreach idx,$(THINKER_MOSS_ACTIVE_INDEXES),$(eval $(call THINKER_MOSS_HOST_RULE,$(word $(idx),$(THINKER_MOSS_HOST_SOURCE_PATHS)),$(word $(idx),$(THINKER_MOSS_MODEL_GETTER_LIST)))))

$(THINKER_MOSS_REGISTRY_SOURCE): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' \
		'#include <string.h>' \
		'' \
		$(foreach getter,$(THINKER_MOSS_MODEL_GETTER_LIST),'extern void *$(getter)(void);') \
		'' \
		'void thinker_moss_registry_link_anchor(void) {}' \
		'' \
		'void *thinker_moss_get_registered_model(const char *getter_name, const char *model_name) {' \
		$(foreach idx,$(THINKER_MOSS_ACTIVE_INDEXES),'  if ((getter_name != 0 && strcmp(getter_name, "$(word $(idx),$(THINKER_MOSS_MODEL_GETTER_LIST))") == 0) ||' '      (model_name != 0 && strcmp(model_name, "$(word $(idx),$(THINKER_MOSS_MODEL_NAME_LIST))") == 0)) {' '    return $(word $(idx),$(THINKER_MOSS_MODEL_GETTER_LIST))();' '  }') \
		'  return 0;' \
		'}' > $@
endif
endif
ifeq ($(THINKER_USE_MOSS),1)
ifneq ($(filter venusa VENUSA VENUSA_CPU,$(THINKER_TARGET_PLATFORM)),)
THINKER_MOSS_PREBUILT_LIBS := ./libs/venusa/chip/libmossruntimeva.a ./libs/venusa/chip/libnnblas.a ./libs/venusa/chip/libvenusadriver.a

ext:
	@for lib in $(THINKER_MOSS_PREBUILT_LIBS); do \
		if [ ! -f "$$lib" ]; then echo "Missing MOSS dependency $$lib"; exit 1; fi; \
		cp "$$lib" $(LIBOUT)/; \
	done
endif
endif

#-mcoproc
ifeq ($(NNBLAS_ON), 1)
CFLAGS  += -I ./core/ops/arcs/nnblas
else
CFLAGS  += -I ./core/ops/arcs/luna
endif
CFLAGS  += -I ./core/ops/arcs/opi_cpy

FORCE:

include $(TOPDIR)/rules.mk
