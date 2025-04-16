# common.mk
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Created by Alfred

ifneq ($(filter -j,$(MAKEFLAGS)),)
    # Nothing to do; -j is already set
else
    MAKEFLAGS += -j8
endif

# VITIS related variables
#AIETOOLS_DIR ?= $(shell realpath $(dir $(shell which xchesscc))/../)
AIETOOLS_DIR ?= /tools/ryzen_ai-1.4.0/vitis_aie_essentials
AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/versal_prod/lib
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib

AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body

CHESSCC1_FLAGS = -f -p me -P ${AIE_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}

CHESSCCWRAP1_FLAGS = aie -I ${AIETOOLS_DIR}/include 
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include 
PEANOWRAP2_FLAGS = -O2 -v -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include 
PEANOWRAP2P_FLAGS = -O2 -v -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include 
