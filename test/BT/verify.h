/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Bezier Tessellation
 * Version using dynamic parallelism
 *
 */

#include "common.h"

void verify(int N_LINES, BezierLine *bLines_check_d,  BezierLine *bLines_h);


