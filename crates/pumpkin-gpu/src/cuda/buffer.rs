//! CUDA 缓冲区管理（存根）。
//!
//! 当前版本使用 CPU fallback 进行缓冲区操作。
//! 完整 GPU 内存管理将在后续迭代中实现。

use crate::common::DeviceError;
use crate::common::GpuBuffer;
use cudarc::driver::CudaContext;
use std::sync::Arc;

// 当前使用 CudaBackend 内的直接实现，此文件保留为接口占位。
