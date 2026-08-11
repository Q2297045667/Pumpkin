//! CUDA 上下文初始化和设备探测。

use std::sync::Arc;

/// 初始化 CUDA 驱动并获取第一个可用设备。
pub fn init_cuda() -> Result<Arc<cudarc::driver::CudaContext>, String> {
    let result = cudarc::driver::result::init();
    match result {
        Ok(()) => {
            tracing::debug!("CUDA 驱动初始化成功");
        }
        Err(e) => {
            return Err(format!("CUDA 驱动初始化失败: {e:?}"));
        }
    }

    // cudarc 0.19: CudaContext::new returns Arc<CudaContext>
    let ctx =
        cudarc::driver::CudaContext::new(0).map_err(|e| format!("无法创建 CUDA 上下文: {e:?}"))?;

    Ok(ctx)
}
