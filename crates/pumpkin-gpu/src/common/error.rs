//! 统一错误类型。

use std::fmt;

/// GPU 操作可能失败的所有原因。
#[derive(Debug)]
pub enum DeviceError {
    /// 设备初始化失败
    InitFailed(String),
    /// 设备内存不足
    OutOfMemory {
        /// 请求的字节数
        requested: usize,
        /// 附加详情
        detail: String,
    },
    /// 缓冲区大小不匹配
    SizeMismatch {
        /// 缓冲区容量
        buffer_len: usize,
        /// 请求的数据长度
        data_len: usize,
    },
    /// 设备与主机之间的数据传输失败
    TransferFailed(String),
    /// Kernel 编译或加载失败
    KernelError(String),
    /// Kernel 启动失败
    LaunchFailed(String),
    /// 请求的操作不被当前后端支持
    Unsupported(String),
    /// 内部错误（不应出现）
    Internal(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitFailed(msg) => write!(f, "设备初始化失败: {msg}"),
            Self::OutOfMemory { requested, detail } => {
                write!(f, "设备内存不足: 请求 {requested} 字节, {detail}")
            }
            Self::SizeMismatch {
                buffer_len,
                data_len,
            } => {
                write!(f, "缓冲区大小不匹配: 缓冲区 {buffer_len}, 数据 {data_len}")
            }
            Self::TransferFailed(msg) => write!(f, "数据传输失败: {msg}"),
            Self::KernelError(msg) => write!(f, "Kernel 错误: {msg}"),
            Self::LaunchFailed(msg) => write!(f, "Kernel 启动失败: {msg}"),
            Self::Unsupported(msg) => write!(f, "不支持的操作: {msg}"),
            Self::Internal(msg) => write!(f, "内部错误: {msg}"),
        }
    }
}

impl std::error::Error for DeviceError {}
