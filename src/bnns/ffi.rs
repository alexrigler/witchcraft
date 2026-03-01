// Raw FFI bindings to BNNS C API
#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::c_void;

// === Data Types ===

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BNNSDataType {
    Float16 = 1,
    Float32 = 2,
    Float64 = 3,
    Int8 = 257,
    Int16 = 258,
    Int32 = 259,
    Int64 = 260,
    UInt8 = 513,
    UInt16 = 514,
    UInt32 = 515,
    UInt64 = 516,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BNNSDataLayout {
    RowMajorMatrix = 0x10000,
    ColumnMajorMatrix = 0x10001,
}

// === Activation Functions ===

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BNNSActivationFunction {
    Identity = 0,
    Relu = 1,
    Leaky = 2,
    Sigmoid = 3,
    Tanh = 4,
    ScaledTanh = 5,
    Abs = 6,
    Linear = 7,
    Clamp = 8,
    IntegerLinearSaturate = 9,
    IntegerLinearSaturatePerChannel = 10,
    Softmax = 11,
    GeluApproximation = 12,
    Relu6 = 13,
    GeluApproximation2 = 14,
    Silu = 16,
    LogSigmoid = 22,
    Gelu = 31,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BNNSActivation {
    pub function: BNNSActivationFunction,
    pub alpha: f32,
    pub beta: f32,
}

// === Tensor Descriptors ===

#[repr(C)]
#[derive(Debug, Clone)]
pub struct BNNSNDArrayDescriptor {
    pub flags: u32,
    pub layout: u32,
    pub size: [usize; 8],
    pub stride: [isize; 8],
    pub data: *mut c_void,
    pub data_type: BNNSDataType,
    pub table_data: *mut c_void,
    pub table_data_type: BNNSDataType,
    pub data_scale: f32,
    pub data_bias: f32,
}

// === Filter Handle ===

pub type BNNSFilter = *mut c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BNNSFilterParameters {
    pub flags: u32,
    pub n_threads: usize,
    pub alloc: *mut c_void,
    pub free: *mut c_void,
}

// === Layer Parameters ===

#[repr(C)]
#[derive(Debug)]
pub struct BNNSLayerParametersFullyConnected {
    pub in_desc: BNNSNDArrayDescriptor,
    pub out_desc: BNNSNDArrayDescriptor,
    pub weights: BNNSNDArrayDescriptor,
    pub bias: BNNSNDArrayDescriptor,
    pub activation: BNNSActivation,
}

#[repr(C)]
#[derive(Debug)]
pub struct BNNSLayerParametersActivation {
    pub in_desc: BNNSNDArrayDescriptor,
    pub out_desc: BNNSNDArrayDescriptor,
    pub activation: BNNSActivation,
    pub axis_flags: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct BNNSLayerParametersNormalization {
    pub in_desc: BNNSNDArrayDescriptor,
    pub out_desc: BNNSNDArrayDescriptor,
    pub beta: BNNSNDArrayDescriptor,
    pub gamma: BNNSNDArrayDescriptor,
    pub moving_mean: BNNSNDArrayDescriptor,
    pub moving_variance: BNNSNDArrayDescriptor,
    pub momentum: f32,
    pub epsilon: f32,
    pub activation: BNNSActivation,
    pub normalization_axis: usize,
}

// Multihead attention projection parameters
#[repr(C)]
#[derive(Debug)]
pub struct BNNSLayerParametersMHAProjection {
    pub qkv_desc: BNNSNDArrayDescriptor,
    pub o_desc: BNNSNDArrayDescriptor,
    pub q_weights: BNNSNDArrayDescriptor,
    pub q_bias: BNNSNDArrayDescriptor,
    pub k_weights: BNNSNDArrayDescriptor,
    pub k_bias: BNNSNDArrayDescriptor,
    pub v_weights: BNNSNDArrayDescriptor,
    pub v_bias: BNNSNDArrayDescriptor,
    pub o_weights: BNNSNDArrayDescriptor,
    pub o_bias: BNNSNDArrayDescriptor,
}

#[repr(C)]
#[derive(Debug)]
pub struct BNNSLayerParametersMultiheadAttention {
    pub query: BNNSNDArrayDescriptor,
    pub key: BNNSNDArrayDescriptor,
    pub value: BNNSNDArrayDescriptor,
    pub output: BNNSNDArrayDescriptor,
    pub num_heads: usize,
    pub add_zero_attn: bool,
    pub key_attn_bias: BNNSNDArrayDescriptor,
    pub value_attn_bias: BNNSNDArrayDescriptor,
    pub dropout: f32,
    pub projection: BNNSLayerParametersMHAProjection,
}

#[repr(C)]
#[derive(Debug)]
pub struct BNNSLayerParametersArithmetic {
    pub in_desc: BNNSNDArrayDescriptor,
    pub out_desc: BNNSNDArrayDescriptor,
    pub operation: u32, // BNNSArithmeticOperation
}

// === Core BNNS Functions ===

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    // Filter creation - Layer API (deprecated in macOS 15 but still works)
    pub fn BNNSFilterCreateLayerFullyConnected(
        layer_params: *const BNNSLayerParametersFullyConnected,
        filter_params: *const BNNSFilterParameters,
    ) -> BNNSFilter;

    pub fn BNNSFilterCreateLayerActivation(
        layer_params: *const BNNSLayerParametersActivation,
        filter_params: *const BNNSFilterParameters,
    ) -> BNNSFilter;

    pub fn BNNSFilterCreateLayerNormalization(
        layer_params: *const BNNSLayerParametersNormalization,
        filter_params: *const BNNSFilterParameters,
    ) -> BNNSFilter;

    pub fn BNNSFilterCreateLayerMultiheadAttention(
        layer_params: *const BNNSLayerParametersMultiheadAttention,
        filter_params: *const BNNSFilterParameters,
    ) -> BNNSFilter;

    pub fn BNNSFilterCreateLayerArithmetic(
        layer_params: *const BNNSLayerParametersArithmetic,
        filter_params: *const BNNSFilterParameters,
    ) -> BNNSFilter;

    // Filter application
    pub fn BNNSFilterApply(
        filter: BNNSFilter,
        in_data: *const c_void,
        out_data: *mut c_void,
    ) -> i32;

    pub fn BNNSFilterApplyBatch(
        filter: BNNSFilter,
        batch_size: usize,
        in_data: *const c_void,
        in_stride: usize,
        out_data: *mut c_void,
        out_stride: usize,
    ) -> i32;

    // Cleanup
    pub fn BNNSFilterDestroy(filter: BNNSFilter);
}

// === Helper Functions ===

impl Default for BNNSActivation {
    fn default() -> Self {
        Self {
            function: BNNSActivationFunction::Identity,
            alpha: 0.0,
            beta: 0.0,
        }
    }
}

impl BNNSActivation {
    pub fn gelu() -> Self {
        Self {
            function: BNNSActivationFunction::Gelu,
            alpha: 0.0,
            beta: 0.0,
        }
    }

    pub fn relu() -> Self {
        Self {
            function: BNNSActivationFunction::Relu,
            alpha: 0.0,
            beta: 0.0,
        }
    }

    pub fn identity() -> Self {
        Self::default()
    }
}

impl Default for BNNSFilterParameters {
    fn default() -> Self {
        Self {
            flags: 0,
            n_threads: 0, // 0 = auto
            alloc: std::ptr::null_mut(),
            free: std::ptr::null_mut(),
        }
    }
}
