// Safe wrappers for BNNS types
use super::ffi::*;
use anyhow::{anyhow, Result};
use std::ffi::c_void;

/// Safe wrapper for BNNS tensor data
pub struct BnnsTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl BnnsTensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    pub fn from_slice(data: &[f32], shape: Vec<usize>) -> Self {
        Self {
            data: data.to_vec(),
            shape,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }

    /// Create BNNS descriptor for this tensor
    pub fn as_descriptor(&self) -> BNNSNDArrayDescriptor {
        let mut size = [0usize; 8];
        for (i, &s) in self.shape.iter().enumerate() {
            size[i] = s;
        }

        BNNSNDArrayDescriptor {
            flags: 0,
            layout: BNNSDataLayout::RowMajorMatrix as u32,
            size,
            stride: [0; 8], // Auto-calculate strides
            data: self.data.as_ptr() as *mut c_void,
            data_type: BNNSDataType::Float32,
            table_data: std::ptr::null_mut(),
            table_data_type: BNNSDataType::Float32,
            data_scale: 1.0,
            data_bias: 0.0,
        }
    }

    /// Create mutable descriptor
    pub fn as_mut_descriptor(&mut self) -> BNNSNDArrayDescriptor {
        let mut size = [0usize; 8];
        for (i, &s) in self.shape.iter().enumerate() {
            size[i] = s;
        }

        BNNSNDArrayDescriptor {
            flags: 0,
            layout: BNNSDataLayout::RowMajorMatrix as u32,
            size,
            stride: [0; 8],
            data: self.data.as_mut_ptr() as *mut c_void,
            data_type: BNNSDataType::Float32,
            table_data: std::ptr::null_mut(),
            table_data_type: BNNSDataType::Float32,
            data_scale: 1.0,
            data_bias: 0.0,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Safe wrapper for BNNS filter (RAII)
pub struct BnnsFilter {
    handle: BNNSFilter,
}

impl BnnsFilter {
    /// Create from raw handle
    ///
    /// # Safety
    /// Handle must be valid from BNNSFilterCreate* function
    pub unsafe fn from_raw(handle: BNNSFilter) -> Result<Self> {
        if handle.is_null() {
            return Err(anyhow!("Failed to create BNNS filter"));
        }
        Ok(Self { handle })
    }

    /// Apply filter to input tensor
    pub fn apply(&self, input: &BnnsTensor, output: &mut BnnsTensor) -> Result<()> {
        let in_desc = input.as_descriptor();
        let out_desc = output.as_mut_descriptor();

        let ret = unsafe {
            BNNSFilterApply(
                self.handle,
                &in_desc as *const _ as *const c_void,
                &out_desc as *const _ as *mut c_void,
            )
        };

        if ret != 0 {
            return Err(anyhow!("BNNSFilterApply failed with code {}", ret));
        }

        Ok(())
    }

    /// Apply filter with batch
    pub fn apply_batch(
        &self,
        batch_size: usize,
        input: &BnnsTensor,
        output: &mut BnnsTensor,
    ) -> Result<()> {
        let in_desc = input.as_descriptor();
        let out_desc = output.as_mut_descriptor();

        let in_stride = input.size() / batch_size;
        let out_stride = output.size() / batch_size;

        let ret = unsafe {
            BNNSFilterApplyBatch(
                self.handle,
                batch_size,
                &in_desc as *const _ as *const c_void,
                in_stride * std::mem::size_of::<f32>(),
                &out_desc as *const _ as *mut c_void,
                out_stride * std::mem::size_of::<f32>(),
            )
        };

        if ret != 0 {
            return Err(anyhow!("BNNSFilterApplyBatch failed with code {}", ret));
        }

        Ok(())
    }
}

impl Drop for BnnsFilter {
    fn drop(&mut self) {
        unsafe {
            BNNSFilterDestroy(self.handle);
        }
    }
}

// Safety: BNNS filters are thread-safe
unsafe impl Send for BnnsFilter {}
unsafe impl Sync for BnnsFilter {}

/// Helper to create empty descriptor
pub fn empty_descriptor() -> BNNSNDArrayDescriptor {
    BNNSNDArrayDescriptor {
        flags: 0,
        layout: BNNSDataLayout::RowMajorMatrix as u32,
        size: [0; 8],
        stride: [0; 8],
        data: std::ptr::null_mut(),
        data_type: BNNSDataType::Float32,
        table_data: std::ptr::null_mut(),
        table_data_type: BNNSDataType::Float32,
        data_scale: 1.0,
        data_bias: 0.0,
    }
}

/// Create descriptor for weights (const data)
pub fn weight_descriptor(data: &[f32], shape: &[usize]) -> BNNSNDArrayDescriptor {
    let mut size = [0usize; 8];
    for (i, &s) in shape.iter().enumerate() {
        size[i] = s;
    }

    BNNSNDArrayDescriptor {
        flags: 0,
        layout: BNNSDataLayout::RowMajorMatrix as u32,
        size,
        stride: [0; 8],
        data: data.as_ptr() as *mut c_void,
        data_type: BNNSDataType::Float32,
        table_data: std::ptr::null_mut(),
        table_data_type: BNNSDataType::Float32,
        data_scale: 1.0,
        data_bias: 0.0,
    }
}
