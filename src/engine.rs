use crate::elem_arithmetic::ElementwiseArithmetic;
use crate::matrix::Matrix;
use crate::matrix_multiply::MatrixMultiply;
use crate::scalar_ops::ScalarOps;
use crate::sigmoid::Sigmoid;
use crate::Operation;
use anyhow::{bail, ensure, Context, Result};
use erupt::{vk1_0 as vk, DeviceLoader};
use genmap::{GenMap, Handle};
use std::collections::HashSet;
use vk_core::{Core, SharedCore};

/// The TensorSludge engine
pub struct TensorSludge {
    command_pool: vk::CommandPool,
    transfer_command_buffer: vk::CommandBuffer,
    passes: GenMap<Pass>,
    matrices: GenMap<Matrix>,
    sigmoid: Sigmoid,
    matrix_multiply: MatrixMultiply,
    elem_arithmetic: ElementwiseArithmetic,
    scalar_ops: ScalarOps,
    core: SharedCore,
}

impl TensorSludge {
    /// Create a new TensorSludge instance
    pub fn new() -> Result<Self> {
        let (core, core_meta) = Core::compute(cfg!(debug_assertions), "TensorSludge")?;

        // Create command buffer
        // Command pool:
        let create_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(core_meta.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool =
            unsafe { core.device.create_command_pool(&create_info, None, None) }.result()?;

        // Create ops
        let sigmoid = Sigmoid::new(core.clone())?;
        let matrix_multiply = MatrixMultiply::new(core.clone())?;
        let elem_arithmetic = ElementwiseArithmetic::new(core.clone())?;
        let scalar_ops = ScalarOps::new(core.clone())?;

        // Create transfer command buffer
        let allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let transfer_command_buffer =
            unsafe { core.device.allocate_command_buffers(&allocate_info) }.result()?[0];

        Ok(Self {
            transfer_command_buffer,
            command_pool,
            matrix_multiply,
            elem_arithmetic,
            scalar_ops,
            matrices: GenMap::with_capacity(10),
            passes: GenMap::with_capacity(10),
            sigmoid,
            core,
        })
    }

    /// Create a new matrix with the specified dimensions
    pub fn matrix(
        &mut self,
        rows: usize,
        cols: usize,
        layers: usize,
        cpu_visible: bool,
        name: impl Into<String>,
    ) -> Result<crate::Matrix> {
        Ok(crate::Matrix(self.matrices.insert(Matrix::new(
            rows,
            cols,
            layers,
            cpu_visible,
            name,
            self.core.clone(),
        )?)))
    }

    fn get_matrix_mut(&mut self, matrix: crate::Matrix) -> Result<&mut Matrix> {
        self.matrices
            .get_mut(matrix.0)
            .context("Matrix was deleted")
    }

    fn get_matrix(&self, matrix: crate::Matrix) -> Result<&Matrix> {
        self.matrices.get(matrix.0).context("Matrix was deleted")
    }

    /// Transfer data between matrices of equal sizes
    pub fn transfer(&mut self, src: crate::Matrix, dst: crate::Matrix) -> Result<()> {
        let src = self.get_matrix(src).context("SRC was deleted")?;
        let dst = self.get_matrix(dst).context("DST was deleted")?;
        ensure!(
            src.size() == dst.size(),
            "Source and destination sizes must match exactly for transfer"
        );
        let size = src.size_bytes();
        let src = src.buffer();
        let dst = dst.buffer();

        let region = vk::BufferCopyBuilder::new()
            .src_offset(0)
            .dst_offset(0)
            .size(size as _);

        unsafe {
            self.core
                .device
                .reset_command_buffer(self.transfer_command_buffer, None)
                .result()?;
            let begin_info = vk::CommandBufferBeginInfoBuilder::new();
            self.core
                .device
                .begin_command_buffer(self.transfer_command_buffer, &begin_info)
                .result()?;
            self.core
                .device
                .cmd_copy_buffer(self.transfer_command_buffer, src, dst, &[region]);
            self.core
                .device
                .end_command_buffer(self.transfer_command_buffer)
                .result()?;
            let command_buffers = [self.transfer_command_buffer];
            let submit_info = vk::SubmitInfoBuilder::new().command_buffers(&command_buffers);
            self.core
                .device
                .queue_submit(self.core.queue, &[submit_info], None)
                .result()?;
            // TODO: Use a fence here!!
            self.core.device.queue_wait_idle(self.core.queue).result()?;
        }

        Ok(())
    }

    pub fn remove_matrix(&mut self, mat: crate::Matrix) -> Result<()> {
        Ok(drop(self.matrices.remove(mat.0)))
    }

    /// Write data to a matrix in row-major order
    pub fn write(&mut self, matrix: crate::Matrix, data: &[f32]) -> Result<()> {
        self.get_matrix_mut(matrix)?.write(data)
    }

    /// Read data from a matrix in row-major order
    pub fn read(&mut self, matrix: crate::Matrix, data: &mut [f32]) -> Result<()> {
        self.get_matrix_mut(matrix)?.read(data)
    }

    /// Create a pass from a sequence of operations
    pub fn create_pass(&mut self, ops: &[crate::Operation]) -> Result<crate::Pass> {
        // Allocate command buffer
        let allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer =
            unsafe { self.core.device.allocate_command_buffers(&allocate_info) }.result()?[0];

        // Create dispatch information and required matrix list
        // This must be a seperate loop because the descriptor set update happens in invoke() for
        // each type of operation
        let mut required_mats = Vec::new();
        let mut dispatch_list: Vec<(Box<dyn Invocation>, Vec<BufferAction>)> = Vec::new();
        for op in ops {
            match op {
                Operation::ScalarMultiply(mat, scalar) => {
                    required_mats.push(mat.0);
                    let matrix = self.matrices.get(mat.0).context("Matrix was deleted")?;
                    let invocation = self.scalar_ops.invoke(matrix, *scalar)?;
                    let actions = vec![BufferAction {
                        matrix: mat.0,
                        read: true,
                        write: true,
                    }];
                    dispatch_list.push((Box::new(invocation), actions));
                }
                Operation::Sigmoid(mat) | Operation::SigmoidDerivative(mat) => {
                    required_mats.push(mat.0);
                    let matrix = self.matrices.get(mat.0).context("Matrix was deleted")?;
                    let invocation = match op {
                        Operation::Sigmoid(_) => self.sigmoid.invoke(matrix),
                        Operation::SigmoidDerivative(_) => self.sigmoid.invoke_deriv(matrix),
                        _ => unreachable!(),
                    }?;
                    let actions = vec![BufferAction {
                        matrix: mat.0,
                        read: true,
                        write: true,
                    }];
                    dispatch_list.push((Box::new(invocation), actions));
                }
                Operation::InplaceAdd(product, scalars)
                | Operation::InplaceSub(product, scalars)
                | Operation::InplaceMultiply(product, scalars) => {
                    required_mats.push(product.0);
                    required_mats.push(scalars.0);
                    let p = self
                        .matrices
                        .get(product.0)
                        .context("Product matrix was deleted")?;
                    let s = self
                        .matrices
                        .get(scalars.0)
                        .context("Scalars matrix was deleted")?;
                    let invocation = match op {
                        Operation::InplaceMultiply(_, _) => {
                            self.elem_arithmetic.invoke_mult(p, s)?
                        }
                        Operation::InplaceAdd(_, _) => self.elem_arithmetic.invoke_add(p, s)?,
                        Operation::InplaceSub(_, _) => self.elem_arithmetic.invoke_sub(p, s)?,
                        _ => unreachable!(),
                    };
                    let actions = vec![
                        BufferAction {
                            matrix: product.0,
                            read: true,
                            write: true,
                        },
                        BufferAction {
                            matrix: scalars.0,
                            read: true,
                            write: false,
                        },
                    ];
                    dispatch_list.push((Box::new(invocation), actions));
                }
                Operation::MatrixMultiply {
                    left,
                    right,
                    left_transpose,
                    right_transpose,
                    dst,
                } => {
                    required_mats.push(left.0);
                    required_mats.push(right.0);
                    required_mats.push(dst.0);
                    let invocation = self.matrix_multiply.invoke(
                        self.matrices.get(left.0).context("Left mat was deleted")?,
                        *left_transpose,
                        self.matrices
                            .get(right.0)
                            .context("Right mat was deleted")?,
                        *right_transpose,
                        self.matrices
                            .get(dst.0)
                            .context("Destination mat was deleted")?,
                    )?;
                    let actions = vec![
                        BufferAction {
                            matrix: left.0,
                            read: true,
                            write: false,
                        },
                        BufferAction {
                            matrix: right.0,
                            read: true,
                            write: false,
                        },
                        BufferAction {
                            matrix: dst.0,
                            read: false,
                            write: true,
                        },
                    ];
                    dispatch_list.push((Box::new(invocation), actions));
                }
            }
        }

        // Write command buffer
        let mut dirty_list = HashSet::new(); // Naughy buffers!
        unsafe {
            let begin_info = vk::CommandBufferBeginInfoBuilder::new();
            self.core
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .result()?;

            for (invocation, actions) in dispatch_list {
                let mut barriers = Vec::new();
                for action in actions {
                    if (action.read || action.write) && dirty_list.remove(&action.matrix) {
                        let buffer = self.matrices.get(action.matrix).unwrap();
                        let dst_flags = match (action.read, action.write) {
                            (true, false) => vk::AccessFlags::SHADER_READ,
                            (true, true) => {
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
                            }
                            (false, true) => vk::AccessFlags::SHADER_WRITE,
                            (false, false) => unreachable!(),
                        };
                        let buf_mem_barrier = vk::BufferMemoryBarrierBuilder::new()
                            .buffer(buffer.buffer())
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(dst_flags)
                            .offset(0)
                            .size(vk::WHOLE_SIZE);

                        barriers.push(buf_mem_barrier);
                    }

                    if action.write {
                        dirty_list.insert(action.matrix);
                    }
                }

                self.core.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    None,
                    &[],
                    &barriers,
                    &[],
                );

                invocation.dispatch(&self.core.device, command_buffer);
            }

            self.core
                .device
                .end_command_buffer(command_buffer)
                .result()?;
        }

        let pass = Pass {
            command_buffer,
            required_mats,
        };

        Ok(crate::Pass(self.passes.insert(pass)))
    }

    /// Run the specified pass on the TensorSludge engine
    pub fn flow(&mut self, pass: crate::Pass) -> Result<()> {
        let pass = self.passes.get(pass.0).context("Pass was deleted")?;
        for matrix in &pass.required_mats {
            if self.matrices.get(*matrix).is_none() {
                bail!("A matrix used in this pass was deleted since its creation");
            }
        }

        unsafe {
            let command_buffers = [pass.command_buffer];
            let submit_info = vk::SubmitInfoBuilder::new().command_buffers(&command_buffers);
            self.core
                .device
                .queue_submit(self.core.queue, &[submit_info], None)
                .result()?;
            // TODO: SWITCH THIS TO A FENCE! So you can interop with other programs more smoothly..
            self.core.device.queue_wait_idle(self.core.queue).result()?;
        }

        Ok(())
    }
}

struct Pass {
    command_buffer: vk::CommandBuffer,
    required_mats: Vec<Handle>,
}

pub trait Invocation {
    fn dispatch(&self, device: &DeviceLoader, command_buffer: vk::CommandBuffer);
}

struct BufferAction {
    matrix: Handle,
    read: bool,
    write: bool,
}

impl Drop for TensorSludge {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device
                .destroy_command_pool(Some(self.command_pool), None);
        }
    }
}
