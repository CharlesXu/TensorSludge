use crate::matrix::Matrix;
use crate::sigmoid::Sigmoid;
use anyhow::{format_err, Context, Result};
use erupt::{
    cstr,
    utils::{
        allocator::{Allocator, AllocatorCreateInfo},
        loading::DefaultEntryLoader,
    },
    vk1_0 as vk, DeviceLoader, EntryLoader, InstanceLoader,
};
use genmap::{GenMap, Handle};
use std::ffi::CString;
use std::sync::{Arc, Mutex};

/// The TensorSludge engine
pub struct TensorSludge {
    matrices: GenMap<Matrix>,
    core: SharedCore,
    sigmoid: Sigmoid,
    queue: vk::Queue,
}

pub struct Core {
    pub allocator: Mutex<Allocator>,
    pub device: DeviceLoader,
    pub instance: InstanceLoader,
    _entry: DefaultEntryLoader,
}

pub type SharedCore = Arc<Core>;

impl TensorSludge {
    /// Create a new TensorSludge instance
    pub fn new() -> Result<Self> {
        let entry = EntryLoader::new()?;

        // Instance
        let name = CString::new("TensorSludge")?;
        let app_info = vk::ApplicationInfoBuilder::new()
            .application_name(&name)
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(&name)
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));

        // Instance and device layers and extensions
        let mut instance_layers = Vec::new();
        let mut instance_extensions = Vec::new();
        let mut device_layers = Vec::new();
        let device_extensions = Vec::new();

        // Vulkan layers and extensions
        if cfg!(debug_assertions) {
            const LAYER_KHRONOS_VALIDATION: *const i8 = cstr!("VK_LAYER_KHRONOS_validation");
            instance_extensions
                .push(erupt::extensions::ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
            instance_layers.push(LAYER_KHRONOS_VALIDATION);
            device_layers.push(LAYER_KHRONOS_VALIDATION);
        }

        // Instance creation
        let create_info = vk::InstanceCreateInfoBuilder::new()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&instance_layers);

        let instance = InstanceLoader::new(&entry, &create_info, None)?;

        // Hardware selection
        let (queue_family_index, physical_device) = select_device(&instance)?;

        // Create logical device and queues
        let create_info = [vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0])];

        let physical_device_features = vk::PhysicalDeviceFeaturesBuilder::new();
        let create_info = vk::DeviceCreateInfoBuilder::new()
            .queue_create_infos(&create_info)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&device_extensions)
            .enabled_layer_names(&device_layers);

        let device = DeviceLoader::new(&instance, physical_device, &create_info, None)?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0, None) };

        // Allocator
        let allocator =
            Allocator::new(&instance, physical_device, AllocatorCreateInfo::default()).result()?;

        let core = Arc::new(Core {
            allocator: Mutex::new(allocator),
            device,
            instance,
            _entry: entry,
        });

        let sigmoid = Sigmoid::new(core.clone())?;

        Ok(Self {
            matrices: GenMap::with_capacity(10),
            sigmoid,
            queue,
            core,
        })
    }

    /// Create a new matrix with the specified dimensions
    pub fn matrix(&mut self, rows: usize, cols: usize) -> Result<crate::Matrix> {
        Ok(crate::Matrix(self.matrices.insert(Matrix::new(
            rows,
            cols,
            self.core.clone(),
        )?)))
    }

    fn get_matrix<'a>(&'a mut self, matrix: crate::Matrix) -> Result<&'a mut Matrix> {
        self.matrices
            .get_mut(matrix.0)
            .context("Matrix was deleted")
    }

    /// Write data to a matrix in row-major order
    pub fn write(&mut self, matrix: crate::Matrix, data: &[f32]) -> Result<()> {
        self.get_matrix(matrix)?.write(data)
    }

    /// Read data from a matrix in row-major order
    pub fn read(&mut self, matrix: crate::Matrix, data: &mut [f32]) -> Result<()> {
        self.get_matrix(matrix)?.read(data)
    }

    /// Create a pass from a sequence of operations
    pub fn create_pass(&mut self, ops: &[crate::Operation]) -> Result<crate::Pass> {
        use crate::Operation as Op;

        // Collect descriptor pool sizes and layouts from each of the operations
        let mut pool_sizes = Vec::new();
        let mut descriptor_set_layouts = Vec::new();
        for op in ops {
            match op {
                Op::Sigmoid(_) => {
                    Sigmoid::desc_pool_sizes(&mut pool_sizes);
                    descriptor_set_layouts.push(self.sigmoid.desc_set_layout());
                }
                _ => todo!("Not all ops are implemented"),
            }
        }

        // Create descriptor pool of appropriate size
        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets(ops.len() as _); // TODO: Some ops might not need descriptor sets at all! This is potentially wasteful
        let descriptor_pool = unsafe {
            self.core
                .device
                .create_descriptor_pool(&create_info, None, None)
        }
        .result()?;

        // Create descriptor sets
        let create_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);
        let descriptor_sets =
            unsafe { self.core.device.allocate_descriptor_sets(&create_info) }.result()?;

        todo!("create_pass")
    }

    /// Run the specified pass on the TensorSludge engine
    pub fn flow(&mut self, pass: crate::Pass) -> Result<()> {
        todo!("flow")
    }
}

fn select_device(instance: &InstanceLoader) -> Result<(u32, vk::PhysicalDevice)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices(None) }.result()?;
    for device in physical_devices {
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(device, None) };
        for (family, properites) in families.iter().enumerate() {
            if properites.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Ok((family as u32, device));
            }
        }
    }
    Err(format_err!("No suitable device found"))
}

use std::sync::MutexGuard;
impl Core {
    pub fn allocator(&self) -> Result<MutexGuard<Allocator>> {
        self.allocator
            .lock()
            .map_err(|_| format_err!("Allocator mutex poisoned"))
    }
}

impl Drop for TensorSludge {
    fn drop(&mut self) {}
}
