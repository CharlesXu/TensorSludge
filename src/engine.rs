use crate::sigmoid::Sigmoid;
use anyhow::{Result, Context, format_err};
use genmap::Handle;
use erupt::{
    cstr,
    utils::{
        allocator::{Allocator, AllocatorCreateInfo, MemoryTypeFinder},
        decode_spv,
        loading::DefaultEntryLoader,
    },
    vk1_0 as vk, DeviceLoader, EntryLoader, InstanceLoader,
};
use std::ffi::CString;
use std::sync::Arc;

/// The TensorSludge engine
pub struct TensorSludge {
    sigmoid: Sigmoid,
    allocator: Allocator,
    queue: vk::Queue,
    device: Arc<DeviceLoader>,
    instance: InstanceLoader,
    _entry: DefaultEntryLoader,
}

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
        let device = Arc::new(device);
        let queue = unsafe { device.get_device_queue(queue_family_index, 0, None) };

        // Allocator
        let mut allocator =
            Allocator::new(&instance, physical_device, AllocatorCreateInfo::default()).result()?;

        let sigmoid = Sigmoid::new(device.clone())?;

        Ok(Self {
            sigmoid,
            allocator,
            queue,
            device,
            instance,
            _entry: entry,
        })
    }

    /// Create a new matrix with the specified dimensions
    pub fn matrix(&mut self, rows: usize, cols: usize) -> Result<crate::Matrix> {
        todo!("matrix")
    }

    /// Write data to a matrix in row-major order
    pub fn write(&mut self, matrix: crate::Matrix, data: &[f32]) -> Result<()> {
        todo!("write")
    }

    /// Read data from a matrix in row-major order
    pub fn read(&mut self, matrix: crate::Matrix, data: &mut [f32]) -> Result<()> {
        todo!("read")
    }

    /// Create a pass from a sequence of operations
    pub fn create_pass(&mut self, ops: &[crate::Operation]) -> Result<crate::Pass> {
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

impl Drop for TensorSludge {
    fn drop(&mut self) {
    }
}
