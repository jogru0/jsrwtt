#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LightUniform {
    pub(super) position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    pub(super) _padding: u32,
    pub(super) color: [f32; 3],
    pub(super) _padding2: u32,
}
