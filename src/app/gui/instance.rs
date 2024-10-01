pub(super) struct Instance {
    pub(super) position: cgmath::Vector3<f32>,
    pub(super) rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    pub(super) fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
pub(super) struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

pub const INSTANCE_RAW_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: size_of::<InstanceRaw>() as _,
    // Step mode Instance means that our shaders will only change to use the
    // next entry when the shader starts processing a new instance.
    step_mode: wgpu::VertexStepMode::Instance,
    // Slots 0 to 4 are used by MODEL_VERTEX_LAYOUT.
    attributes: &wgpu::vertex_attr_array![
        // A mat4 takes up 4 vertex slots as it is technically 4 vec4s
        5 => Float32x4,
        6 => Float32x4,
        7 => Float32x4,
        8 => Float32x4,

        // And a mat3 with 3 vec3s, I guess.
        9 => Float32x3,
        10 => Float32x3,
        11 => Float32x3,
    ],
};
