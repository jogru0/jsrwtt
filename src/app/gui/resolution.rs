#[derive(Clone, Copy)]
pub struct Resolution {
    width: u32,
    height: u32,
}

#[derive(Debug)]
pub enum ResolutionError {
    InvalidZero,
}

impl Resolution {
    pub fn new(width: u32, height: u32) -> Result<Self, ResolutionError> {
        if width == 0 || height == 0 {
            Err(ResolutionError::InvalidZero)
        } else {
            Ok(Resolution { width, height })
        }
    }

    pub fn width(self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}

impl TryFrom<PhysicalSize<u32>> for Resolution {
    type Error = ResolutionError;

    fn try_from(value: PhysicalSize<u32>) -> Result<Self, Self::Error> {
        Resolution::new(value.width, value.height)
    }
}

use std::{convert::TryFrom, fmt::Display};

use winit::dpi::PhysicalSize;

impl Display for Resolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}
