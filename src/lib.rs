// Copyright 2026 The Libernet Team
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod starkom {
    pub mod proto {
        pub mod plonk {
            pub mod v8 {
                include!(concat!(env!("OUT_DIR"), "/starkom.proto.plonk.v8.rs"));
            }
        }
    }
}

mod chip;
mod expr;
mod lexer;
mod parser;
mod plonk;
mod utils;
mod witness;

pub use chip::*;
pub use expr::*;
pub use plonk::*;
pub use witness::*;
