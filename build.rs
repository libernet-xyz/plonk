use std::io::Result;
use std::path::PathBuf;

const PROTOS: [&str; 2] = ["circuit.proto", "proof.proto"];

fn main() -> Result<()> {
    let pcs_proto_dir = PathBuf::from(std::env::var("DEP_STARKOM_PCS_V9_PROTO_DIR").unwrap());
    prost_build::Config::new()
        .extern_path(
            ".starkom.proto.pcs.v9",
            "::starkom_pcs::starkom::proto::pcs::v9",
        )
        .compile_protos(
            &PROTOS.map(|proto| format!("proto/{proto}")),
            &[PathBuf::from("proto"), pcs_proto_dir.clone()],
        )?;

    let include_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("include");
    std::fs::create_dir_all(include_dir.join("plonk"))?;
    std::fs::create_dir_all(include_dir.join("pcs"))?;
    for proto in PROTOS {
        std::fs::copy(
            format!("proto/{proto}"),
            include_dir.join("plonk").join(proto),
        )?;
    }
    std::fs::copy(
        pcs_proto_dir.join("pcs/pcs.proto"),
        include_dir.join("pcs/pcs.proto"),
    )?;
    println!("cargo::metadata=proto_dir={}", include_dir.display());

    Ok(())
}
