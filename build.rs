use std::path::PathBuf;

fn main() {
    // On wasm32 there is no C++ toolchain/stdlib; the crate uses the pure-Rust
    // inference engine (src/pure) instead of the vendored C++ core.
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "wasm32" {
        return;
    }

    let nam_core = PathBuf::from("NeuralAmpModelerCore");
    let nam_src = nam_core.join("NAM");
    let deps = nam_core.join("Dependencies");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .std("c++20")
        // Include paths
        .include(&nam_core)
        .include(deps.join("eigen"))
        .include(deps.join("nlohmann"))
        // Unity build: all NAM sources compiled in one TU to preserve
        // static ConfigParserHelper registrations from linker stripping.
        .file("shim/nam_all.cpp")
        // C shim
        .file("shim/nam_shim.cpp")
        // Optimise — NAM inference is CPU-intensive
        .opt_level(3);

    // Platform-specific flags
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" | "ios" => {
            build.flag("-stdlib=libc++");
        }
        "windows" => {
            build.define("NOMINMAX", None);
            build.define("WIN32_LEAN_AND_MEAN", None);
        }
        _ => {}
    }

    build.compile("nam_core");

    // Link C++ stdlib
    match target_os.as_str() {
        "macos" | "ios" => println!("cargo:rustc-link-lib=c++"),
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
        }
        _ => {}
    }

    // Build the reference renderer tool for integration tests.
    // This is a standalone C++ executable that processes audio through NAM
    // directly, so we can compare its output against our Rust bindings.
    // Host-only: it shells out to the host g++, meaningless when
    // cross-compiling (iOS).
    if target_os == "ios" {
        return;
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ref_tool = out_dir.join("reference_render");

    let mut ref_build = cc::Build::new();
    ref_build
        .cpp(true)
        .std("c++20")
        .include(&nam_core)
        .include(deps.join("eigen"))
        .include(deps.join("nlohmann"))
        .opt_level(3)
        .warnings(false);

    match target_os.as_str() {
        "macos" => {
            ref_build.flag("-stdlib=libc++");
        }
        "windows" => {
            ref_build.define("NOMINMAX", None);
            ref_build.define("WIN32_LEAN_AND_MEAN", None);
        }
        _ => {}
    }

    // Compile object files
    let ref_objs: Vec<PathBuf> = ["tools/reference_render.cpp", "shim/nam_all.cpp"]
        .iter()
        .map(|f| {
            let src = PathBuf::from(f);
            let obj = out_dir.join(src.file_stem().unwrap()).with_extension("o");
            let mut obj_build = cc::Build::new();
            obj_build
                .cpp(true)
                .std("c++20")
                .include(&nam_core)
                .include(deps.join("eigen"))
                .include(deps.join("nlohmann"))
                .opt_level(3)
                .warnings(false);
            match target_os.as_str() {
                "macos" => {
                    obj_build.flag("-stdlib=libc++");
                }
                "windows" => {
                    obj_build.define("NOMINMAX", None);
                    obj_build.define("WIN32_LEAN_AND_MEAN", None);
                }
                _ => {}
            }
            let _ = std::process::Command::new("g++")
                .arg("-std=c++20")
                .arg("-O3")
                .arg("-c")
                .arg(f)
                .arg("-o")
                .arg(&obj)
                .arg(format!("-I{}", nam_core.display()))
                .arg(format!("-I{}", deps.join("eigen").display()))
                .arg(format!("-I{}", deps.join("nlohmann").display()))
                .status();
            obj
        })
        .collect();

    // Link into executable
    let mut link_cmd = std::process::Command::new("g++");
    link_cmd.arg("-o").arg(&ref_tool);
    for obj in &ref_objs {
        link_cmd.arg(obj);
    }
    if target_os == "linux" {
        link_cmd.arg("-lstdc++fs");
    }
    let _ = link_cmd.status();

    println!(
        "cargo:rustc-env=NAM_REFERENCE_RENDER={}",
        ref_tool.display()
    );

    // Rerun if sources change
    println!("cargo:rerun-if-changed=shim/nam_shim.cpp");
    println!("cargo:rerun-if-changed=shim/nam_shim.h");
    println!("cargo:rerun-if-changed=shim/nam_all.cpp");
    println!("cargo:rerun-if-changed=tools/reference_render.cpp");
    println!("cargo:rerun-if-changed={}", nam_src.display());
}
