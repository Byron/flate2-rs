use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const MAX_SLOWDOWN_RATIO: f64 = 2.0;

#[test]
#[ignore = "run with `cargo test --release --test backend-regression-bench -- --ignored --nocapture`"]
fn zlib_rs_does_not_regress_against_zlib_ng() {
    assert!(
        !cfg!(debug_assertions),
        "run this benchmark with `cargo test --release --test backend-regression-bench -- --ignored --nocapture`"
    );

    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("backend-bench");
    fs::create_dir_all(&output_dir).unwrap();

    let baseline = run_benchmark("zlib-ng,miniz_oxide", output_dir.join("zlib-ng.csv"));
    let candidate = run_benchmark("zlib-rs,miniz_oxide", output_dir.join("zlib-rs.csv"));

    for (api, baseline_row) in &baseline {
        let candidate_row = candidate.get(api).unwrap();
        let slowdown = candidate_row.nanos as f64 / baseline_row.nanos as f64;
        assert!(
            slowdown <= MAX_SLOWDOWN_RATIO,
            "zlib-rs benchmark `{api}` regressed: {:.2}x slower than zlib-ng ({} ns vs {} ns). CSV results: {} and {}",
            slowdown,
            candidate_row.nanos,
            baseline_row.nanos,
            output_dir.join("zlib-ng.csv").display(),
            output_dir.join("zlib-rs.csv").display(),
        );
    }
}

fn run_benchmark(features: &str, csv_path: PathBuf) -> BTreeMap<String, BenchmarkRow> {
    let output = Command::new(env!("CARGO"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "run",
            "--quiet",
            "--release",
            "--example",
            "benchmark",
            "--no-default-features",
            "--features",
            features,
        ])
        .output()
        .unwrap();

    if !output.status.success() {
        panic!(
            "benchmark run failed for `{features}`:\nstdout:\n{}\n\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    fs::write(&csv_path, &output.stdout).unwrap();
    parse_rows(std::str::from_utf8(&output.stdout).unwrap())
}

fn parse_rows(csv: &str) -> BTreeMap<String, BenchmarkRow> {
    let mut rows = BTreeMap::new();
    for line in csv.lines().skip(1).filter(|line| !line.trim().is_empty()) {
        let mut fields = line.split(',');
        let backend = fields.next().unwrap();
        let api = fields.next().unwrap().to_owned();
        let _iterations = fields.next().unwrap();
        let _bytes = fields.next().unwrap();
        let nanos = fields.next().unwrap().parse::<u128>().unwrap();
        let _throughput = fields.next().unwrap();
        assert!(fields.next().is_none(), "unexpected CSV row: {}", line);
        assert!(
            backend == "zlib-ng" || backend == "zlib-rs",
            "unexpected backend `{}` in row `{}`",
            backend,
            line
        );
        rows.insert(api, BenchmarkRow { nanos });
    }
    rows
}

struct BenchmarkRow {
    nanos: u128,
}
