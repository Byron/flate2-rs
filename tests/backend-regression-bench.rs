use flate2::write::ZlibEncoder;
use flate2::{
    read::ZlibDecoder, Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status,
};
use std::fs;
use std::hint::black_box;
use std::io::{Read, Write};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const CHUNK_IN: usize = 2 * 1024;
const CHUNK_OUT: usize = 2 * 1024 * 1024;
const PLAIN_LEN: usize = 16 * 1024 * 1024;
const BENCH_TARGET_SAMPLE_TIME: Duration = Duration::from_millis(200);
// Keep this odd so `samples[samples.len() / 2]` stays the true median.
const BENCH_SAMPLES: usize = 5;
const BENCH_MAX_ITERS_PER_SAMPLE: usize = 12;

struct BenchmarkData {
    plain: Vec<u8>,
    zlib: Vec<u8>,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    max_regression_factor: f64,
    run: fn(&BenchmarkData),
}

#[derive(Clone)]
struct BaselineRecord {
    backend: String,
    case: String,
    baseline_ns_per_byte: f64,
    max_regression_factor: f64,
}

struct BenchmarkResult {
    case: &'static str,
    iterations_per_sample: usize,
    samples: usize,
    ns_per_byte: f64,
    baseline_ns_per_byte: f64,
    max_regression_factor: f64,
}

const CASES: &[Case] = &[
    Case {
        name: "compress_chunked_large_output_buf",
        max_regression_factor: 1.75,
        run: run_compress_chunked_large_output_buf,
    },
    Case {
        name: "compress_uninit_chunked_large_output_buf",
        max_regression_factor: 1.75,
        run: run_compress_uninit_chunked_large_output_buf,
    },
    Case {
        name: "decompress_chunked_large_output_buf",
        max_regression_factor: 1.75,
        run: run_decompress_chunked_large_output_buf,
    },
    Case {
        name: "decompress_uninit_chunked_large_output_buf",
        max_regression_factor: 1.75,
        run: run_decompress_uninit_chunked_large_output_buf,
    },
];

fn benchmark_data() -> BenchmarkData {
    let line =
        b"The quick brown fox jumps over the lazy dog. 0123456789 abcdefghijklmnopqrstuvwxyz\n";
    let mut plain = Vec::with_capacity(PLAIN_LEN);
    while plain.len() < PLAIN_LEN {
        plain.extend_from_slice(line);
    }
    plain.truncate(PLAIN_LEN);

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&plain).unwrap();
    let zlib = encoder.finish().unwrap();

    BenchmarkData { plain, zlib }
}

fn initialized_output_chunk() -> Box<[u8]> {
    vec![0u8; CHUNK_OUT].into_boxed_slice()
}

fn uninit_output_chunk() -> Box<[MaybeUninit<u8>]> {
    vec![MaybeUninit::<u8>::uninit(); CHUNK_OUT].into_boxed_slice()
}

fn initialized_prefix(output: &[MaybeUninit<u8>], bytes_written: usize) -> &[u8] {
    unsafe {
        // SAFETY: The benchmark only reads the prefix reported by total_out() for a
        // completed compress/decompress call, which is the initialized portion.
        std::slice::from_raw_parts(output.as_ptr() as *const u8, bytes_written)
    }
}

fn run_decompress_chunked_large_output_buf(data: &BenchmarkData) {
    let mut decoder = Decompress::new(true);
    let mut chunk = initialized_output_chunk();
    let mut result = Vec::with_capacity(data.plain.len());

    loop {
        let prior_out = decoder.total_out();
        let in_start = decoder.total_in() as usize;
        let in_end = (in_start + CHUNK_IN).min(data.zlib.len());
        let status = decoder
            .decompress(
                &data.zlib[in_start..in_end],
                &mut chunk,
                FlushDecompress::None,
            )
            .unwrap();
        let bytes_written = (decoder.total_out() - prior_out) as usize;
        result.extend_from_slice(&chunk[..bytes_written]);
        if status == Status::StreamEnd {
            break;
        }
    }

    assert_eq!(result, data.plain);
}

fn run_decompress_uninit_chunked_large_output_buf(data: &BenchmarkData) {
    let mut decoder = Decompress::new(true);
    let mut chunk = uninit_output_chunk();
    let mut result = Vec::with_capacity(data.plain.len());

    loop {
        let prior_out = decoder.total_out();
        let in_start = decoder.total_in() as usize;
        let in_end = (in_start + CHUNK_IN).min(data.zlib.len());
        let status = decoder
            .decompress_uninit(
                &data.zlib[in_start..in_end],
                &mut chunk,
                FlushDecompress::None,
            )
            .unwrap();
        let bytes_written = (decoder.total_out() - prior_out) as usize;
        result.extend_from_slice(initialized_prefix(&chunk, bytes_written));
        if status == Status::StreamEnd {
            break;
        }
    }

    assert_eq!(result, data.plain);
}

fn run_compress_chunked_large_output_buf(data: &BenchmarkData) {
    let mut encoder = Compress::new(Compression::fast(), true);
    let mut chunk = initialized_output_chunk();
    let mut result = Vec::with_capacity(data.zlib.len() * 2);

    loop {
        let prior_out = encoder.total_out();
        let in_start = encoder.total_in() as usize;
        let in_end = (in_start + CHUNK_IN).min(data.plain.len());
        let flush = if in_end == data.plain.len() {
            FlushCompress::Finish
        } else {
            FlushCompress::None
        };
        let status = encoder
            .compress(&data.plain[in_start..in_end], &mut chunk, flush)
            .unwrap();
        let bytes_written = (encoder.total_out() - prior_out) as usize;
        result.extend_from_slice(&chunk[..bytes_written]);
        if status == Status::StreamEnd {
            break;
        }
    }

    assert_roundtrip_matches_plain(&result, &data.plain);
}

fn run_compress_uninit_chunked_large_output_buf(data: &BenchmarkData) {
    let mut encoder = Compress::new(Compression::fast(), true);
    let mut chunk = uninit_output_chunk();
    let mut result = Vec::with_capacity(data.zlib.len() * 2);

    loop {
        let prior_out = encoder.total_out();
        let in_start = encoder.total_in() as usize;
        let in_end = (in_start + CHUNK_IN).min(data.plain.len());
        let flush = if in_end == data.plain.len() {
            FlushCompress::Finish
        } else {
            FlushCompress::None
        };
        let status = encoder
            .compress_uninit(&data.plain[in_start..in_end], &mut chunk, flush)
            .unwrap();
        let bytes_written = (encoder.total_out() - prior_out) as usize;
        result.extend_from_slice(initialized_prefix(&chunk, bytes_written));
        if status == Status::StreamEnd {
            break;
        }
    }

    assert_roundtrip_matches_plain(&result, &data.plain);
}

fn assert_roundtrip_matches_plain(compressed: &[u8], plain: &[u8]) {
    let mut decoder = ZlibDecoder::new(compressed);
    let mut decoded = Vec::with_capacity(plain.len());
    decoder.read_to_end(&mut decoded).unwrap();
    assert_eq!(decoded, plain);
}

fn benchmark_case(data: &BenchmarkData, case: Case) -> BenchmarkResult {
    let warmup_started = Instant::now();
    (case.run)(data);
    let warmup_elapsed = warmup_started.elapsed();
    let iterations_per_sample = sample_iterations(warmup_elapsed);

    let mut samples = Vec::with_capacity(BENCH_SAMPLES);
    for _ in 0..BENCH_SAMPLES {
        let sample_started = Instant::now();
        for _ in 0..iterations_per_sample {
            (case.run)(black_box(data));
        }
        samples.push(sample_started.elapsed());
    }
    samples.sort_unstable();

    let median = samples[samples.len() / 2];
    let iterations = iterations_per_sample as f64;
    let bytes = data.plain.len() as f64;
    let ns_per_byte = median.as_nanos() as f64 / (iterations * bytes);

    BenchmarkResult {
        case: case.name,
        iterations_per_sample,
        samples: BENCH_SAMPLES,
        ns_per_byte,
        baseline_ns_per_byte: 0.0,
        max_regression_factor: case.max_regression_factor,
    }
}

fn sample_iterations(warmup_elapsed: Duration) -> usize {
    let warmup_nanos = warmup_elapsed.as_nanos();
    let target_nanos = BENCH_TARGET_SAMPLE_TIME.as_nanos();
    let iterations = if warmup_nanos == 0 {
        // Extremely fast warmups should still produce bounded sample counts and
        // must not divide by zero when computing iterations.
        BENCH_MAX_ITERS_PER_SAMPLE
    } else {
        (target_nanos / warmup_nanos) as usize
    };
    iterations.clamp(1, BENCH_MAX_ITERS_PER_SAMPLE)
}

fn baseline_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("backend-regression-bench-baseline.csv")
}

fn load_baselines() -> Vec<BaselineRecord> {
    let contents = fs::read_to_string(baseline_path()).unwrap();
    contents
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .skip(1)
        .map(parse_baseline_record)
        .collect()
}

fn parse_baseline_record(line: &str) -> BaselineRecord {
    let mut fields = line.split(',');
    let backend = fields
        .next()
        .expect("missing backend field in baseline CSV")
        .trim()
        .to_owned();
    let case = fields
        .next()
        .expect("missing case field in baseline CSV")
        .trim()
        .to_owned();
    let baseline_ns_per_byte = fields
        .next()
        .expect("missing baseline_ns_per_byte field in baseline CSV")
        .trim()
        .parse()
        .expect("invalid baseline_ns_per_byte field in baseline CSV");
    let max_regression_factor = fields
        .next()
        .expect("missing max_regression_factor field in baseline CSV")
        .trim()
        .parse()
        .expect("invalid max_regression_factor field in baseline CSV");
    assert!(
        fields.next().is_none(),
        "unexpected trailing baseline fields"
    );
    BaselineRecord {
        backend,
        case,
        baseline_ns_per_byte,
        max_regression_factor,
    }
}

fn apply_baseline(backend: &str, result: &mut BenchmarkResult, baselines: &[BaselineRecord]) {
    let baseline = baselines
        .iter()
        .find(|baseline| baseline.backend == backend && baseline.case == result.case)
        .unwrap_or_else(|| {
            panic!(
                "missing baseline for backend={backend}, case={}",
                result.case
            )
        });
    result.baseline_ns_per_byte = baseline.baseline_ns_per_byte;
    result.max_regression_factor = baseline.max_regression_factor;
}

fn results_dir() -> PathBuf {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    target_dir.join("backend-bench")
}

fn write_results_csv(backend: &str, results: &[BenchmarkResult]) {
    let dir = results_dir();
    fs::create_dir_all(&dir).unwrap();

    let mut csv = String::from(
        "backend,case,iterations_per_sample,samples,ns_per_byte,baseline_ns_per_byte,max_regression_factor,allowed_ns_per_byte,status\n",
    );
    for result in results {
        let allowed_ns_per_byte = result.baseline_ns_per_byte * result.max_regression_factor;
        let status = if result.ns_per_byte <= allowed_ns_per_byte {
            "pass"
        } else {
            "fail"
        };
        csv.push_str(&format!(
            "{backend},{},{},{},{:.9},{:.9},{:.3},{:.9},{}\n",
            result.case,
            result.iterations_per_sample,
            result.samples,
            result.ns_per_byte,
            result.baseline_ns_per_byte,
            result.max_regression_factor,
            allowed_ns_per_byte,
            status,
        ));
    }

    fs::write(dir.join(format!("{backend}.csv")), csv).unwrap();
}

#[cfg(feature = "zlib-ng")]
fn backend_name() -> &'static str {
    "zlib-ng"
}

#[cfg(all(not(feature = "zlib-ng"), feature = "zlib-ng-compat"))]
fn backend_name() -> &'static str {
    "zlib-ng-compat"
}

#[cfg(all(
    not(feature = "zlib-ng"),
    not(feature = "zlib-ng-compat"),
    feature = "zlib-rs"
))]
fn backend_name() -> &'static str {
    "zlib-rs"
}

#[cfg(all(
    not(feature = "zlib-ng"),
    not(feature = "zlib-ng-compat"),
    not(feature = "zlib-rs"),
    any(
        feature = "zlib",
        feature = "zlib-default",
        feature = "cloudflare_zlib"
    )
))]
fn backend_name() -> &'static str {
    "zlib"
}

#[cfg(all(
    not(feature = "zlib-ng"),
    not(feature = "zlib-ng-compat"),
    not(feature = "zlib-rs"),
    not(feature = "zlib"),
    not(feature = "zlib-default"),
    not(feature = "cloudflare_zlib")
))]
fn backend_name() -> &'static str {
    "rust_backend"
}

#[test]
#[ignore]
fn backend_regression_bench() {
    let backend = backend_name();
    let data = benchmark_data();
    let baselines = load_baselines();
    let mut results = Vec::with_capacity(CASES.len());

    for case in CASES {
        let mut result = benchmark_case(&data, *case);
        apply_baseline(backend, &mut result, &baselines);
        results.push(result);
    }

    write_results_csv(backend, &results);

    let failures: Vec<_> = results
        .iter()
        .filter(|result| {
            result.ns_per_byte > result.baseline_ns_per_byte * result.max_regression_factor
        })
        .collect();
    assert!(
        failures.is_empty(),
        "backend regression benchmark failures for {backend}: {}",
        failures
            .iter()
            .map(|result| format!(
                "{} measured {:.9} ns/byte, allowed {:.9} ns/byte",
                result.case,
                result.ns_per_byte,
                result.baseline_ns_per_byte * result.max_regression_factor
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
}
