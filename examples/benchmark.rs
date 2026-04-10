use std::fmt::Write as _;
use std::io::{Cursor, Read};
use std::time::{Duration, Instant};

use flate2::{bufread, read, Decompress, FlushDecompress, Status};

const DATA_BYTES: usize = 8 * 1024 * 1024;
const ITERATIONS: usize = 8;

fn main() {
    let plain = sample_data();
    let gzip = gzip_bytes(&plain);
    let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);

    let results = [
        bench_case("read::GzDecoder", plain.len(), || {
            let mut decoder = read::GzDecoder::new(gzip.as_slice());
            let mut output = Vec::with_capacity(plain.len());
            decoder.read_to_end(&mut output).unwrap();
            output
        }),
        bench_case("bufread::GzDecoder", plain.len(), || {
            let mut decoder = bufread::GzDecoder::new(Cursor::new(gzip.as_slice()));
            let mut output = Vec::with_capacity(plain.len());
            decoder.read_to_end(&mut output).unwrap();
            output
        }),
        bench_case("read::ZlibDecoder", plain.len(), || {
            let mut decoder = read::ZlibDecoder::new(zlib.as_slice());
            let mut output = Vec::with_capacity(plain.len());
            decoder.read_to_end(&mut output).unwrap();
            output
        }),
        bench_case("Decompress::decompress", plain.len(), || {
            let mut decoder = Decompress::new(true);
            let mut output = vec![0; plain.len()];
            let status = decoder
                .decompress(zlib.as_slice(), &mut output, FlushDecompress::Finish)
                .unwrap();
            assert_eq!(status, Status::StreamEnd);
            output
        }),
    ];

    println!("backend,api,iterations,bytes,nanos,throughput_mib_s");
    for result in results {
        println!(
            "{},{},{},{},{},{}",
            backend_name(),
            result.api,
            result.iterations,
            result.bytes,
            result.nanos,
            result.throughput_mib_s
        );
    }
}

struct BenchmarkResult {
    api: &'static str,
    iterations: usize,
    bytes: usize,
    nanos: u128,
    throughput_mib_s: String,
}

fn bench_case<F>(api: &'static str, expected_len: usize, mut run_once: F) -> BenchmarkResult
where
    F: FnMut() -> Vec<u8>,
{
    let output = run_once();
    assert_eq!(output.len(), expected_len);

    let start = Instant::now();
    let mut total_bytes = 0usize;
    for _ in 0..ITERATIONS {
        let output = run_once();
        assert_eq!(output.len(), expected_len);
        total_bytes += output.len();
    }
    let elapsed = start.elapsed();

    BenchmarkResult {
        api,
        iterations: ITERATIONS,
        bytes: total_bytes,
        nanos: elapsed.as_nanos(),
        throughput_mib_s: format_throughput(total_bytes, elapsed),
    }
}

fn format_throughput(bytes: usize, elapsed: Duration) -> String {
    let mib_per_s = bytes as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64();
    format!("{mib_per_s:.2}")
}

fn sample_data() -> Vec<u8> {
    let mut output = String::with_capacity(DATA_BYTES + (DATA_BYTES / 8));
    let mut i = 0usize;
    while output.len() < DATA_BYTES {
        let _ = write!(
            output,
            "GET /objects/{i:08x}/chunks/{:08x} HTTP/1.1\r\nhost: example.invalid\r\netag: {:08x}{:08x}\r\naccept-encoding: gzip,deflate\r\n\r\n",
            i.wrapping_mul(17),
            i.wrapping_mul(97),
            i.wrapping_mul(131)
        );
        output.push_str("content-type=application/octet-stream;");
        output.push_str("cache-control=max-age=31536000;");
        output.push_str("vary=accept-encoding;");
        output.push_str("payload=");
        for j in 0..128u8 {
            output.push(char::from(b'a' + ((i as u8).wrapping_add(j) % 26)));
        }
        output.push('\n');
        i += 1;
    }
    output.truncate(DATA_BYTES);
    output.into_bytes()
}

fn gzip_bytes(plain: &[u8]) -> Vec<u8> {
    let deflate = miniz_oxide::deflate::compress_to_vec(plain, 6);
    let mut output = Vec::with_capacity(deflate.len() + 18);
    output.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff]);
    output.extend_from_slice(&deflate);

    let mut crc = crc32fast::Hasher::new();
    crc.update(plain);
    output.extend_from_slice(&crc.finalize().to_le_bytes());
    output.extend_from_slice(&(plain.len() as u32).to_le_bytes());
    output
}

fn backend_name() -> &'static str {
    #[cfg(feature = "zlib-ng")]
    {
        return "zlib-ng";
    }

    #[cfg(all(not(feature = "zlib-ng"), feature = "zlib-rs"))]
    {
        return "zlib-rs";
    }

    #[cfg(all(not(feature = "zlib-ng"), not(feature = "zlib-rs"), feature = "zlib"))]
    {
        return "zlib";
    }

    #[cfg(all(
        not(feature = "zlib-ng"),
        not(feature = "zlib-rs"),
        not(feature = "zlib")
    ))]
    {
        return "rust_backend";
    }
}
