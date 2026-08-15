use neural_amp_modeler::pure::PureNamModel;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let mut m = PureNamModel::from_bytes(&bytes).unwrap();
    m.reset(48000.0, 128);
    let input = vec![0.1f64; 128];
    let mut out = vec![0.0f64; 128];
    let start = std::time::Instant::now();
    let blocks = 48000 / 128;
    for _ in 0..blocks { m.process(&input, &mut out); }
    let el = start.elapsed();
    println!("{path}: 1s of audio in {:?} ({}x realtime)", el, 1.0 / el.as_secs_f64());
}
