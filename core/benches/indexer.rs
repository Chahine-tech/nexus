use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(_c: &mut Criterion) {
    // benches to be filled as needed
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
