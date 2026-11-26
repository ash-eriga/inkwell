//! Startup phase profiler (`INKWELL_LOADPROF=1`): accumulating timers +
//! absolute marks, printed to stderr. Zero-cost when off (one atomic load).
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static ON: AtomicBool = AtomicBool::new(false);
static T0: Mutex<Option<Instant>> = Mutex::new(None);
static ACC: Mutex<Vec<(&'static str, f64)>> = Mutex::new(Vec::new());

/// Read the env flag and pin the process-relative clock. Idempotent.
pub fn init() {
    ON.store(std::env::var_os("INKWELL_LOADPROF").is_some(), Ordering::Relaxed);
    let mut t = T0.lock().expect("prof");
    if t.is_none() {
        *t = Some(Instant::now());
    }
}

#[must_use]
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

fn t0() -> Instant {
    T0.lock().expect("prof").unwrap_or_else(Instant::now)
}

/// Add `ms` to phase `name`'s accumulator.
pub fn add(name: &'static str, ms: f64) {
    if on() {
        ACC.lock().expect("prof").push((name, ms));
    }
}

/// Absolute mark: `loadprof@ <name> <ms since init>`.
pub fn mark(name: &str) {
    if on() {
        eprintln!("loadprof@ {name} {:.1}", t0().elapsed().as_secs_f64() * 1e3);
    }
}

/// RAII phase timer; records on drop.
#[derive(Debug)]
pub struct Guard(&'static str, Instant);

/// Time a scope.
#[must_use]
pub fn guard(name: &'static str) -> Guard {
    Guard(name, Instant::now())
}

impl Drop for Guard {
    fn drop(&mut self) {
        add(self.0, self.1.elapsed().as_secs_f64() * 1e3);
    }
}

/// Print the aggregated phase table (largest first), then clear.
pub fn dump() {
    if !on() {
        return;
    }
    let acc = std::mem::take(&mut *ACC.lock().expect("prof"));
    let mut agg: std::collections::BTreeMap<&'static str, (f64, usize)> = Default::default();
    for (n, ms) in acc {
        let e = agg.entry(n).or_default();
        e.0 += ms;
        e.1 += 1;
    }
    let mut v: Vec<_> = agg.into_iter().collect();
    v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    for (n, (ms, c)) in v {
        eprintln!("loadprof: {n:<18} {ms:9.1}ms n={c}");
    }
}
