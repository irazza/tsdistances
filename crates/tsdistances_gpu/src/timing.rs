// Timing instrumentation module for detailed performance analysis
// Enable with the "timing" feature flag

use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering};

static TIMING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable timing collection
pub fn enable_timing() {
    TIMING_ENABLED.store(true, Ordering::SeqCst);
}

/// Disable timing collection
pub fn disable_timing() {
    TIMING_ENABLED.store(false, Ordering::SeqCst);
}

/// Check if timing is enabled
pub fn is_timing_enabled() -> bool {
    TIMING_ENABLED.load(Ordering::SeqCst)
}

/// Detailed timing breakdown for GPU operations
#[derive(Debug, Clone, Default)]
pub struct OperationTimings {
    // Setup phase
    pub device_properties_query: Duration,
    pub data_padding: Duration,
    pub buffer_struct_creation: Duration,
    
    // Per-chunk timings (accumulated)
    pub chunk_buffer_fill: Duration,
    pub command_buffer_creation: Duration,
    pub kernel_params_build: Duration,
    pub data_upload: Duration,
    pub kernel_dispatches: Duration,
    pub data_download: Duration,
    pub fence_wait: Duration,
    pub result_copy: Duration,
    
    // Cleanup
    pub buffer_clear: Duration,
    
    // Counters
    pub num_chunks: usize,
    pub num_kernel_dispatches: usize,
    pub total_pairs: usize,
}

impl OperationTimings {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn total_time(&self) -> Duration {
        self.device_properties_query
            + self.data_padding
            + self.buffer_struct_creation
            + self.chunk_buffer_fill
            + self.command_buffer_creation
            + self.kernel_params_build
            + self.data_upload
            + self.kernel_dispatches
            + self.data_download
            + self.fence_wait
            + self.result_copy
            + self.buffer_clear
    }
    
    pub fn gpu_only_time(&self) -> Duration {
        self.data_upload + self.kernel_dispatches + self.data_download + self.fence_wait
    }
    
    pub fn cpu_overhead_time(&self) -> Duration {
        self.total_time() - self.gpu_only_time()
    }
    
    pub fn print_report(&self) {
        let total = self.total_time();
        let total_secs = total.as_secs_f64();
        
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║                    TIMING BREAKDOWN REPORT                    ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");
        
        println!("\n┌─────────────────────────────────────────────────────────────────┐");
        println!("│ SETUP PHASE                                                     │");
        println!("├─────────────────────────────────────────────────────────────────┤");
        self.print_timing_line("Device properties query", self.device_properties_query, total_secs);
        self.print_timing_line("Data padding/flatten", self.data_padding, total_secs);
        self.print_timing_line("Buffer struct creation", self.buffer_struct_creation, total_secs);
        
        println!("├─────────────────────────────────────────────────────────────────┤");
        println!("│ GPU EXECUTION ({} chunks, {} dispatches)              │", 
            self.num_chunks, self.num_kernel_dispatches);
        println!("├─────────────────────────────────────────────────────────────────┤");
        self.print_timing_line("Chunk buffer fill (CPU)", self.chunk_buffer_fill, total_secs);
        self.print_timing_line("Command buffer creation", self.command_buffer_creation, total_secs);
        self.print_timing_line("Kernel params build", self.kernel_params_build, total_secs);
        self.print_timing_line("Data upload (CPU→GPU)", self.data_upload, total_secs);
        self.print_timing_line("Kernel dispatches", self.kernel_dispatches, total_secs);
        self.print_timing_line("Data download (GPU→CPU)", self.data_download, total_secs);
        self.print_timing_line("Fence wait", self.fence_wait, total_secs);
        self.print_timing_line("Result copy to matrix", self.result_copy, total_secs);
        
        println!("├─────────────────────────────────────────────────────────────────┤");
        println!("│ CLEANUP                                                         │");
        println!("├─────────────────────────────────────────────────────────────────┤");
        self.print_timing_line("Buffer clear", self.buffer_clear, total_secs);
        
        println!("└─────────────────────────────────────────────────────────────────┘");
        
        println!("\n┌─────────────────────────────────────────────────────────────────┐");
        println!("│ SUMMARY                                                         │");
        println!("├─────────────────────────────────────────────────────────────────┤");
        println!("│ Total time:       {:>10.3?}                                   │", total);
        println!("│ GPU time:         {:>10.3?} ({:>5.1}%)                         │", 
            self.gpu_only_time(), 100.0 * self.gpu_only_time().as_secs_f64() / total_secs);
        println!("│ CPU overhead:     {:>10.3?} ({:>5.1}%)                         │", 
            self.cpu_overhead_time(), 100.0 * self.cpu_overhead_time().as_secs_f64() / total_secs);
        println!("│ Pairs computed:   {:>10}                                   │", self.total_pairs);
        println!("│ Throughput:       {:>10.0} pairs/sec                        │", 
            self.total_pairs as f64 / total_secs);
        println!("└─────────────────────────────────────────────────────────────────┘");
    }
    
    fn print_timing_line(&self, label: &str, duration: Duration, total_secs: f64) {
        let pct = 100.0 * duration.as_secs_f64() / total_secs;
        println!("│ {:25} {:>10.3?} ({:>5.1}%)               │", label, duration, pct);
    }
}

/// Scoped timer that adds duration to a reference
pub struct ScopedTimer<'a> {
    start: Instant,
    target: &'a mut Duration,
    enabled: bool,
}

impl<'a> ScopedTimer<'a> {
    pub fn new(target: &'a mut Duration) -> Self {
        Self {
            start: Instant::now(),
            target,
            enabled: is_timing_enabled(),
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        if self.enabled {
            *self.target += self.start.elapsed();
        }
    }
}

/// Macro for timing a block of code
#[macro_export]
macro_rules! time_block {
    ($timings:expr, $field:ident, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        if $crate::timing::is_timing_enabled() {
            $timings.$field += start.elapsed();
        }
        result
    }};
}

// Thread-local storage for timing data
thread_local! {
    static CURRENT_TIMINGS: std::cell::RefCell<Option<OperationTimings>> = const { std::cell::RefCell::new(None) };
}

pub fn start_timing_session() -> OperationTimings {
    OperationTimings::new()
}

pub fn end_timing_session(timings: OperationTimings) {
    if is_timing_enabled() {
        timings.print_report();
    }
}
