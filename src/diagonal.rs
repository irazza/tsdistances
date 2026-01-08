use crate::matrix::{Matrix, WavefrontMatrix};

pub fn diagonal_distance<M: Matrix>(
    a: &[f64],
    b: &[f64],
    init_val: f64,
    sakoe_chiba_band: f64,
    init_lambda: impl Fn(&[f64], &[f64], usize, usize, f64, f64, f64) -> f64 + Copy,
    dist_lambda: impl Fn(&[f64], &[f64], usize, usize, f64, f64, f64) -> f64 + Copy,
    use_upper_bound: bool,
) -> f64 {
    diagonal_distance_::<M>(
        a.len(),
        b.len(),
        init_val,
        sakoe_chiba_band,
        |i, j, x, y, z| init_lambda(a, b, i, j, x, y, z),
        |i, j, x, y, z| dist_lambda(a, b, i, j, x, y, z),
        use_upper_bound,
    )
}

fn diagonal_distance_<M: Matrix>(
    a_len: usize,
    b_len: usize,
    init_val: f64,
    sakoe_chiba_band: f64,
    init_lambda: impl Fn(usize, usize, f64, f64, f64) -> f64,
    dist_lambda: impl Fn(usize, usize, f64, f64, f64) -> f64,
    use_upper_bound: bool,
) -> f64 {
    assert!(a_len <= b_len);
    let mut matrix = WavefrontMatrix::new(a_len, b_len, init_val);

    let upper_bound = if use_upper_bound {
        let min_len = a_len.min(b_len) as f64;
        let mut distance = 0.0;
        for i in 0..min_len as usize {
            distance = dist_lambda(i, i, f64::INFINITY, distance, f64::INFINITY);
        }

        if b_len > a_len {
            for i in min_len as usize..b_len {
                distance = dist_lambda(a_len - 1, i, f64::INFINITY, f64::INFINITY, distance);
            }
        }

        distance
    } else {
        0.0
    };

    let mut i = 0;
    let mut j = 0;
    let mut s = 0;
    let mut e = 0;

    matrix.set_diagonal_cell(0, 0, 0.0);

    let start_coord = M::index_mat_to_diag(0, 0).1;
    let end_coord = M::index_mat_to_diag(a_len, b_len).1;

    let band_size = sakoe_chiba_band * (a_len as f64);

    let mut bound_indexes = [(isize::MIN, isize::MAX), (isize::MIN, isize::MAX)];
    let mut parity = 0;

    for d in 2..(a_len + b_len + 1) {
        matrix.set_diagonal_cell(d, d as isize, init_val);

        let (s_, e_) = if sakoe_chiba_band < 1.0 {
            let mid_coord = start_coord as f64
                + (end_coord as f64 - start_coord as f64) / (a_len + b_len) as f64 * d as f64;
            (
                s.max((mid_coord - band_size).floor() as isize),
                e.min((mid_coord + band_size).ceil() as isize),
            )
        } else {
            (s, e)
        };

        let mut i1: usize = i;
        let mut j1: usize = j;

        let mut s_step = s;

        // Pre init for sakoe chiba band skipped cells

        // TODO: optimize the indices so that no useless cells are set to init_val (especially when using upper bound)
        for k in (s..s_).step_by(2) {
            matrix.set_diagonal_cell(d, k, init_val);
            i1 = i1.wrapping_sub(1);
            j1 += 1;
            s_step += 2;
        }

        let mut first_cell = e_;
        let mut last_cell = s_step;

        let lower_bound_index = bound_indexes[0].0.min(bound_indexes[1].0).max(s_step);
        let upper_bound_index = bound_indexes[0].1.max(bound_indexes[1].1).min(e_);

        let s_step_loop_skips = (lower_bound_index - s_step + 1) / 2;

        let mut s_step = s_step + s_step_loop_skips * 2;
        let e_ = upper_bound_index;

        i1 = i1.wrapping_sub(s_step_loop_skips as usize);
        j1 += s_step_loop_skips as usize;

        for k in (s_step..e_ + 1).step_by(2) {
            let dleft = matrix.get_diagonal_cell(d - 1, k - 1);
            let ddiag = matrix.get_diagonal_cell(d - 2, k);
            let dup = matrix.get_diagonal_cell(d - 1, k + 1);

            let dist = if i1 == 1 || j1 == 1 {
                init_lambda(i1, j1, dleft, ddiag, dup)
            } else {
                dist_lambda(i1, j1, dleft, ddiag, dup)
            };

            if !use_upper_bound || dist <= upper_bound {
                first_cell = first_cell.min(k);
                last_cell = last_cell.max(k);
            }

            matrix.set_diagonal_cell(d, k, dist);

            i1 = i1.wrapping_sub(1);
            j1 += 1;
            s_step += 2;
        }

        let next_parity = (parity + 1) % 2;
        bound_indexes[next_parity].0 = first_cell - 1;
        bound_indexes[next_parity].1 = last_cell + 1;
        parity = next_parity;

        // Post init for sakoe chiba band skipped cells
        for k in (s_step..(e + 1)).step_by(2) {
            matrix.set_diagonal_cell(d, k, init_val);
            i1 = i1.wrapping_sub(1);
            j1 += 1;
        }

        if d <= a_len {
            i += 1;
            s -= 1;
            e += 1;
        } else if d <= b_len {
            j += 1;
            s += 1;
            e += 1;
        } else {
            j += 1;
            s += 1;
            e -= 1;
        }
    }

    let (rx, cx) = M::index_mat_to_diag(a_len, b_len);

    matrix.get_diagonal_cell(rx, cx)
}
