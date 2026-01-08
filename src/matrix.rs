pub trait Matrix: Sync + Send {
    fn new(a_len: usize, b_len: usize, init_val: f64) -> Self;
    fn set_diagonal_cell(&mut self, diag_row: usize, diag_offset: isize, value: f64);
    fn get_diagonal_cell(&self, diag_row: usize, diag_offset: isize) -> f64;

    fn index_mat_to_diag(i: usize, j: usize) -> (usize, isize) {
        (i + j, (j as isize) - (i as isize))
    }
    fn index_diag_to_mat(r: usize, c: isize) -> (usize, usize) {
        ((r as isize - c) as usize / 2, (r as isize + c) as usize / 2)
    }

    fn debug_print(&self);
}

pub struct FullMatrix {
    matrix: Vec<Vec<f64>>,
    diag_len: usize,
    a_len: usize,
    b_len: usize,
}

impl Matrix for FullMatrix {
    fn new(a_len: usize, b_len: usize, init_val: f64) -> Self {
        let rowcount = a_len + b_len + 1;
        let diag_len = a_len + b_len + 1;

        Self {
            matrix: vec![vec![init_val; diag_len]; rowcount],
            diag_len,
            a_len,
            b_len,
        }
    }

    fn get_diagonal_cell(&self, diag_row: usize, diag_offset: isize) -> f64 {
        self.matrix[diag_row][self.diag_len.overflowing_add_signed(diag_offset).0 % self.diag_len]
    }

    fn set_diagonal_cell(&mut self, diag_row: usize, diag_offset: isize, value: f64) {
        self.matrix[diag_row]
            [self.diag_len.overflowing_add_signed(diag_offset).0 % self.diag_len] = value;
    }

    fn debug_print(&self) {
        let mut matrix = vec![vec![0.0; self.b_len + 1]; self.a_len + 1];
        let mut row = 0;
        let mut col: isize = 0;

        for (row_idx, matrix_row) in matrix.iter_mut().enumerate() {
            let mut row1 = row;
            let mut col1 = col;

            for cell in matrix_row.iter_mut() {
                *cell = self.get_diagonal_cell(row1, col1);
                row1 += 1;
                col1 += 1;
                print!("{cell:.1}, ");
            }
            println!();
            row = row_idx + 1;
            col -= 1;
        }
    }
}

pub struct WavefrontMatrix {
    diagonal: Vec<f64>,
    mask: usize,
}

impl Matrix for WavefrontMatrix {
    fn new(a_len: usize, _b_len: usize, init_val: f64) -> Self {
        let diag_len = 2 * (a_len + 1).next_power_of_two();

        Self {
            diagonal: vec![init_val; diag_len],
            mask: diag_len - 1,
        }
    }

    #[inline(always)]
    fn get_diagonal_cell(&self, _diag_row: usize, diag_offset: isize) -> f64 {
        self.diagonal[diag_offset as usize & self.mask]
    }

    #[inline(always)]
    fn set_diagonal_cell(&mut self, _diag_row: usize, diag_offset: isize, value: f64) {
        self.diagonal[diag_offset as usize & self.mask] = value;
    }

    fn debug_print(&self) {
        println!("{:?}", self.diagonal)
    }
}

pub struct CheckMatrix {
    full: FullMatrix,
    optim: WavefrontMatrix,
}

impl Matrix for CheckMatrix {
    fn new(a_len: usize, b_len: usize, init_val: f64) -> Self {
        Self {
            full: FullMatrix::new(a_len, b_len, init_val),
            optim: WavefrontMatrix::new(a_len, b_len, init_val),
        }
    }

    fn get_diagonal_cell(&self, diag_row: usize, diag_offset: isize) -> f64 {
        let full = self.full.get_diagonal_cell(diag_row, diag_offset);
        let optim = self.optim.get_diagonal_cell(diag_row, diag_offset);
        if full != optim {
            self.full.debug_print();
            self.optim.debug_print();

            assert_eq!(
                full, optim,
                "Mismatch at ({diag_row}, {diag_offset}): full={full}, optim={optim}"
            );
        }
        full
    }

    fn set_diagonal_cell(&mut self, diag_row: usize, diag_offset: isize, value: f64) {
        // if diag_offset == -14 {
        //     println!("Setting diag_offset: {} to value: {}", diag_offset, value);
        // }

        self.full.set_diagonal_cell(diag_row, diag_offset, value);
        self.optim.set_diagonal_cell(diag_row, diag_offset, value);
    }

    fn debug_print(&self) {
        self.full.debug_print();
        self.optim.debug_print();
    }
}

// let next_power_of_two = 2 * a_len.next_power_of_two();
// let mut diagonal = vec![init_val; next_power_of_two];
// let mask = next_power_of_two - 1;
