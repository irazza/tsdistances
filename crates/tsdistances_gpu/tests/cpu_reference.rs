// CPU reference implementations for correctness testing
// These are simple, readable implementations to verify GPU results

/// CPU reference implementation of DTW (Dynamic Time Warping)
pub fn dtw_cpu(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let m = b.len();
    
    // Create DP matrix with infinity values
    let mut dp = vec![vec![f32::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).powi(2);
            dp[i][j] = cost + dp[i - 1][j - 1].min(dp[i - 1][j].min(dp[i][j - 1]));
        }
    }
    
    dp[n][m]
}

/// CPU reference implementation of ERP (Edit distance with Real Penalty)
pub fn erp_cpu(a: &[f32], b: &[f32], gap_penalty: f32) -> f32 {
    let n = a.len();
    let m = b.len();
    
    let mut dp = vec![vec![f32::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    
    // Initialize first row and column
    for i in 1..=n {
        dp[i][0] = dp[i - 1][0] + (a[i - 1] - gap_penalty).abs();
    }
    for j in 1..=m {
        dp[0][j] = dp[0][j - 1] + (b[j - 1] - gap_penalty).abs();
    }
    
    for i in 1..=n {
        for j in 1..=m {
            let match_cost = dp[i - 1][j - 1] + (a[i - 1] - b[j - 1]).abs();
            let delete_a = dp[i - 1][j] + (a[i - 1] - gap_penalty).abs();
            let delete_b = dp[i][j - 1] + (b[j - 1] - gap_penalty).abs();
            dp[i][j] = match_cost.min(delete_a.min(delete_b));
        }
    }
    
    dp[n][m]
}

/// CPU reference implementation of LCSS (Longest Common Subsequence Similarity)
pub fn lcss_cpu(a: &[f32], b: &[f32], epsilon: f32) -> f32 {
    let n = a.len();
    let m = b.len();
    
    let mut dp = vec![vec![0.0f32; m + 1]; n + 1];
    
    for i in 1..=n {
        for j in 1..=m {
            let dist = (a[i - 1] - b[j - 1]).abs();
            if dist <= epsilon {
                dp[i][j] = dp[i - 1][j - 1] + 1.0;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    
    // Convert similarity to distance
    let similarity = dp[n][m];
    let min_len = n.min(m) as f32;
    1.0 - similarity / min_len
}

/// CPU reference implementation of MSM (Move-Split-Merge)
pub fn msm_cpu(a: &[f32], b: &[f32]) -> f32 {
    const C: f32 = 1.0;
    
    fn cost_function(x: f32, y: f32, z: f32) -> f32 {
        C + (((y.min(z) - x).max(x - z.max(x))).max(0.0))
    }
    
    let n = a.len();
    let m = b.len();
    
    let mut dp = vec![vec![f32::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    
    for i in 1..=n {
        for j in 1..=m {
            let match_cost = dp[i - 1][j - 1] + (a[i - 1] - b[j - 1]).abs();
            
            let a_prev = if i > 1 { a[i - 2] } else { 0.0 };
            let b_prev = if j > 1 { b[j - 2] } else { 0.0 };
            
            let split_a = dp[i - 1][j] + cost_function(a[i - 1], a_prev, b[j - 1]);
            let split_b = dp[i][j - 1] + cost_function(b[j - 1], a[i - 1], b_prev);
            
            dp[i][j] = match_cost.min(split_a.min(split_b));
        }
    }
    
    dp[n][m]
}

/// CPU reference implementation of TWE (Time Warp Edit)
pub fn twe_cpu(a: &[f32], b: &[f32], stiffness: f32, penalty: f32) -> f32 {
    let n = a.len();
    let m = b.len();
    
    let mut dp = vec![vec![f32::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    
    let delete_addition = penalty + stiffness;
    
    for i in 1..=n {
        for j in 1..=m {
            let a_prev = if i > 1 { a[i - 2] } else { 0.0 };
            let b_prev = if j > 1 { b[j - 2] } else { 0.0 };
            
            // Deletion in a
            let del_a = dp[i - 1][j] + (a_prev - a[i - 1]).abs() + delete_addition;
            
            // Deletion in b
            let del_b = dp[i][j - 1] + (b_prev - b[j - 1]).abs() + delete_addition;
            
            // Match
            let match_current = (a[i - 1] - b[j - 1]).abs();
            let match_previous = (a_prev - b_prev).abs();
            let match_cost = dp[i - 1][j - 1] + match_current + match_previous 
                + stiffness * (2.0 * ((i as isize - j as isize).abs() as f32));
            
            dp[i][j] = del_a.min(del_b.min(match_cost));
        }
    }
    
    dp[n][m]
}

/// CPU reference implementation of WDTW (Weighted DTW)
pub fn wdtw_cpu(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    let n = a.len();
    let m = b.len();
    
    let mut dp = vec![vec![f32::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    
    for i in 1..=n {
        for j in 1..=m {
            let weight_idx = ((i as i32 - j as i32).abs()) as usize;
            let weight = if weight_idx < weights.len() { weights[weight_idx] } else { 1.0 };
            let cost = (a[i - 1] - b[j - 1]).powi(2) * weight;
            dp[i][j] = cost + dp[i - 1][j - 1].min(dp[i - 1][j].min(dp[i][j - 1]));
        }
    }
    
    dp[n][m]
}

/// CPU reference implementation of ADTW (Amerced DTW)
pub fn adtw_cpu(a: &[f32], b: &[f32], w: f32) -> f32 {
    let n = a.len();
    let m = b.len();
    
    let mut dp = vec![vec![f32::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).powi(2);
            let match_cost = dp[i - 1][j - 1] + cost;
            let insert_cost = dp[i - 1][j] + cost + w;
            let delete_cost = dp[i][j - 1] + cost + w;
            dp[i][j] = match_cost.min(insert_cost.min(delete_cost));
        }
    }
    
    dp[n][m]
}

/// Batch CPU implementation for matrices
pub fn dtw_matrix_cpu(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    a.iter()
        .map(|ai| b.iter().map(|bj| dtw_cpu(ai, bj)).collect())
        .collect()
}

pub fn erp_matrix_cpu(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, gap_penalty: f32) -> Vec<Vec<f32>> {
    a.iter()
        .map(|ai| b.iter().map(|bj| erp_cpu(ai, bj, gap_penalty)).collect())
        .collect()
}

pub fn lcss_matrix_cpu(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, epsilon: f32) -> Vec<Vec<f32>> {
    a.iter()
        .map(|ai| b.iter().map(|bj| lcss_cpu(ai, bj, epsilon)).collect())
        .collect()
}

pub fn msm_matrix_cpu(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    a.iter()
        .map(|ai| b.iter().map(|bj| msm_cpu(ai, bj)).collect())
        .collect()
}

pub fn twe_matrix_cpu(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, stiffness: f32, penalty: f32) -> Vec<Vec<f32>> {
    a.iter()
        .map(|ai| b.iter().map(|bj| twe_cpu(ai, bj, stiffness, penalty)).collect())
        .collect()
}

pub fn wdtw_matrix_cpu(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, weights: &[f32]) -> Vec<Vec<f32>> {
    a.iter()
        .map(|ai| b.iter().map(|bj| wdtw_cpu(ai, bj, weights)).collect())
        .collect()
}

pub fn adtw_matrix_cpu(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, w: f32) -> Vec<Vec<f32>> {
    a.iter()
        .map(|ai| b.iter().map(|bj| adtw_cpu(ai, bj, w)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }
    
    #[test]
    fn test_dtw_cpu_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(dtw_cpu(&a, &b), 0.0, 1e-6));
    }
    
    #[test]
    fn test_dtw_cpu_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 4.0];
        // Last element differs by 1, so squared distance = 1
        let result = dtw_cpu(&a, &b);
        assert!(result > 0.0);
    }
    
    #[test]
    fn test_erp_cpu_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(erp_cpu(&a, &b, 0.0), 0.0, 1e-6));
    }
    
    #[test]
    fn test_lcss_cpu_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        // Identical sequences should have distance 0 (similarity = max)
        assert!(approx_eq(lcss_cpu(&a, &b, 0.1), 0.0, 1e-6));
    }
    
    #[test]
    fn test_adtw_cpu_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(adtw_cpu(&a, &b, 0.1), 0.0, 1e-6));
    }
}
