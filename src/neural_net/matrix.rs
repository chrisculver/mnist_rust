use rand::prelude::*;
use rand_distr::StandardNormal; // more efficient then Normal.
use std::ops;
use std::ops::{Index, IndexMut};

pub struct Matrix {
    elems: Vec<Vec<f64>>,
    nr: usize,
    nc: usize,
}

impl Matrix {
    pub fn new(nr: usize, nc: usize) -> Self {
        let mut elems = Vec::new();
        for r in 0..nr {
            elems.push(Vec::new());
            for _c in 0..nc {
                elems[r].push(thread_rng().sample(StandardNormal));
            }
        }

        Self {
            elems: elems,
            nr: nr,
            nc: nc,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nr, self.nc)
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<f64>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.elems[index]
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.elems[index.0][index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.elems[index.0].index_mut(index.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn index_tests() {
        let mut m = Matrix::new(2, 2);
        m[(0, 0)] = 0.0_f64;
        assert_eq!(m[(0, 0)], 0.0_f64)
    }

    #[test]
    fn shape_tests() {
        let m = Matrix::new(3, 5);
        let shape = m.shape();
        assert_eq!(shape.0, m.elems.len());
        assert_eq!(shape.1, m.elems[0].len());
    }
}
