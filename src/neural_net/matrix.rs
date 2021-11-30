use crate::neural_net;
use crate::neural_net::vector::Vector;
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

    pub fn times(&self, v: &Vector) -> Vector {
        let mut res = Vector::new(self.nr);

        for r in 0..self.nr {
            res[r]=0 as f64;
            for c in 0..self.nc {
                res[r] += self.elems[r][c] * v[c];
            }
        }
        res
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

impl From<Vec<f64>> for Vector {
    fn from(v: Vec<f64>) -> Self {
        let mut res = Vector::new(v.len());
        for i in 0..v.len() {
            res[i] = v[i];
        }
        res
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

    #[test]
    fn dot_test() {
        let mut m = Matrix::new(2, 3);
        let mut v = Vector::new(3);
        for i in 0..2 {
            for j in 0..3 {
                m[(i, j)] = (i * 3 + j) as f64;
                v[j] = j as f64;
            }
        }
        let r = m.times(&v);

        let mut expected = Vector::new(2);
        expected[0] = 5 as f64;
        expected[1] = 14 as f64;

        assert_eq!(r, expected);
    }
}
