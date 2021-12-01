use rand::prelude::*;
use rand_distr::StandardNormal; // more efficient then Normal.
use std::fmt;
use std::ops;
use std::ops::{Index, IndexMut};

#[derive(PartialEq, Debug)]
pub struct Vector {
    elems: Vec<f64>,
}

impl Vector {
    pub fn len(&self) -> usize {
        self.elems.len()
    }

    pub fn new(n: usize) -> Self {
        let mut elems = Vec::new();
        for _i in 0..n {
            elems.push(thread_rng().sample(StandardNormal));
        }

        Self { elems: elems }
    }
}

impl Index<usize> for Vector {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.elems[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.elems.index_mut(index)
    }
}

impl ops::Add<&Vector> for &Vector {
    type Output = Vector;
    fn add(self, rhs: &Vector) -> Vector {
        let mut res = Vector {
            elems: Vec::<f64>::new(),
        };
        for i in 0..self.len() {
            res.elems.push(0 as f64);
            res[i] = self[i] + rhs[i];
        }

        res
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s: String = "[".to_owned();
        for v in &self.elems {
            s += &(v.to_string() + ",");
        }
        s += "]";
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn index_tests() {
        let mut v = Vector::new(4);
        v[3] = 1.0;
        v[2] = 0.0;
        assert_eq!(v[3], 1.0_f64);
        assert_eq!(v[2], 0.0_f64);
    }
}
