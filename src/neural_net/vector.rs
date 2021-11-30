use rand::prelude::*;
use rand_distr::StandardNormal; // more efficient then Normal.
use std::ops;
use std::ops::{Index, IndexMut};

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

impl ops::Add<Vector> for Vector {
    type Output = Vector;
    fn add(self, rhs: Vector) -> Vector {
        let mut res = Vector {
            elems: Vec::<f64>::new(),
        };
        for i in 0..self.len() {
            res[i] = self[i] + rhs[i]
        }

        res
    }
}
