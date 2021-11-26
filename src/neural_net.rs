use rand::prelude::*;
use rand_distr::StandardNormal; // more efficient then Normal.
use std::ops;
use std::ops::{Index, IndexMut};

// the sigmoid function
trait Sigmoid {
    fn sigmoid(self) -> f64;
    fn sigmoid_prime(self) -> f64;
}

impl Sigmoid for f64 {
    fn sigmoid(self) -> f64 {
        1_f64 / (1_f64 + (-self).exp())
    }

    fn sigmoid_prime(self) -> f64 {
        self.sigmoid() * (1_f64 - self.sigmoid())
    }
}

struct Vector {
    elems: Vec<f64>,
}

impl Vector {
    fn len(&self) -> usize {
        self.elems.len()
    }

    fn new(n: usize) -> Self {
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

struct Matrix {
    elems: Vec<Vec<f64>>,
    nr: usize,
    nc: usize,
}

impl Matrix {
    fn new(nr: usize, nc: usize) -> Self {
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
}

struct Network {
    num_layers: usize, //len of sizes
    sizes: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Vector>,
}

impl Network {
    fn new(sizes: Vec<usize>) -> Self {
        Network {
            num_layers: sizes.len(),
            sizes: sizes,
            weights: Vec::<Matrix>::new(),
            biases: Vec::<Vector>::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn some_sigmoid_values() {
        // e^0 = 1 so sigmoid(0.0) = 1/(1+1) = 0.5
        assert_eq!(0.5, (0_f64).sigmoid());

        assert_eq!(0.25, (0_f64).sigmoid_prime());
    }
}
