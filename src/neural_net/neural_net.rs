use crate::neural_net::matrix::Matrix;
use crate::neural_net::vector::Vector;

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

struct Network {
    num_layers: usize, //len of sizes
    sizes: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Vector>,
}

impl Network {
    fn new(sizes: Vec<usize>) -> Self {
        let mut weights = Vec::<Matrix>::new();
        let mut biases = Vec::<Vector>::new();

        for i in 0..sizes.len() {
            if i != 0 {
                weights.push(Matrix::new(sizes[i - 1], sizes[i]));
                biases.push(Vector::new(sizes[i]));
            }
        }

        Network {
            num_layers: sizes.len(),
            sizes: sizes,
            weights: weights,
            biases: biases,
        }
    }

    fn feedforward(&self, a: Vector) -> Vector {
        println!("Feeding forward");
        let mut a = Vector::from(a);
        for i in 0..a.len() {
            println!("{}", a[i]);
        }
        println!("Getting new a");
        for i in 0..(self.num_layers - 1) {
            a = &(self.weights[i].times(a)) + &self.biases[i];
        }
        println!("New a has len {}", a.len());

        for i in 0..a.len() {
            a[i] = a[i].sigmoid();
        }
        a
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

    #[test]
    fn test_network_size() {
        let net = Network::new(vec![24, 5, 2]);
        assert_eq!(net.biases[0].len(), 5);
        assert_eq!(net.biases[1].len(), 2);

        assert_eq!(net.weights.len(), 2);
        assert_eq!(net.weights[0].shape(), (24, 5));
        assert_eq!(net.weights[1].shape(), (5, 2));
    }

    #[test]
    fn test_minimal_network() {
        let mut net = Network::new(vec![2, 1]);
        println!("{}", net.biases.len());
        println!("{}", net.weights.len());
        for b in &mut net.biases {
            println!("{}", b.len());
            for i in 0..b.len() {
                b[i] = 0 as f64;
            }
        }

        for w in &mut net.weights {
            println!("{}", w.shape().0);
            println!("{}", w.shape().1);
            let (nr, nc) = w.shape();
            for r in 0..nr {
                for c in 0..nc {
                    w[(r, c)] = 0.0;
                }
            }
        }

        let res = net.feedforward(Vector::from(vec![0., 0.]));
        assert_eq!(res, Vector::from(vec![0.5]))
    }
}
