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

    fn feedforward(&self, mut a: Vec<f64>) -> Vec<f64> {
        for i in 0..(self.num_layers - 1) {
            //    a = self.weights[i].dot(&Vector::from(a)) + self.biases[i];
        }
        for elem in &mut a {
            *elem = elem.sigmoid();
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
}
