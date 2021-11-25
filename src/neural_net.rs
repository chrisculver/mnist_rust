// the sigmoid function
trait Sigmoid {
    fn sigmoid(&self) -> f64;
}

impl Sigmoid for f64 {
    fn sigmoid(&self) -> f64 {
        1.0/(1.0 + (-self).exp())
    }
}


// a neural network layer,
struct Network {
   num_layers: usize, //len of sizes
   sizes: Vec<usize>,
   biases: Vec<
   weights: 
}




#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn some_sigmoid_values() {
        // e^0 = 1 so sigmoid(0.0) = 1/(1+1) = 0.5
        assert_eq!(0.5,(0.0 as f64).sigmoid())
    }
}
