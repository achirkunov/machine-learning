#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product());
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self { data: vec![0.0; size], shape }
    }

    pub fn randn(shape: Vec<usize>) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| rng.gen::<f64>() - 0.5).collect();
        Self { data, shape }
    }
}
